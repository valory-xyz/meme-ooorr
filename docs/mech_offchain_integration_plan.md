# Mech-interact off-chain integration plan (meme-ooorr)

Scope: wire meme-ooorr onto mech-interact v0.32.4 (off-chain HTTP request/response support). Default behaviour stays on-chain; operators opt in per service.yaml by flipping `use_offchain: true` inside `MECH_MARKETPLACE_CONFIG` **and** setting an endpoint for the executor to reach — either `offchain_url` (static per-service) or `use_dynamic_mech_selection: true` (discover URL from the on-chain manifest). With both left at their shipped defaults (`null` and `false`), `MechMarketplaceConfig.__post_init__` raises at startup.

Reference implementations that ship the same shape:

- trader `feat/offchain-mech-integration` — commit `471646e7a` ("feat(offchain): wire mech-interact off-chain branch through Trader FSM").
- optimus `feat/optimus-offchain-mech` — commits `0bc4cc79` and follow-up `54bf9fcc` ("fix(offchain): register new mech-interact contracts in both agents").

## Design summary

mech-interact's off-chain path splits the request round into three new terminal states:

- `FinishedOffchainMechRequestRound` — HTTP 200, response is ready to poll via HTTP (skip on-chain settlement).
- `FinishedOffchainMechDepositNeededRound` — structured 402, an `approve + BalanceTracker.depositFor` multisend was built and must land on-chain before the request can be replayed.
- `FailedOffchainMechRequestRound` — every ranked mech was tried and none succeeded.

`MechResponseBehaviour` already branches internally on `use_offchain` (mech-interact `behaviours/response.py:640`) and polls `/fetch_offchain_info` in the off-chain case. So the response leg needs no meme-ooorr-side FSM changes — the existing `MechResponseRound → PostMechResponseRound` transition still fires.

meme-ooorr uses `PostTxDecisionMakingRound + PostTxDecisionMakingBehaviour` as the tx-submitter dispatcher (same role as `PostTxSettlementRound` in trader/optimus). The deposit's `tx_submitter` is set to `OFFCHAIN_DEPOSIT_TX_SUBMITTER` (mech-interact `states/request.py:44`) inside `OffchainRequestExecutor` so it doesn't collide with any existing submitter id. We add one entry to the submitter→event dict.

## Concrete changes

### 1. Vendor updated mech_interact_abci + 3 new contracts

- Replace `packages/valory/skills/mech_interact_abci/` with the v0.32.4 tree (adds `behaviours/offchain_request.py`, `behaviours/offchain_response.py`, three new degenerate rounds under `states/final_states.py`, seven off-chain knobs on `MechMarketplaceConfig`, `OFFCHAIN_DEPOSIT_TX_SUBMITTER` sentinel at `states/request.py:44`).
- Vendor three new contract packages (mirrors trader/optimus):
  - `packages/valory/contracts/mech_marketplace/`
  - `packages/valory/contracts/balance_tracker_fixed_price_native/`
  - `packages/valory/contracts/balance_tracker_fixed_price_token/`
- `packages/packages.json`:
  - Bump `skill/valory/mech_interact_abci/0.1.0` hash to v0.32.4 (`bafybeici5tsgtlypnyrnp5colj7katnsyjfdzt6js4xdbnpdms4232muje`).
  - Add three new `contract/valory/*` entries under `third_party`.
- `packages/valory/skills/memeooorr_chained_abci/skill.yaml` — bump the sibling `mech_interact_abci` skill dep hash to match.

### 2. Agent contract registration

Add ONLY the three new contracts to `packages/valory/agents/memeooorr/aea-config.yaml:contracts:`:

- `valory/mech_marketplace`
- `valory/balance_tracker_fixed_price_native`
- `valory/balance_tracker_fixed_price_token`

Pre-existing mech contract registration (`agent_mech`, `mech`, `mech_mm`, `mech_marketplace_legacy`, `agent_registry`, `erc20`, `ierc1155`, `nvm_balance_tracker_*`, Nevermined set) is left as-is per user direction. Those live in `packages.json` third_party already and the on-chain path is assumed to work in production today.

### 3. Config surface — off-chain knobs on `MECH_MARKETPLACE_CONFIG`

Extend the env-default dict in both locations with seven off-chain fields, defaulting `use_offchain: false` so today's FSM entries are unchanged. Timeout note: v0.32.4 keeps `MechResponseRound` on the shared `MechInteractEvent.ROUND_TIMEOUT` (rc2's briefly-lived dedicated `RESPONSE_ROUND_TIMEOUT` was reverted before the tag). meme-ooorr overrides `MechInteractEvent.ROUND_TIMEOUT` in `MemeooorrChainedSkillAbciApp.event_to_timeout` at `memeooorr_chained_abci/models.py` via `round_timeout_seconds × MULTIPLIER_MECH` (`MULTIPLIER_MECH = 20`), giving 600s effective — comfortably above v0.32.4's ≥330s poll-budget guard for the off-chain path. Same wiring was in place under v0.32.2, so the response-round timeout is unchanged across the bump.

- `packages/dvilela/services/memeooorr/service.yaml:108` (the deployed service.yaml override)
- `packages/valory/agents/memeooorr/aea-config.yaml:247` (the agent's own `mech_marketplace_config` under the mech_interact_abci skill block)

New fields:

- `use_offchain: false`
- `offchain_url: null`
- `offchain_deposit_target_calls: 10`
- `auto_deposit_cap_per_cycle: 5000000` (5 USDC on base at 6 decimals — meme-ooorr pays mech in USDC)
- `offchain_poll_interval_seconds: 3.0`
- `offchain_poll_timeout_seconds: 300.0`
- `offchain_failover_max_retries: 2`

The full dict must be present in the env default so an operator override that only sets `use_offchain: true` still validates against `MechMarketplaceConfig.__post_init__` (mech-interact `models.py:274-307`).

### 4. FSM wiring — `memeooorr_chained_abci/composition.py`

Four new edges under `abci_app_transition_mapping`:

- `MechFinalStates.FinishedOffchainMechRequestRound` → `MechResponseStates.MechResponseRound` — happy path, skip settlement, poll HTTP.
- `MechFinalStates.FinishedOffchainMechDepositNeededRound` → `TransactionSettlementAbci.RandomnessTransactionSubmissionRound` — settle the `approve + depositFor` multisend.
- `MemeooorrAbci.FinishedForOffchainMechDepositSettledRound` → `MechRequestStates.MechRequestRound` — deposit landed, re-enter the request round so `OffchainRequestExecutor._retry_pending` re-POSTs the cached `offchain_pending_request`.
- `MechFinalStates.FailedOffchainMechRequestRound` → `MemeooorrAbci.FailedMechRequestRound` — mirror the existing `FinishedMechRequestSkipRound` bail path so failure bookkeeping (quarantine, tool penalty, etc.) fires identically for both paths.

### 5. Event + degenerate round in `memeooorr_abci`

- `memeooorr_abci/rounds.py` — add `Event.OFFCHAIN_MECH_DEPOSIT_SETTLED = "offchain_mech_deposit_settled"` to the `Event(Enum)` at line 73.
- `memeooorr_abci/rounds.py` — add `class FinishedForOffchainMechDepositSettledRound(DegenerateRound)`, register it in `final_states` and the top-level rounds set.
- Update the `MemeooorrAbciApp` docstring's "Final states" list (regen via `make generators`).

### 6. `PostTxDecisionMakingBehaviour` dispatch

`memeooorr_abci/behaviour_classes/chain.py:862-867` — one new entry in `submitter_to_event`:

```
from packages.valory.skills.mech_interact_abci.states.request import (
    OFFCHAIN_DEPOSIT_TX_SUBMITTER,
)
...
OFFCHAIN_DEPOSIT_TX_SUBMITTER: Event.OFFCHAIN_MECH_DEPOSIT_SETTLED.value,
```

The existing `Event.NONE` fall-through + `logger.error("Unknown tx_submitter…")` stays as-is.

### 7. Tests

- `memeooorr_chained_abci/tests/test_composition.py` — parametrise each new edge individually. Flat set assertions would let a silent target swap pass (this bit trader in review).
- `memeooorr_abci/tests/` — add:
  - a positive test that `PostTxDecisionMakingBehaviour` dispatches `OFFCHAIN_DEPOSIT_TX_SUBMITTER` to `Event.OFFCHAIN_MECH_DEPOSIT_SETTLED`;
  - a negative test that an unknown submitter still returns `Event.NONE` and logs the error at ERROR (protects the fall-through against a future refactor turning the dict lookup into a KeyError).
- Off-path traversal is intentionally not tested at the composition level: `abci_app_transition_mapping` is a static dict at import time, so `use_offchain=false` doesn't gate any edge — whether the off-chain terminals are ever reached is decided inside the vendored mech_interact_abci behaviour, which owns that test.

### 8. Regen + lock

```
make fix-abci-app-specs
make generators
autonomy packages lock
autonomy packages lock --check
```

Then `autonomy push-all` before pushing the branch (per the always-push-after-package-change rule).

### 9. CI parity

Read `.github/workflows/*.yaml` — run every `tox -e` env locally, including `check-hash`, `check-packages`, `check-doc-hashes`, skill test envs, mypy, pylint, black, isort, flake8, darglint.

## Explicit non-goals

- Not registering pre-existing mech contracts (`agent_mech`, `mech`, `mech_mm`, `mech_marketplace_legacy`, `agent_registry`, `erc20`, `ierc1155`, `nvm_balance_tracker_*`, Nevermined set) at the agent aea-config level. That gap exists in main and is a separate hygiene concern.
- Not touching `funds_manager`, `agent_performance_summary_abci`, `agent_db_abci`, or any twitter / genai connection.
- Not rolling out — `use_offchain` stays `false` in the env default; ops flips per deployment via `MECH_MARKETPLACE_CONFIG`.

## Rollback

Set `use_offchain: false` in the operator env override for `MECH_MARKETPLACE_CONFIG`. The on-chain FSM entries are unchanged; nothing else from the v0.32.4 skill bump alters on-chain behaviour (the response-round timeout stays on the shared `MechInteractEvent.ROUND_TIMEOUT` we already override to 600s — see the config-surface note above — same as under v0.32.2). No code removal is needed to revert to today's routing.
