# FSM Audit Report

**Scope:** All FSM skills in `packages/`

**Dev packages:**
- `packages/dvilela/skills/memeooorr_abci/`
- `packages/dvilela/skills/memeooorr_chained_abci/`
- `packages/valory/skills/agent_db_abci/`
- `packages/valory/skills/agent_performance_summary_abci/`

**Third-party packages:**
- `packages/valory/skills/registration_abci/`
- `packages/valory/skills/reset_pause_abci/`
- `packages/valory/skills/transaction_settlement_abci/`
- `packages/valory/skills/termination_abci/`
- `packages/valory/skills/mech_interact_abci/`

**Dev/Third-Party Classification:** Based on `packages/packages.json`
**Date:** 2026-03-09

## CLI Tool Results

| Tool | Result |
|------|--------|
| `fsm-specs` (all 4 dev skills) | **PASS** |
| `handlers` | **FAIL** — `funds_manager` missing handler file (third-party, not in scope) |
| `dialogues` | **FAIL** — `agent_performance_summary_abci` missing `abci_dialogues` in `skill.yaml` (dev) |
| `docstrings` | **FAIL** — `memeooorr_abci/rounds.py` and `agent_db_abci/rounds.py` need updating (dev) |

---

## Critical Findings

### C5: Invalid Event String Default in PostTxDecisionMakingBehaviour
- **File:** `packages/dvilela/skills/memeooorr_abci/behaviour_classes/chain.py:856`
- **Issue:** The `event` variable is initialized to `"None"` (capital N). If `tx_submitter` doesn't match any of the three expected round IDs, the payload is sent with `"None"`. In `PostTxDecisionMakingRound` (extends `EventRoundBase`), `end_block()` at `rounds.py:232` calls `Event(self.most_voted_payload)`, which tries `Event("None")`. The Event enum defines `NONE = "none"` (lowercase), so `Event("None")` raises `ValueError`, crashing the agent.
- **Code:**
  ```python
  # behaviour_classes/chain.py:856
  event = "None"  # BUG: "None" != Event.NONE.value ("none")

  if self.synchronized_data.tx_submitter == CallCheckpointBehaviour.matching_round.auto_round_id():
      event = Event.DONE.value
  if self.synchronized_data.tx_submitter == ActionPreparationBehaviour.matching_round.auto_round_id():
      event = Event.ACTION.value
  if self.synchronized_data.tx_submitter == MechRequestBehaviour.matching_round.auto_round_id():
      event = Event.MECH.value

  # rounds.py:232 — consumer side:
  event = Event(self.most_voted_payload)  # Event("None") → ValueError!
  ```
- **Fix:**
  ```python
  event = Event.DONE.value  # safe default
  ```

## High Findings

No findings.

## Medium Findings

### M2: Unused Event Enum Members
- **File:** `packages/dvilela/skills/memeooorr_abci/rounds.py:73-94`
- **Issue:** Six Event enum members are never referenced anywhere in the codebase (not in `transition_function`, `end_block()`, or behaviours): `REFINE`, `NOT_ENOUGH_FEEDBACK`, `NO_MEMES`, `TO_DEPLOY`, `TO_ACTION_TWEET`, `NONE`. These are dead config from earlier versions.
- **Code:**
  ```python
  REFINE = "refine"                        # unused
  NOT_ENOUGH_FEEDBACK = "not_enough_feedback"  # unused
  NO_MEMES = "no_memes"                    # unused
  TO_DEPLOY = "to_deploy"                  # unused
  TO_ACTION_TWEET = "to_action_tweet"      # unused
  NONE = "none"                            # unused (becomes used if C5 fix uses it)
  ```
- **Fix:** Remove unused members. Note: `NONE` will become used if the C5 fix defaults to `Event.NONE.value`.

### M-CLI-1: Missing `abci_dialogues` in skill.yaml
- **File:** `packages/valory/skills/agent_performance_summary_abci/skill.yaml`
- **Issue:** `autonomy analyse dialogues` reports that `abci_dialogues` is not defined. This can cause runtime issues when the skill participates in ABCI consensus within the composed service.
- **Fix:** Add the standard `abci_dialogues` definition to `skill.yaml`.

### M-CLI-2: Docstring / Transition Function Drift
- **Files:** `packages/dvilela/skills/memeooorr_abci/rounds.py`, `packages/valory/skills/agent_db_abci/rounds.py`
- **Issue:** `autonomy analyse docstrings` reports AbciApp docstrings are out of sync with the actual `transition_function` definitions.
- **Fix:** Run `autonomy analyse docstrings --update` to regenerate.

## Low Findings

### L1: Dead Code — Overwritten Variable Assignment
- **File:** `packages/valory/skills/agent_db_abci/agent_db_client.py:361`
- **Issue:** `payload` is assigned on line 361 and immediately overwritten on line 362.
- **Code:**
  ```python
  payload = {f"{value_type}_value": value}      # line 361 — dead
  payload = {                                     # line 362 — overwrites
      "agent_id": agent_instance.agent_id,
      "attr_def_id": attribute_def.attr_def_id,
      f"{value_type}_value": value,
  }
  ```
- **Fix:** Delete line 361.

### L1: Dead Code — Unused `@classmethod` Generator
- **File:** `packages/valory/skills/agent_db_abci/agents_fun_db.py:96-106`
- **Issue:** `AgentsFunDB.register()` is a `@classmethod` using `yield from` (making it a generator) but is never called. The `AgentsFunDatabase.initialize()` method is used instead. Additionally, `@classmethod` combined with generator semantics is inherently problematic — calling `cls.register()` returns a generator object rather than executing the body.
- **Fix:** Remove this dead method.

### L1: Type Annotation Mismatch
- **File:** `packages/valory/skills/agent_db_abci/behaviours.py:63`
- **Issue:** Type annotation declares `Set` but the value is a `list`.
- **Code:**
  ```python
  behaviours: Set[Type[BaseBehaviour]] = [AgentDBBehaviour]  # list, not set
  ```
- **Fix:** Change to `behaviours: Set[Type[BaseBehaviour]] = {AgentDBBehaviour}`

## Third-Party Impact Assessment

All five third-party skills passed critical and high-severity checks. No upstream issues to report.

| Third-Party Skill | Finding | Composed Into Service? | Critical Impact on Service? | Notes |
|-|-|-|-|-|
| `registration_abci` | None | Yes | No | — |
| `reset_pause_abci` | None | Yes | No | — |
| `transaction_settlement_abci` | None | Yes | No | — |
| `termination_abci` | None | Yes | No | — |
| `mech_interact_abci` | None | Yes | No | — |
| `funds_manager` | CLI: missing handler file | No (vendored only) | No | Third-party packaging issue |

## Summary

| Severity | Count (dev only) |
|----------|------------------|
| Critical | 1                |
| High     | 0                |
| Medium   | 3                |
| Low      | 3                |

## Notes
- **Composition chain verified:** `memeooorr_chained_abci` has 18 final→initial state mappings, all correct. `cross_period_persisted_keys` properly defined. TerminationAbciApp background app correctly configured.
- **Clean checks (no issues):** C1 (shared mutable refs), C2 (operator precedence), C3 (transition completeness), C4 (dead timeouts), H1 (background apps), H2 (chain completeness), H3 (resource lifecycle), M1 (payload class), M3 (db conditions), M4 (thread join), M5 (key mismatch).
- **Test coverage:** No test files exist for `agent_db_abci` or `agent_performance_summary_abci`. T-category checks (T1-T6) were not applicable for those skills.
- **False positives excluded:** `ROUND_TIMEOUT` in library skills (convention), `synchronized_data.update()` return not captured (mutates in-place), mutable class-level dicts in behaviours (single-instance).
- The C5 bug is latent — it only triggers when `tx_submitter` doesn't match any of the three expected round IDs (e.g., after a new tx submitter is added without updating this behaviour). When triggered, it crashes the agent with `ValueError`.
