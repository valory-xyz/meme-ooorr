# Meme-ooorr

Autonomous agent service built on open-aea / open-autonomy (Valory stack).

## Project structure

- `packages/dvilela/` — dev packages (author: `dvilela`)
  - `skills/memeooorr_abci` — main ABCI skill
  - `skills/memeooorr_chained_abci` — chained ABCI skill
  - `contracts/meme_factory`, `contracts/service_registry` — on-chain contracts
  - `connections/twikit`, `connections/mirror_db`, `connections/tweepy` — custom connections
  - `agents/memeooorr`, `services/memeooorr` — agent & service definitions
- `packages/valory/` — third-party / upstream Valory packages
  - `skills/agent_db_abci`, `skills/agent_performance_summary_abci` — valory-authored dev skills
- `packages/open_aea/` — upstream open-aea packages
- `tox.ini` — lint, type-check, and test environments

## Dependencies

- open-aea 2.1.0, open-autonomy 0.21.16
- Python >=3.10, <3.15
- tomte 0.6.2 (linting / testing meta-package)

## Running commands

All commands (`tox`, `autonomy`, `pytest`, etc.) must be run through uv:

```bash
uv run tox -e black-check
uv run autonomy packages lock
uv run pytest packages/dvilela/skills/memeooorr_abci/tests
```

## Package management

**Do NOT use `autonomy packages sync --all`.** Use only `--update-packages`:

```bash
uv run autonomy packages sync --update-packages
```

After any package file change (including lint-only changes), re-lock hashes:

```bash
uv run autonomy packages lock
```

Package hashes live in `packages/packages.json`.

## Running tests

Tests run via tox. To run all package tests locally:

```bash
uv run tox -e py3.12-linux
```

Individual package tests:

```bash
uv run pytest packages/dvilela/skills/memeooorr_abci/tests
```

## Linting

```bash
uv run tox -e black-check   # formatting check
uv run tox -e isort-check    # import order check
uv run tox -e flake8         # style
uv run tox -e mypy           # type checking
uv run tox -e pylint         # static analysis
```

Fix formatting:

```bash
uv run tox -e black && uv run tox -e isort
```

## Hash management

After editing any file under `packages/`, regenerate and lock hashes:

```bash
uv run autonomy packages lock
```

To check hashes without modifying:

```bash
uv run autonomy packages lock --check
```

## Key conventions

- `PYTHONPATH` must point to the repo root so `packages.*` imports resolve locally
- tox uses `allowlist_externals` (not `whitelist_externals` — tox 4)
- `asyncio_mode=strict` in pytest config — use `@pytest.mark.asyncio` explicitly
- mypy targets Python 3.10 syntax
- pylint has many codes disabled — see `tox.ini [testenv:pylint]` for the full list
