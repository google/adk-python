# Repository Guidelines

## Project Structure & Module Organization
The unified security agent lives in `agents/` with tool adapters under `agents/_tools/`. API and background services sit in `backend/`, while the Flask surface (`app.py`) and Jinja templates reside in `frontend/` and `templates/`. Cloud automation assets live in `cloud_functions/`, shared configs in `config/`, and reusable docs in `docs/`. Tests accompany services in `backend/tests`, `frontend/tests`, and the root `tests/` suite; sample datasets and notebooks remain in `examples/` and `unified_data_api/`.

## Build, Test, and Development Commands
Run `pip install -r requirements.txt` (or `requirements-minimal.txt` for lighter agents) to sync dependencies. Use `./scripts/start_all.sh` to launch backend, Flask UI, and Chainlit UI in one step; `./scripts/stop_all.sh` stops them cleanly. For ad‑hoc interfaces, invoke `python app.py` for the Flask UI, `chainlit run chainlit_app.py` for the chat UI, and `python mcp_server.py` for MCP experimentation.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, type hints, and module-level docstrings as illustrated in `app.py`. Favor `snake_case` for functions/modules and `PascalCase` for classes. Tool IDs exported from `agents/_tools` should use descriptive, hyphen-free names (for example `get_security_statistics`). Keep configuration files declarative (YAML or JSON) and prefer environment variable lookups via `python-dotenv` utilities.

## Testing Guidelines
Pytest drives validation (see `pytest.ini`). Place unit specs beside their service (`backend/tests/test_api.py`) and broader scenarios in `tests/` with names matching `test_*.py`. Markers `@pytest.mark.unit`, `integration`, and `asyncio` gate targeted runs; run `pytest -m unit` for fast feedback and `pytest -v --strict-markers` before merging. Record new fixtures under `tests/conftest.py`, and update `TEST_SUMMARY.md` when coverage-critical flows change.

## Commit & Pull Request Guidelines
Commits use `type: short summary` (for example `feat: Add intelligent query caching`) with emojis only when reinforcing status, mirroring existing history. Squash WIP commits, link issues in the body, and describe interface or schema changes explicitly. Pull requests should summarize the agent impact, list validation commands, attach screenshots for UI shifts, and tag reviewers from security and data owners when BigQuery schemas or IAM scopes evolve.

## Security & Configuration Tips
Store project IDs, dataset/table defaults, and credentials in `.env`; never hard-code secrets. Regenerate service-account keys via `scripts/deployment/` helpers and document rotations in `ROADMAP.md`. Cache warmers in `cache/` and scheduler jobs in `scheduled_queries/` must retain least-privilege roles; coordinate with `launch_iam_swarm.sh` before editing IAM automation.
