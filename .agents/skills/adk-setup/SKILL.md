---
name: adk-setup
description: >-
  Sets up a local ADK Python development environment in a git clone of the
  open-source adk-python repository: a uv virtual environment, all dependency
  extras, pre-commit hooks, and a first unit-test run. Runs only when
  explicitly requested, never on its own. Use when asked to set up, bootstrap,
  or repair a development checkout, install project dependencies, fix a
  missing or broken .venv, or prepare a machine for contributing a pull
  request. Don't use for debugging a running agent (use adk-debug), for commit
  and pull-request mechanics (use adk-git), or for re-running formatters on a
  checkout that is already set up (pre-commit run --all-files).
disable-model-invocation: true
---

# ADK Python Development Setup

Set up a working `adk-python` checkout: a uv-managed virtual environment with
every dependency extra, pre-commit hooks, and a green unit-test run.

## Prerequisites

1. **Python.** ADK supports 3.10 through 3.14 (`requires-python = ">=3.10"` in
   `pyproject.toml`). These steps use 3.11, the version the repo's own tooling
   defaults to.

   ```bash
   python3 --version
   ```

2. **uv.** Dependencies are pinned in `uv.lock`; a hand-rolled `pip`/`venv`
   environment will not reproduce the locked versions.

   ```bash
   uv --version
   ```

   Install it if missing:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Setup

Run these from the repository root.

1. **Create and activate the virtual environment:**

   ```bash
   uv venv --python "python3.11" ".venv"
   source .venv/bin/activate
   ```

2. **Install every dependency extra:**

   ```bash
   uv sync --all-extras
   ```

   `--all-extras` is what pulls in the `test` and `dev` extras. Syncing only
   the default dependencies leaves `pytest` and the linters uninstalled.

3. **Install pre-commit and tox as standalone tools:**

   ```bash
   uv tool install pre-commit
   uv tool install tox --with tox-uv
   ```

   Both are also in the `dev` extra; installing them as uv tools puts them on
   PATH so the git hook and multi-version test runs work without an activated
   venv.

4. **Install `addlicense` (optional, requires Go):**

   ```bash
   go install github.com/google/addlicense@latest
   ```

   Without it the pre-commit hook prints `Warning: addlicense not installed,
   skipping` and passes, so a missing Go toolchain does not block setup — CI
   catches missing license headers instead.

5. **Install the git hooks:**

   ```bash
   pre-commit install
   ```

   On each commit the hooks auto-format with `isort`, `pyink`, and `mdformat`,
   then run `ruff`, `addlicense`, `codespell`, and the repo's own compliance
   checks.

6. **Verify the environment:**

   ```bash
   pytest tests/unittests -n auto
   ```

   A green run here is the success criterion for setup. `-n auto` needs
   `pytest-xdist`, which arrives with the `test` extra in step 2.

## Key commands

| Task                                 | Command                                            |
| :----------------------------------- | :------------------------------------------------- |
| Run unit tests                       | `pytest tests/unittests`                           |
| Run unit tests in parallel           | `pytest tests/unittests -n auto`                   |
| Run one test file                    | `pytest tests/unittests/agents/test_base_agent.py` |
| Run tests on every supported Python  | `tox` (uv downloads any missing interpreters)      |
| Format and lint everything           | `pre-commit run --all-files`                       |
| Launch the web UI                    | `adk web {agents_dir}`                             |
| Run an agent from the CLI            | `adk run {agent_dir}`                              |
| Build the wheel                      | `uv build`                                         |
