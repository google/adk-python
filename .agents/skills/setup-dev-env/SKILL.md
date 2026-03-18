---
name: setup-dev-env
description: Set up a local development environment for the ADK Python project. Use when the user wants to get started developing, set up their environment, install dependencies, or prepare for contributing.
disable-model-invocation: true
---

Set up the local development environment for ADK Python.

## Prerequisites

Check the following before proceeding:

1. **Python 3.11+**

   ```bash
   python3 --version
   ```

2. **uv package manager** (required — do not use pip/venv directly)
   ```bash
   uv --version
   ```
   If not installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Setup Steps

Run these commands from the project root:

3. **Create and activate a virtual environment:**
   ```bash
   uv venv --python "python3.11" ".venv"
   source .venv/bin/activate
   ```

4. **Install all dependencies for development:**
   ```bash
   uv sync --all-extras
   ```

5. **Install development tools:**
   ```bash
   uv tool install pre-commit
   uv tool install tox --with tox-uv
   ```

6. **Install addlicense (requires Go):**
   ```bash
   go version && go install github.com/google/addlicense@latest
   ```
   If Go is not installed, tell the user:
   "Go is required for the addlicense tool. Please install Go from
   https://go.dev/dl/ and then re-run `/setup-dev-env` to complete
   the setup."

7. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

8. **Verify everything works by running tests locally (Fast):**
   ```bash
   pytest tests/unittests -x -q --no-header 2>&1 | tail -5
   ```

## Troubleshooting

- If `uv run` fails with `401 Unauthorized` against `us-python.pkg.dev/artifact-foundry-prod/...`, run `gpkg setup` to fix credentials before retrying.

## Key Commands Reference

| Task                                 | Command                                           |
| :----------------------------------- | :------------------------------------------------ |
| Run unit tests (Fast)                | `pytest tests/unittests`                          |
| Run tests across all Python versions | `tox`                                             |
| Format codebase                      | `pre-commit run --all-files`                      |
| Run tests in parallel                | `pytest tests/unittests -n auto`                  |
| Run specific test file               | `pytest tests/unittests/agents/test_llm_agent.py` |
| Launch web UI                        | `adk web path/to/agents_dir`                      |
| Run agent via CLI                    | `adk run path/to/my_agent`                        |
| Build wheel                          | `uv build`                                        |
