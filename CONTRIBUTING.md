# How to contribute

We'd love to accept your patches and contributions to this project.

- [How to contribute](#how-to-contribute)
- [Before you begin](#before-you-begin)
  - [Sign our Contributor License Agreement](#sign-our-contributor-license-agreement)
  - [Review our community guidelines](#review-our-community-guidelines)
- [Contribution workflow](#contribution-workflow)
  - [Finding Issues to Work On](#finding-issues-to-work-on)
  - [Requirement for PRs](#requirement-for-prs)
  - [Large or Complex Changes](#large-or-complex-changes)
  - [Testing Requirements](#testing-requirements)
  - [Unit Tests](#unit-tests)
  - [Manual End-to-End (E2E) Tests](#manual-end-to-end-e2e-tests)
  - [Documentation](#documentation)
  - [Development Setup](#development-setup)
  - [Code reviews](#code-reviews)

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Contribution workflow

### Finding Issues to Work On

- Browse issues labeled **`good first issue`** (newcomer-friendly) or **`help wanted`** (general contributions).
- For other issues, please kindly ask before contributing to avoid
  duplication.

### Requirement for PRs

- All PRs, other than small documentation or typo fixes, should have an Issue
  associated. If a relevant issue doesn't exist, please create one first or
  you may instead describe the bug or feature directly within the PR
  description, following the structure of our issue templates.
- Small, focused PRs. Keep changes minimal—one concern per PR.
- For bug fixes or features, please provide logs or screenshot after the fix
  is applied to help reviewers better understand the fix.
- Please include a `testing plan` section in your PR to describe how you
  will test. This will save time for PR review. See `Testing Requirements`
  section for more details.

### Large or Complex Changes

For substantial features or architectural revisions:

- Open an Issue First: Outline your proposal, including design considerations
  and impact.
- Gather Feedback: Discuss with maintainers and the community to ensure
  alignment and avoid duplicate work

### Testing Requirements

To maintain code quality and prevent regressions, all code changes must include
comprehensive tests and verifiable end-to-end (E2E) evidence.

#### Unit Tests

Please add or update unit tests for your change. Please include a summary of
passed `pytest` results.

Requirements for unit tests:

- **Coverage:** Cover new features, edge cases, error conditions, and typical
  use cases.
- **Location:** Add or update tests under `tests/unittests/`, following
  existing naming conventions (e.g., `test_<module>_<feature>.py`).
- **Framework:** Use `pytest`. Tests should be:
  - Fast and isolated.
  - Written clearly with descriptive names.
  - Free of external dependencies (use mocks or fixtures as needed).
- **Quality:** Aim for high readability and maintainability; include
  docstrings or comments for complex scenarios.

#### Manual End-to-End (E2E) Tests

Manual E2E tests ensure integrated flows work as intended. Your tests should
cover all scenarios. Sometimes, it's also good to ensure relevant functionality
is not impacted.

Depending on your change:

- **ADK Web:**

  - Use the `adk web` to verify functionality.
  - Capture and attach relevant screenshots demonstrating the UI/UX changes
    or outputs.
  - Label screenshots clearly in your PR description.

- **Runner:**

  - Provide the testing setup. For example, the agent definition, and the
    runner setup.
  - Execute the `runner` tool to reproduce workflows.
  - Include the command used and console output showing test results.
  - Highlight sections of the log that directly relate to your change.

### Documentation

For any changes that impact user-facing documentation (guides, API reference,
tutorials), please open a PR in the
[adk-docs](https://github.com/google/adk-docs) repository to update the relevant
part before or alongside your code PR.

## Development Setup

1. **Clone the repository:**

   ```shell
   gh repo clone google/adk-python -- -b v2
   cd adk-python
   ```

1. **Install uv:**

   Check out
   [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

1. **Setup Development Tools:**

   We use `pre-commit` for code formatting and license enforcement,
   `tox` with `tox-uv` for isolated multi-version testing, and
   `addlicense` for Apache 2.0 license headers.

   ```shell
   uv tool install pre-commit
   uv tool install tox --with tox-uv
   ```

   Optionally, install Google's `addlicense` tool for license header
   checks (requires Go):

   ```shell
   go install github.com/google/addlicense@latest
   ```

   If `addlicense` is not installed, the pre-commit hook will be
   skipped and CI will catch missing headers.

   Install the git hooks to automatically format and check your code
   before committing:

   ```shell
   pre-commit install
   ```

   The pre-commit hooks run `isort`, `pyink`, `addlicense`, and
   `mdformat` automatically on each commit.

1. **Create virtual environment and install dependencies:**

   ```shell
   uv venv --python "python3.11" ".venv"
   source .venv/bin/activate
   uv sync --all-extras
   ```

1. **Run unit tests locally (Fast):**

   If you just want to run tests quickly while developing, run `pytest`:

   ```shell
   pytest ./tests/unittests
   ```

1. **Run multi-version unit tests (Required before PR):**

   ADK guarantees compatibility across Python versions. You must run the full test suite across all supported versions using `tox`. This will execute tests in pristine, isolated environments.

   ```shell
   tox
   ```

   _(Note: `uv` will automatically download any Python interpreters you are missing!)_

1. **Auto-format the code:**

   If you installed the git hooks in Step 3, this happens automatically on commit. To run it manually across all files:

   ```shell
   pre-commit run --all-files
   ```

1. **Build the wheel file:**

   ```shell
   uv build
   ```

1. **Test the locally built wheel file:** Have a simple testing folder setup as
   mentioned in the
   [quickstart](https://google.github.io/adk-docs/get-started/quickstart/).

   Then following below steps to test your changes:

   Create a clean venv and activate it:

   ```shell
   VENV_PATH=~/venvs/adk-quickstart
   ```

   ```shell
   command -v deactivate >/dev/null 2>&1 && deactivate
   ```

   ```shell
   rm -rf $VENV_PATH \
     && python3 -m venv $VENV_PATH \
     && source $VENV_PATH/bin/activate
   ```

   Install the locally built wheel file:

   ```shell
   pip install dist/google_adk-<version>-py3-none-any.whl
   ```

## Contributing Resources

[Contributing folder](https://github.com/google/adk-python/tree/main/contributing)
has resources that are helpful for contributors.

## AI-Assisted Development

This repo includes built-in skills for AI coding agents
(Antigravity, Gemini CLI, and others) to help with ADK development:

- **`setup-dev-env`** — Set up the local development environment:
  install dependencies, configure pre-commit hooks, and verify
  the setup.

- **`adk-debug`** — Debug ADK agents: inspect sessions, trace event
  flows, check LLM requests/responses, diagnose tool call issues.
  Supports both `adk web` (browser UI) and `adk run` (CLI) workflows.

- **`adk-workflow`** — Build graph-based workflow agents: function
  nodes, LLM agent nodes, edge patterns, routing, parallel processing
  (fan-out and ParallelWorker), human-in-the-loop, state management,
  and best practices. Includes reference docs and tested samples.

These skills are in `.agents/skills/` and are automatically available
when using compatible AI coding tools in this repo.

The `AGENTS.md` file provides additional project context that can
be used as LLM input.
