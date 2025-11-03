# How to contribute

We'd love to accept your patches and contributions to this project.

-   [How to contribute](#how-to-contribute)
-   [Before you begin](#before-you-begin)
    -   [Sign our Contributor License Agreement](#sign-our-contributor-license-agreement)
    -   [Review our community guidelines](#review-our-community-guidelines)
-   [Contribution workflow](#contribution-workflow)
    -   [Finding Issues to Work On](#finding-issues-to-work-on)
    -   [Requirement for PRs](#requirement-for-prs)
    -   [Large or Complex Changes](#large-or-complex-changes)
    -   [Testing Requirements](#testing-requirements)
    -   [Unit Tests](#unit-tests)
    -   [Manual End-to-End (E2E) Tests](#manual-end-to-end-e2e-tests)
    -   [Documentation](#documentation)
    -   [Development Setup](#development-setup)
    -   [Code reviews](#code-reviews)

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

-   Browse issues labeled **`good first issue`** (newcomer-friendly) or **`help
    wanted`** (general contributions).
-   For other issues, please kindly ask before contributing to avoid
    duplication.

### Requirement for PRs

-   All PRs, other than small documentation or typo fixes, should have a Issue
    associated. If a relevant issue doesn't exist, please create one first or
    you may instead describe the bug or feature directly within the PR
    description, following the structure of our issue templates.
-   Small, focused PRs. Keep changes minimal—one concern per PR.
-   For bug fixes or features, please provide logs or screenshot after the fix
    is applied to help reviewers better understand the fix.
-   Please include a `testing plan` section in your PR to talk about how you
    will test. This will save time for PR review. See `Testing Requirements`
    section for more details.

### Large or Complex Changes

For substantial features or architectural revisions:

-   Open an Issue First: Outline your proposal, including design considerations
    and impact.
-   Gather Feedback: Discuss with maintainers and the community to ensure
    alignment and avoid duplicate work

### Testing Requirements

To maintain code quality and prevent regressions, all code changes must include
comprehensive tests and verifiable end-to-end (E2E) evidence.

#### Unit Tests

Please add or update unit tests for your change. Please include a summary of
passed `pytest` results.

Requirements for unit tests:

-   **Coverage:** Cover new features, edge cases, error conditions, and typical
    use cases.
-   **Location:** Add or update tests under `tests/unittests/`, following
    existing naming conventions (e.g., `test_<module>_<feature>.py`).
-   **Framework:** Use `pytest`. Tests should be:
    -   Fast and isolated.
    -   Written clearly with descriptive names.
    -   Free of external dependencies (use mocks or fixtures as needed).
-   **Quality:** Aim for high readability and maintainability; include
    docstrings or comments for complex scenarios.

#### Manual End-to-End (E2E) Tests

Manual E2E tests ensure integrated flows work as intended. Your tests should
cover all scenarios. Sometimes, it's also good to ensure relevant functionality
is not impacted.

Depending on your change:

-   **ADK Web:**

    -   Use the `adk web` to verify functionality.
    -   Capture and attach relevant screenshots demonstrating the UI/UX changes
        or outputs.
    -   Label screenshots clearly in your PR description.

-   **Runner:**

    -   Provide the testing setup. For example, the agent definition, and the
        runner setup.
    -   Execute the `runner` tool to reproduce workflows.
    -   Include the command used and console output showing test results.
    -   Highlight sections of the log that directly relate to your change.

### Documentation

For any changes that impact user-facing documentation (guides, API reference,
tutorials), please open a PR in the
[adk-docs](https://github.com/google/adk-docs) repository to update relevant
part before or alongside your code PR.

## Development Setup

1.  **Clone the repository:**

    ```shell
    gh repo clone google/adk-python
    cd adk-python
    ```

2.  **Install uv:**

    Check out
    [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3.  **Create and activate a virtual environment:**

    **NOTE**: ADK supports Python 3.9+. Python 3.11 and above is strongly
    recommended.

    Create a workspace venv using uv.

    ```shell
    uv venv --python "python3.11" ".venv"
    ```

    Activate the workspace venv.

    ```shell
    source .venv/bin/activate
    ```

    **windows** `shell source .\.venv\Scripts\activate`

4.  **Install dependencies:**

    ```shell
    uv sync --all-extras
    ```

    **NOTE**: for convenience, installing all extra deps as a starting point.

5.  **Run unit tests:**

    ```shell
    pytest ./tests/unittests
    ```

    NOTE: for accurate repro of test failure, only include `test`, `eval` and
    `a2a` as extra dependencies.

    ```shell
    uv sync --extra test --extra eval --extra a2a
    pytest ./tests/unittests
    ```

6.  **Auto-format the code:**

    **NOTE**: We use `isort` and `pyink` for styles. Use the included
    autoformat.sh to auto-format.

    ```shell
    ./autoformat.sh
    ```

7.  **Build the wheel file:**

    ```shell
    uv build
    ```

8.  **Test the locally built wheel file:** Have a simple testing folder setup as
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
has resources that is helpful for contributors.

## Vibe Coding

If you want to contribute by leveraging vibe coding, the AGENTS.md
(https://github.com/google/adk-python/tree/main/AGENTS.md) could be used as
context to your LLM.
