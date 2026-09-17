# ADK Pull Request Triaging Assistant

The ADK Pull Request (PR) Triaging Assistant is a Python-based agent designed to help manage and triage GitHub pull requests for the `google/adk-python` repository. It uses a large language model to work out which component an incoming pull request belongs to, and assigns that component's owner as the assignee. It never labels a pull request; the component it picks is only used to look up the owner.

This agent can be operated in two distinct modes:

- an interactive mode for local use
- a fully automated GitHub Actions workflow.

______________________________________________________________________

## Interactive Mode

This mode allows you to run the agent locally to review its recommendations in real-time before any changes are made to your repository's pull requests.

### Features

- **Web Interface**: The agent's interactive mode can be rendered in a web browser using the ADK's `adk web` command.
- **User Approval**: In interactive mode, the agent is instructed to ask for your confirmation before assigning an owner to a GitHub pull request.

### Running in Interactive Mode

To run the agent in interactive mode, first set the required environment variables. Then, execute the following command in your terminal:

```bash
adk web
```

This will start a local server and provide a URL to access the agent's web interface in your browser.

______________________________________________________________________

## GitHub Workflow Mode

For automated, hands-off PR triaging, the agent can be integrated directly into your repository's CI/CD pipeline using a GitHub Actions workflow.

### Workflow Triggers

The GitHub workflow runs when a PR is `opened`, `reopened`, or marked `ready_for_review`, and on manual dispatch.

It deliberately does not run on pushes to an open PR, and there is no periodic backfill. Either one would re-assign an owner that a maintainer had just taken off, so removing an assignee stays a decision the agent cannot undo.

### Automated Assignment

When running as part of the GitHub workflow, the agent operates non-interactively. It assigns the component owner directly without requiring user approval, and skips any PR that already has an assignee. This behavior is configured by setting the `INTERACTIVE` environment variable to `0` in the workflow file.

### Workflow Configuration

The workflow is defined in a YAML file (`.github/workflows/pr-triage.yml`). This file contains the steps to check out the code, set up the Python environment, install dependencies, and run the triaging script with the necessary environment variables and secrets.

______________________________________________________________________

## Setup and Configuration

Whether running in interactive or workflow mode, the agent requires the following setup.

### Dependencies

The agent requires the following Python libraries.

```bash
pip install --upgrade pip
pip install google-adk
```

### Environment Variables

The following environment variables are required for the agent to connect to the necessary services.

- `GITHUB_TOKEN`: **(Required)** A GitHub Personal Access Token with `pull_requests:write` permissions. Needed for both interactive and workflow modes.
- `GOOGLE_API_KEY`: **(Required)** Your API key for the Gemini API. Needed for both interactive and workflow modes.
- `OWNER`: The GitHub organization or username that owns the repository (e.g., `google`). Needed for both modes.
- `REPO`: The name of the GitHub repository (e.g., `adk-python`). Needed for both modes.
- `INTERACTIVE`: Controls the agent's interaction mode. For the automated workflow, this is set to `0`. For interactive mode, it should be set to `1` or left unset.

For local execution in interactive mode, you can place these variables in a `.env` file in the project's root directory. For the GitHub workflow, they should be configured as repository secrets.
