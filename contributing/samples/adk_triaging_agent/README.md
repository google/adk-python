# ADK Issue Triaging Assistant

The ADK Issue Triaging Assistant is a Python-based agent designed to help manage and triage GitHub issues for the `google/adk-python` repository. It uses a large language model to analyze issues, recommend appropriate component labels, set issue types, and assign owners based on predefined rules.

This agent can be operated in two distinct modes: an interactive mode for local use or as a fully automated GitHub Actions workflow.

---

## Triaging Workflow

The agent performs different actions based on the issue state:

| Condition | Actions |
|-----------|---------|
| Issue without component label | Add component label + Set issue type (Bug/Feature) |
| Issue with "planned" label but no assignee | Assign owner based on component label |
| Issue with "planned" label AND no component label | Add component label + Set type + Assign owner |

### Component Labels
The agent can assign the following component labels, each mapped to an owner:
- `a2a`, `agent engine`, `auth`, `bq`, `core`, `documentation`, `eval`, `live`, `mcp`, `models`, `services`, `tools`, `tracing`, `web`, `workflow`

### Issue Types
Based on the issue content, the agent will set the issue type to:
- **Bug**: For bug reports
- **Feature**: For feature requests

---

## Interactive Mode

This mode allows you to run the agent locally to review its recommendations in real-time before any changes are made to your repository's issues.

### Features
* **Web Interface**: The agent's interactive mode can be rendered in a web browser using the ADK's `adk web` command.
* **User Approval**: In interactive mode, the agent is instructed to ask for your confirmation before applying labels or assigning owners.

### Running in Interactive Mode
To run the agent in interactive mode, first set the required environment variables. Then, execute the following command in your terminal:

```bash
adk web
```
This will start a local server and provide a URL to access the agent's web interface in your browser.

---

## GitHub Workflow Mode

For automated, hands-off issue triaging, the agent can be integrated directly into your repository's CI/CD pipeline using a GitHub Actions workflow.

### Workflow Triggers
The GitHub workflow is configured to run on specific triggers:

1.  **New Issues (`opened`)**: When a new issue is created, the agent adds an appropriate component label and sets the issue type.

2.  **Planned Label Added (`labeled` with "planned")**: When an issue is labeled as "planned", the agent assigns an owner based on the component label. If the issue doesn't have a component label yet, the agent will also add one.

3.  **Scheduled Runs**: The workflow runs every 6 hours to process any issues that need triaging (either missing component labels or missing assignees for "planned" issues).

### Automated Actions
When running as part of the GitHub workflow, the agent operates non-interactively:
- **Component Labeling**: Automatically applies the most appropriate component label
- **Issue Type Setting**: Sets the issue type to Bug or Feature based on content
- **Owner Assignment**: Only assigns owners for issues marked as "planned"

This behavior is configured by setting the `INTERACTIVE` environment variable to `0` in the workflow file.

### Workflow Configuration
The workflow is defined in a YAML file (`.github/workflows/triage.yml`). This file contains the steps to check out the code, set up the Python environment, install dependencies, and run the triaging script with the necessary environment variables and secrets.

---

## Setup and Configuration

Whether running in interactive or workflow mode, the agent requires the following setup.

### Dependencies
The agent requires the following Python libraries.

```bash
pip install --upgrade pip
pip install google-adk requests
```

### Environment Variables
The following environment variables are required for the agent to connect to the necessary services.

* `GITHUB_TOKEN`: **(Required)** A GitHub Personal Access Token with `issues:write` permissions. Needed for both interactive and workflow modes.
* `GOOGLE_API_KEY`: **(Required)** Your API key for the Gemini API. Needed for both interactive and workflow modes.
* `OWNER`: The GitHub organization or username that owns the repository (e.g., `google`). In the workflow, this is automatically set from the repository context.
* `REPO`: The name of the GitHub repository (e.g., `adk-python`). In the workflow, this is automatically set from the repository context.
* `INTERACTIVE`: Controls the agent's interaction mode. For the automated workflow, this is set to `0`. For interactive mode, it should be set to `1` or left unset.

For local execution in interactive mode, you can place these variables in a `.env` file in the project's root directory. For the GitHub workflow, they should be configured as repository secrets.