# ADK Release Analyzer Agent

The ADK Release Analyzer Agent is a Python-based agent designed to help keep
documentation up-to-date with code changes. It analyzes the differences between
two releases of the `google/adk-python` repository, identifies required updates
in the `google/adk-docs` repository, and automatically generates a GitHub issue
with detailed instructions for documentation changes.

This agent can be operated in two distinct modes:

* an interactive mode for local use
* a fully automated mode for integration into workflows.

---

## Interactive Mode

This mode allows you to run the agent locally to review its recommendations in
real-time before any changes are made.

### Features

* **Web Interface**: The agent's interactive mode can be rendered in a web
browser using the ADK's `adk web` command.
* **User Approval**: In interactive mode, the agent is instructed to ask for
your confirmation before creating an issue on GitHub with the documentation
update instructions.
* **Question & Answer**: You ask questions about the releases and code changes.
The agent will provide answers based on related information.

### Running in Interactive Mode
To run the agent in interactive mode, first set the required environment
variables, ensuring `INTERACTIVE` is set to `1` or is unset. Then, execute the
following command in your terminal:

```bash
adk web contributing/samples/adk_documentation
```

This will start a local server and provide a URL to access the agent's web
interface in your browser.

---

## Automated Mode

For automated, hands-off analysis, the agent can be run as a script (`main.py`),
for example as part of a CI/CD pipeline. The workflow is configured in
`.github/workflows/analyze-releases-for-adk-docs-updates.yml` and automatically
checks the most recent two releases for docs updates.

### Workflow Triggers
The GitHub workflow is configured to run on specific triggers:

- **Release Events**: The workflow executes automatically whenever a new release
is `published`.

- **Manual Dispatch**: The workflow also runs when manually triggered for
testing and retrying.

### Automated Issue Creation

When running in automated mode, the agent operates non-interactively. It creates
a GitHub issue with the documentation update instructions directly without
requiring user approval. This behavior is configured by setting the
`INTERACTIVE` environment variable to `0`.

---

## Setup and Configuration

Whether running in interactive or automated mode, the agent requires the
following setup.

### Dependencies

The agent requires the following Python libraries.

```bash
pip install --upgrade pip
pip install google-adk
```

### Environment Variables

The following environment variables are required for the agent to connect to
the necessary services.

* `GITHUB_TOKEN`: **(Required)** A GitHub Personal Access Token with issues:write permissions for the documentation repository.
* `GOOGLE_API_KEY`: **(Required)** Your API key for the Gemini API.
* `DOC_OWNER`: The GitHub organization or username that owns the documentation repository (defaults to `google`).
* `CODE_OWNER`: The GitHub organization or username that owns the code repository (defaults to `google`).
* `DOC_REPO`: The name of the documentation repository (defaults to `adk-docs`).
* `CODE_REPO`: The name of the code repository (defaults to `adk-python`).
* `LOCAL_REPOS_DIR_PATH`: The local directory to clone the repositories into (defaults to `/tmp`).
* `INTERACTIVE`: Controls the agent's interaction mode. Set to 1 for interactive mode (default), and 0 for automated mode.

For local execution, you can place these variables in a `.env` file in the
project's root directory. For automated workflows, they should be configured as
environment variables or secrets.