# ADK Pull Request Triaging Assistant

The ADK Pull Request (PR) Triaging Assistant is a Python-based agent designed to help manage and triage GitHub pull requests for the `google/adk-python` repository. It uses a large language model to work out which component an incoming pull request belongs to, and assigns one of that component's owners as the assignee; when a component has several owners, they take turns by PR number. It never labels a pull request; the component it picks is only used to look up the owner.

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

## Setup and Configuration

### Dependencies

The agent requires the following Python libraries.

```bash
pip install --upgrade pip
pip install google-adk
```

### Environment Variables

The following environment variables are required for the agent to connect to the necessary services.

- `GITHUB_TOKEN`: **(Required)** A GitHub Personal Access Token with `pull_requests:write` permissions.
- `GOOGLE_API_KEY`: **(Required)** Your API key for the Gemini API.
- `OWNER`: The GitHub organization or username that owns the repository (e.g., `google`).
- `REPO`: The name of the GitHub repository (e.g., `adk-python`).
- `INTERACTIVE`: Controls the agent's interaction mode. Set to `0` to assign without asking, or `1` (or unset) for interactive approval.

For local execution, place these variables in `.env` in the project root.
