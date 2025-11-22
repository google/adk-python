# ADK Stale Issue Auditor Agent

This directory contains an autonomous agent designed to audit a GitHub repository for stale issues, helping to maintain repository hygiene and ensure that all open items are actionable.

The agent operates as a "Repository Auditor," proactively scanning all open issues rather than waiting for a specific trigger. It uses a combination of deterministic Python tools and the semantic understanding of a Large Language Model (LLM) to make intelligent decisions about the state of a conversation.

---

## Core Logic & Features

The agent's primary goal is to identify issues where a maintainer has requested information from the author, and to manage the lifecycle of that issue based on the author's response (or lack thereof).

**The agent follows a precise decision tree:**

1.  **Audits All Open Issues:** On each run, the agent fetches a batch of the oldest open issues in the repository.
2.  **Identifies Pending Issues:** It analyzes the full timeline of each issue to see if the last human action was a comment from a repository maintainer.
3.  **Semantic Intent Analysis:** If the last comment was from a maintainer, the agent uses the LLM to determine if the comment was a **question or a request for clarification**.
4.  **Marks as Stale:** If the maintainer's question has gone unanswered by the author for a configurable period (e.g., 7 days), the agent will:
    *   Apply a `stale` label to the issue.
    *   Post a comment notifying the author that the issue is now considered stale and will be closed if no further action is taken.
    *   Proactively add a `request clarification` label if it's missing, to make the issue's state clear.
5.  **Handles Activity:** If any non-maintainer (the author or a third party) comments on an issue, the agent will automatically remove the `stale` label, marking the issue as active again.
6.  **Closes Stale Issues:** If an issue remains in the `stale` state for another configurable period (e.g., 7 days) with no new activity, the agent will post a final comment and close the issue.

### Self-Configuration

A key feature of this agent is its ability to self-configure. It does not require a hard-coded list of maintainer usernames. On each run, it uses the GitHub API to dynamically fetch the list of users with write access to the repository, ensuring its logic is always based on the current team.

---

## Configuration

The agent is configured entirely via environment variables, which should be set as secrets in the GitHub Actions workflow environment.

### Required Secrets

| Secret Name | Description |
| :--- | :--- |
| `GITHUB_TOKEN` | A GitHub Personal Access Token (PAT) with the required permissions. It's recommended to use a PAT from a dedicated "bot" account.
| `GOOGLE_API_KEY` | An API key for the Google AI (Gemini) model used for the agent's reasoning.

### Required PAT Permissions

The `GITHUB_TOKEN` requires the following **Repository Permissions**:
*   **Issues**: `Read & write` (to read issues, add labels, comment, and close)
*   **Administration**: `Read-only` (to read the list of repository collaborators/maintainers)

### Optional Configuration

These environment variables can be set in the workflow file to override the defaults in `settings.py`.

| Variable Name | Description | Default |
| :--- | :--- | :--- |
| `STALE_HOURS_THRESHOLD` | The number of hours of inactivity after a maintainer's question before an issue is marked as `stale`. | `168` (7 days) |
| `CLOSE_HOURS_AFTER_STALE_THRESHOLD` | The number of hours after being marked `stale` before an issue is closed. | `168` (7 days) |
| `ISSUES_PER_RUN` | The maximum number of oldest open issues to process in a single workflow run. | `100` |
| `LLM_MODEL_NAME`| LLM model to use. | `gemini-2.5-flash` |

---

## Deployment

To deploy this agent, a GitHub Actions workflow file (`.github/workflows/stale-bot.yml`) is included. This workflow runs on a daily schedule and executes the agent's main script.

Ensure the necessary repository secrets are configured and the `stale` and `request clarification` labels exist in the repository.