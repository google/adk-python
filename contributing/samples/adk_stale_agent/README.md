# ADK Stale Issue Auditor Agent

This directory contains an autonomous, **GraphQL-powered** agent designed to audit a GitHub repository for stale issues. It maintains repository hygiene by ensuring all open items are actionable and responsive.

Unlike traditional "Stale Bots" that only look at timestamps, this agent uses a **Unified History Trace** and an **LLM (Large Language Model)** to understand the *context* of a conversation. It distinguishes between a maintainer asking a question (stale candidate) vs. a maintainer providing a status update (active).

---

## Core Logic & Features

The agent operates as a "Repository Auditor," proactively scanning open issues using a high-efficiency decision tree.

### 1. Smart State Verification (GraphQL)
Instead of making multiple expensive API calls, the agent uses a single **GraphQL** query per issue to reconstruct the entire history of the conversation. It combines:
*   **Comments**
*   **Description/Body Edits** ("Ghost Edits")
*   **Title Renames**
*   **State Changes** (Reopens)

It sorts these events chronologically to determine the **Last Active Actor**.

### 2. The "Last Actor" Rule
The agent follows a precise logic flow based on who acted last:

*   **If Author/User acted last:** The issue is **ACTIVE**.
    *   This includes comments, title changes, and *silent* description edits.
    *   **Action:** The agent immediately removes the `stale` label.
    *   **Silent Update Alert:** If the user edited the description but *did not* comment, the agent posts a specific alert: *"Notification: The author has updated the issue description..."* to ensure maintainers are notified (since GitHub does not trigger notifications for body edits).
    *   **Spam Prevention:** The agent checks if it has already alerted about a specific silent edit to avoid spamming the thread.

*   **If Maintainer acted last:** The issue is **POTENTIALLY STALE**.
    *   The agent passes the text of the maintainer's last comment to the LLM.

### 3. Semantic Intent Analysis (LLM)
If the maintainer was the last person to speak, the LLM analyzes the comment text to determine intent:
*   **Question/Request:** "Can you provide logs?" / "Please try v2.0."
    *   **Verdict:** **STALE** (Waiting on Author).
    *   **Action:** If the time threshold is met, the agent adds the `stale` label. It also checks for the `request clarification` label and adds it if missing.
*   **Status Update:** "We are working on a fix." / "Added to backlog."
    *   **Verdict:** **ACTIVE** (Waiting on Maintainer).
    *   **Action:** No action taken. The issue remains open without stale labels.

### 4. Lifecycle Management
*   **Marking Stale:** After `STALE_HOURS_THRESHOLD` (default: 7 days) of inactivity following a maintainer's question.
*   **Closing:** After `CLOSE_HOURS_AFTER_STALE_THRESHOLD` (default: 7 days) of continued inactivity while marked stale.

---

## Performance & Safety

*   **GraphQL Optimized:** Fetches comments, edits, labels, and timeline events in a single network request to minimize latency and API quota usage.
*   **Search API Filtering:** Uses the GitHub Search API to pre-filter issues created recently, ensuring the bot doesn't waste cycles analyzing brand-new issues.
*   **Rate Limit Aware:** Includes intelligent sleeping and retry logic (exponential backoff) to handle GitHub API rate limits (HTTP 429) gracefully.
*   **Execution Metrics:** Logs the time taken and API calls consumed for every issue processed.

---

## Configuration

The agent is configured via environment variables, typically set as secrets in GitHub Actions.

### Required Secrets

| Secret Name | Description |
| :--- | :--- |
| `GITHUB_TOKEN` | A GitHub Personal Access Token (PAT) or Service Account Token with `repo` scope. |
| `GOOGLE_API_KEY` | An API key for the Google AI (Gemini) model used for reasoning. |

### Optional Configuration

These variables control the timing thresholds and model selection.

| Variable Name | Description | Default |
| :--- | :--- | :--- |
| `STALE_HOURS_THRESHOLD` | Hours of inactivity after a maintainer's question before marking as `stale`. | `168` (7 days) |
| `CLOSE_HOURS_AFTER_STALE_THRESHOLD` | Hours after being marked `stale` before the issue is closed. | `168` (7 days) |
| `LLM_MODEL_NAME`| The specific Gemini model version to use. | `gemini-2.5-flash` |
| `OWNER` | Repository owner (auto-detected in Actions). | (Environment dependent) |
| `REPO` | Repository name (auto-detected in Actions). | (Environment dependent) |

---

## Deployment

To deploy this agent, a GitHub Actions workflow file (`.github/workflows/stale-bot.yml`) is recommended.

### Directory Structure Note
Because this agent resides within the `adk-python` package structure, the workflow must ensure the script is executed correctly to handle imports.

