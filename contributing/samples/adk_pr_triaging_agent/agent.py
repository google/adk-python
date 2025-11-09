# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any

from adk_pr_triaging_agent.settings import GITHUB_BASE_URL
from adk_pr_triaging_agent.settings import IS_INTERACTIVE
from adk_pr_triaging_agent.settings import OWNER
from adk_pr_triaging_agent.settings import REPO
from adk_pr_triaging_agent.utils import error_response
from adk_pr_triaging_agent.utils import get_diff
from adk_pr_triaging_agent.utils import post_request
from adk_pr_triaging_agent.utils import read_file
from adk_pr_triaging_agent.utils import run_graphql_query
from google.adk import Agent
import requests

ALLOWED_LABELS = [
    "documentation",
    "services",
    "tools",
    "mcp",
    "eval",
    "live",
    "models",
    "tracing",
    "core",
    "web",
]

CONTRIBUTING_MD = read_file(
    Path(__file__).resolve().parents[3] / "CONTRIBUTING.md"
)

APPROVAL_INSTRUCTION = (
    "Do not ask for user approval for labeling or commenting! If you can't find"
    " appropriate labels for the PR, do not label it."
)
if IS_INTERACTIVE:
  APPROVAL_INSTRUCTION = (
      "Only label or comment when the user approves the labeling or commenting!"
  )


def get_pull_request_details(pr_number: int) -> str:
  """Get the details of the specified pull request.

  Args:
    pr_number: number of the GitHub pull request.

  Returns:
    The status of this request, with the details when successful.
  """
  print(f"Fetching details for PR #{pr_number} from {OWNER}/{REPO}")
  query = """
    query($owner: String!, $repo: String!, $prNumber: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $prNumber) {
          id
          number
          title
          body
          state
          author {
            login
          }
          labels(last: 10) {
            nodes {
              name
            }
          }
          files(last: 50) {
            nodes {
              path
            }
          }
          comments(last: 50) {
            nodes {
              id
              body
              createdAt
              author {
                login
              }
            }
          }
          commits(last: 50) {
            nodes {
              commit {
                url
                message
              }
            }
          }
          statusCheckRollup {
            state
            contexts(last: 20) {
              nodes {
                ... on StatusContext {
                  context
                  state
                  targetUrl
                }
                ... on CheckRun {
                  name
                  status
                  conclusion
                  detailsUrl
                }
              }
            }
          }
        }
      }
    }
  """
  variables = {"owner": OWNER, "repo": REPO, "prNumber": pr_number}
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/pulls/{pr_number}"

  try:
    response = run_graphql_query(query, variables)
    if "errors" in response:
      return error_response(str(response["errors"]))

    pr = response.get("data", {}).get("repository", {}).get("pullRequest")
    if not pr:
      return error_response(f"Pull Request #{pr_number} not found.")

    # Filter out main merge commits.
    original_commits = pr.get("commits", {}).get("nodes", {})
    if original_commits:
      filtered_commits = [
          commit_node
          for commit_node in original_commits
          if not commit_node["commit"]["message"].startswith(
              "Merge branch 'main' into"
          )
      ]
      pr["commits"]["nodes"] = filtered_commits

    # Get diff of the PR and truncate it to avoid exceeding the maximum tokens.
    pr["diff"] = get_diff(url)[:10000]

    return {"status": "success", "pull_request": pr}
  except requests.exceptions.RequestException as e:
    return error_response(str(e))


def add_label_to_pr(pr_number: int, label: str) -> dict[str, Any]:
  """Adds a specified label on a pull request.

  Args:
      pr_number: the number of the GitHub pull request
      label: the label to add

  Returns:
      The the status of this request, with the applied label and response when
      successful.
  """
  print(f"Attempting to add label '{label}' to PR #{pr_number}")
  if label not in ALLOWED_LABELS:
    return error_response(
        f"Error: Label '{label}' is not an allowed label. Will not apply."
    )

  # Pull Request is a special issue in GitHub, so we can use issue url for PR.
  label_url = (
      f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{pr_number}/labels"
  )
  label_payload = [label]

  try:
    response = post_request(label_url, label_payload)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")

  return {
      "status": "success",
      "applied_label": label,
      "response": response,
  }


def add_comment_to_pr(pr_number: int, comment: str) -> dict[str, Any]:
  """Add the specified comment to the given PR number.

  Args:
    pr_number: the number of the GitHub pull request
    comment: the comment to add

  Returns:
    The the status of this request, with the applied comment when successful.
  """
  print(f"Attempting to add comment '{comment}' to issue #{pr_number}")

  # Pull Request is a special issue in GitHub, so we can use issue url for PR.
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{pr_number}/comments"
  payload = {"body": comment}

  try:
    post_request(url, payload)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")
  return {
      "status": "success",
      "added_comment": comment,
  }


root_agent = Agent(
    model="gemini-2.5-pro",
    name="adk_pr_triaging_assistant",
    description="Triage ADK pull requests.",
    instruction=f"""
      # 1. Identity
      You are a Pull Request (PR) triaging bot for the GitHub {REPO} repo with the owner {OWNER}.

      # 2. Responsibilities
      Your core responsibility includes:
      - Get the pull request details.
      - Add a label to the pull request.
      - Check if the pull request is following the contribution guidelines.
      - Add a comment to the pull request if it's not following the guidelines.

      **IMPORTANT: {APPROVAL_INSTRUCTION}**

      # 3. Guidelines & Rules
      Here are the rules for labeling:
      - If the PR is about documentations, label it with "documentation".
      - If it's about session, memory, artifacts services, label it with "services"
      - If it's about UI/web, label it with "web"
      - If it's related to tools, label it with "tools"
      - If it's about agent evaluation, then label it with "eval".
      - If it's about streaming/live, label it with "live".
      - If it's about model support(non-Gemini, like Litellm, Ollama, OpenAI models), label it with "models".
      - If it's about tracing, label it with "tracing".
      - If it's agent orchestration, agent definition, label it with "core".
      - If it's about Model Context Protocol (e.g. MCP tool, MCP toolset, MCP session management etc.), label it with "mcp".
      - If you can't find an appropriate labels for the PR, follow the previous instruction that starts with "IMPORTANT:".

      Here is the contribution guidelines:
      `{CONTRIBUTING_MD}`

      Here are the guidelines for checking if the PR is following the guidelines:
      - The "statusCheckRollup" in the pull request details may help you to identify if the PR is following some of the guidelines (e.g. CLA compliance).

      Here are the guidelines for the comment:
      - **Be Polite and Helpful:** Start with a friendly tone.
      - **Be Specific:** Clearly list only the sections from the contribution guidelines that are still missing.
      - **Address the Author:** Mention the PR author by their username (e.g., `@username`).
      - **Provide Context:** Explain *why* the information or action is needed.
      - **Do not be repetitive:** If you have already commented on an PR asking for information, do not comment again unless new information has been added and it's still incomplete.
      - **Identify yourself:** Include a bolded note (e.g. "Response from ADK Triaging Agent") in your comment to indicate this comment was added by an ADK Answering Agent.

      **Example Comment for a PR:**
      > **Response from ADK Triaging Agent**
      >
      > Hello @[pr-author-username], thank you for creating this PR!
      >
      > This PR is a bug fix, could you please associate the github issue with this PR? If there is no existing issue, could you please create one?
      >
      > In addition, could you please provide logs or screenshot after the fix is applied?
      >
      > This information will help reviewers to review your PR more efficiently. Thanks!

      # 4. Steps
      When you are given a PR, here are the steps you should take:
      - Call the `get_pull_request_details` tool to get the details of the PR.
      - Skip the PR (i.e. do not label or comment) if any of the following is true:
        - the PR is closed
        - the PR is labeled with "google-contributor"
        - the PR is already labelled with the above labels (e.g. "documentation", "services", "tools", etc.).
      - Check if the PR is following the contribution guidelines.
        - If it's not following the guidelines, recommend or add a comment to the PR that points to the contribution guidelines (https://github.com/google/adk-python/blob/main/CONTRIBUTING.md).
        - If it's following the guidelines, recommend or add a label to the PR.

      # 5. Output
      Present the following in an easy to read format highlighting PR number and your label.
      - The PR summary in a few sentence
      - The label you recommended or added with the justification
      - The comment you recommended or added to the PR with the justification
    """,
    tools=[
        get_pull_request_details,
        add_label_to_pr,
        add_comment_to_pr,
    ],
)
