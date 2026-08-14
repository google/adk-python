# Copyright 2026 Google LLC
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

from typing import Any

from adk_pr_triaging_agent.settings import GITHUB_BASE_URL
from adk_pr_triaging_agent.settings import IS_INTERACTIVE
from adk_pr_triaging_agent.settings import OWNER
from adk_pr_triaging_agent.settings import REPO
from adk_pr_triaging_agent.utils import error_response
from adk_pr_triaging_agent.utils import get_diff
from adk_pr_triaging_agent.utils import get_request
from adk_pr_triaging_agent.utils import is_assignable
from adk_pr_triaging_agent.utils import post_request
from adk_pr_triaging_agent.utils import run_graphql_query
from component_owners import LABEL_TO_OWNER
from google.adk import Agent
import requests

# LABEL_TO_OWNER (component -> owner GitHub login; the owner becomes the PR's
# assignee) is imported from component_owners and shared verbatim with
# adk_triaging_agent, so the two can't drift. Keep it in sync with OWNERS.
# This agent only reads the map to pick an assignee; the component names are
# never written to the PR as labels.

APPROVAL_INSTRUCTION = (
    "Do not ask for user approval for assigning!"
    " If you can't tell which component the PR belongs to, leave it"
    " unassigned."
)
if IS_INTERACTIVE:
  APPROVAL_INSTRUCTION = "Only assign when the user approves the action!"


def get_pull_request_details(pr_number: int) -> dict[str, Any]:
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
          assignees(first: 10) {
            nodes {
              login
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


def assign_owner_to_pr(pr_number: int, component: str) -> dict[str, Any]:
  """Assign the component owner (the shepherd) to a PR.

  The owner is looked up from `LABEL_TO_OWNER` so the contributor can see who is
  shepherding their PR. GitHub only allows assigning users with
  repo write/triage access, so a non-assignable owner is reported as skipped
  rather than silently dropped.

  Args:
    pr_number: the number of the GitHub pull request
    component: the component the PR belongs to

  Returns:
    The status of this request, with the assigned owner when successful.
  """
  owner = LABEL_TO_OWNER.get(component)
  if not owner:
    return error_response(
        f"Error: no owner mapped for component '{component}'."
    )
  print(f"Attempting to assign owner '{owner}' to PR #{pr_number}")
  if not is_assignable(owner):
    return {
        "status": "skipped",
        "reason": f"'{owner}' is not assignable (needs repo access)",
        "owner": owner,
    }

  # Pull Request is a special issue in GitHub, so we can use the issue url.
  assignee_url = (
      f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{pr_number}/assignees"
  )
  try:
    response = post_request(assignee_url, {"assignees": [owner]})
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")

  return {
      "status": "success",
      "assigned_owner": owner,
      "response": response,
  }


def list_unassigned_pull_requests(pr_count: int) -> dict[str, Any]:
  """List open pull requests that have nobody assigned.

  Skips pull requests labeled google-contributor, which are shepherded by their
  own author.

  Args:
    pr_count: number of pull requests to return

  Returns:
    The status of this request, with a list of pull requests when successful.
  """
  url = f"{GITHUB_BASE_URL}/search/issues"
  query = f"repo:{OWNER}/{REPO} is:open is:pr no:assignee"
  params = {
      "q": query,
      "sort": "updated",
      "order": "desc",
      "per_page": 100,
      "page": 1,
  }

  try:
    response = get_request(url, params)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")

  issues = response.get("items", [])
  unassigned_prs = []

  for pr in issues:
    pr_labels = {label["name"] for label in pr.get("labels", [])}
    if "google-contributor" in pr_labels:
      continue

    unassigned_prs.append({
        "number": pr["number"],
        "title": pr["title"],
    })

    if len(unassigned_prs) >= pr_count:
      break

  return {"status": "success", "pull_requests": unassigned_prs}


root_agent = Agent(
    model="gemini-3.5-flash",
    name="adk_pr_triaging_assistant",
    description="Assign component owners to ADK pull requests.",
    instruction=f"""
      # 1. Identity
      You are a Pull Request (PR) triaging bot for the GitHub {REPO} repo with the owner {OWNER}.

      # 2. Responsibilities
      Your core responsibility includes:
      - Get the pull request details.
      - Work out which component the pull request belongs to.
      - Assign that component's owner (the shepherd) to the pull request.

      Never label a pull request. The component you pick is only used to look up
      the owner, and is never written to the pull request.

      **IMPORTANT: {APPROVAL_INSTRUCTION}**

      # 3. Guidelines & Rules
      Here are the rules for picking the component:
      - If the PR is about documentations, the component is "documentation".
      - If it's about session, memory, artifacts services, the component is "services".
      - If it's about UI/web, the component is "web".
      - If it's related to tools, the component is "tools".
      - If it's about agent evaluation, the component is "eval".
      - If it's about streaming/live, the component is "live".
      - If it's about model support(non-Gemini, like Litellm, Ollama, OpenAI models), the component is "models".
      - If it's about tracing, the component is "tracing".
      - If it's about authentication or authorization, the component is "auth".
      - If it's about BigQuery integration, the component is "bq".
      - If it's about ADK CLI commands (e.g. create, deploy, eval) or CLI tools, the component is "cli".
      - If it's about third-party integrations (e.g. CrewAI, LangChain, Slack) excluding BigQuery, the component is "integrations".
      - If it's about GCP Skills Registry (GCPSkillRegistry), skill prompt models, or dynamic skill toolsets, the component is "skills".
      - If it's about workflow agents or workflow execution, the component is "workflow".
      - If it's agent orchestration, agent definition, the component is "core".
      - If it's about Model Context Protocol (e.g. MCP tool, MCP toolset, MCP session management etc.), the component is "mcp".
      - If you can't find an appropriate component for the PR, follow the previous instruction that starts with "IMPORTANT:".

      # 4. Steps
      - If you are asked to find pull requests that need an owner, use `list_unassigned_pull_requests` first.
      - For each pull request:
        - Call the `get_pull_request_details` tool to get the details of the PR.
        - Skip the PR (i.e. do not assign anyone) if any of the following is true:
          - the PR is closed
          - the PR is labeled with "google-contributor"
          - the PR already has an assignee
        - Work out the component, then call `assign_owner_to_pr` with it.
        - If the tool reports the owner is not assignable, just note it.

      # 5. Output
      Present the following in an easy to read format highlighting PR number and the owner you assigned.
      - The PR summary in a few sentence
      - The owner you assigned with the justification, or why you assigned nobody
    """,
    tools=[
        list_unassigned_pull_requests,
        get_pull_request_details,
        assign_owner_to_pr,
    ],
)
