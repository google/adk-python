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

from typing import Any

from adk_triaging_agent.settings import BOT_LABEL
from adk_triaging_agent.settings import GITHUB_BASE_URL
from adk_triaging_agent.settings import IS_INTERACTIVE
from adk_triaging_agent.settings import OWNER
from adk_triaging_agent.settings import REPO
from adk_triaging_agent.utils import error_response
from adk_triaging_agent.utils import get_request
from adk_triaging_agent.utils import patch_request
from adk_triaging_agent.utils import post_request
from google.adk.agents.llm_agent import Agent
import requests

LABEL_TO_OWNER = {
    "agent engine": "yeesian",
    "documentation": "polong-lin",
    "services": "DeanChensj",
    "question": "",
    "tools": "seanzhou1023",
    "eval": "ankursharmas",
    "live": "hangfei",
    "models": "genquan9",
    "tracing": "Jacksunwei",
    "core": "Jacksunwei",
    "web": "wyf7107",
}

APPROVAL_INSTRUCTION = (
    "Do not ask for user approval for labeling! If you can't find appropriate"
    " labels for the issue, do not label it."
)
if IS_INTERACTIVE:
  APPROVAL_INSTRUCTION = "Only label them when the user approves the labeling!"


def list_unlabeled_issues(issue_count: int) -> dict[str, Any]:
  """List most recent `issue_count` numer of unlabeled issues in the repo.

  Args:
    issue_count: number of issues to return

  Returns:
    The status of this request, with a list of issues when successful.
  """
  url = f"{GITHUB_BASE_URL}/search/issues"
  query = f"repo:{OWNER}/{REPO} is:open is:issue no:label"
  params = {
      "q": query,
      "sort": "created",
      "order": "desc",
      "per_page": issue_count,
      "page": 1,
  }

  try:
    response = get_request(url, params)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")
  issues = response.get("items", None)

  unlabeled_issues = []
  for issue in issues:
    if not issue.get("labels", None):
      unlabeled_issues.append(issue)
  return {"status": "success", "issues": unlabeled_issues}


def add_label_and_owner_to_issue(
    issue_number: int, label: str
) -> dict[str, Any]:
  """Add the specified label and owner to the given issue number.

  Args:
    issue_number: issue number of the Github issue.
    label: label to assign

  Returns:
    The the status of this request, with the applied label and assigned owner
    when successful.
  """
  print(f"Attempting to add label '{label}' to issue #{issue_number}")
  if label not in LABEL_TO_OWNER:
    return error_response(
        f"Error: Label '{label}' is not an allowed label. Will not apply."
    )

  label_url = (
      f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{issue_number}/labels"
  )
  label_payload = [label, BOT_LABEL]

  try:
    response = post_request(label_url, label_payload)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")

  owner = LABEL_TO_OWNER.get(label, None)
  if not owner:
    return {
        "status": "warning",
        "message": (
            f"{response}\n\nLabel '{label}' does not have an owner. Will not"
            " assign."
        ),
        "applied_label": label,
    }

  assignee_url = (
      f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{issue_number}/assignees"
  )
  assignee_payload = {"assignees": [owner]}

  try:
    response = post_request(assignee_url, assignee_payload)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")

  return {
      "status": "success",
      "message": response,
      "applied_label": label,
      "assigned_owner": owner,
  }


def change_issue_type(issue_number: int, issue_type: str) -> dict[str, Any]:
  """Change the issue type of the given issue number.

  Args:
    issue_number: issue number of the Github issue, in string foramt.
    issue_type: issue type to assign

  Returns:
    The the status of this request, with the applied issue type when successful.
  """
  print(
      f"Attempting to change issue type '{issue_type}' to issue #{issue_number}"
  )
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{issue_number}"
  payload = {"type": issue_type}

  try:
    response = patch_request(url, payload)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")

  return {"status": "success", "message": response, "issue_type": issue_type}


root_agent = Agent(
    model="gemini-2.5-pro",
    name="adk_triaging_assistant",
    description="Triage ADK issues.",
    instruction=f"""
      You are a triaging bot for the Github {REPO} repo with the owner {OWNER}. You will help get issues, and recommend a label.
      IMPORTANT: {APPROVAL_INSTRUCTION}

      Here are the rules for labeling:
      - If the user is asking about documentation-related questions, label it with "documentation".
      - If it's about session, memory services, label it with "services"
      - If it's about UI/web, label it with "web"
      - If the user is asking about a question, label it with "question"
      - If it's related to tools, label it with "tools"
      - If it's about agent evalaution, then label it with "eval".
      - If it's about streaming/live, label it with "live".
      - If it's about model support(non-Gemini, like Litellm, Ollama, OpenAI models), label it with "models".
      - If it's about tracing, label it with "tracing".
      - If it's agent orchestration, agent definition, label it with "core".
      - If it's about agent engine, label it with "agent engine".
      - If you can't find a appropriate labels for the issue, follow the previous instruction that starts with "IMPORTANT:".

      Call the `add_label_and_owner_to_issue` tool to label the issue, which will also assign the issue to the owner of the label.

      After you label the issue, call the `change_issue_type` tool to change the issue type:
      - If the issue is a bug report, change the issue type to "Bug".
      - If the issue is a feature request, change the issue type to "Feature".
      - Otherwise, **do not change the issue type**.

      Present the followings in an easy to read format highlighting issue number and your label.
      - the issue summary in a few sentence
      - your label recommendation and justification
      - the owner of the label if you assign the issue to an owner
    """,
    tools=[
        list_unlabeled_issues,
        add_label_and_owner_to_issue,
        change_issue_type,
    ],
)
