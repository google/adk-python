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

import os

from google.adk import Agent
import requests

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
  raise ValueError("GITHUB_TOKEN environment variable not set")

OWNER = os.getenv("OWNER", "google")
REPO = os.getenv("REPO", "adk-python")
BOT_LABEL = os.getenv("BOT_LABEL", "bot_triaged")

BASE_URL = "https://api.github.com"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

ALLOWED_LABELS = [
    "documentation",
    "services",
    "question",
    "tools",
    "eval",
    "live",
    "models",
    "tracing",
    "core",
    "web",
]


def is_interactive():
  return os.environ.get("INTERACTIVE", "1").lower() in ["true", "1"]


def list_issues(issue_count: int):
  """
  Generator to list all issues for the repository by handling pagination.

  Args:
    issue_count: number of issues to return

  """
  query = f"repo:{OWNER}/{REPO} is:open is:issue no:label"

  unlabelled_issues = []
  url = f"{BASE_URL}/search/issues"

  params = {
      "q": query,
      "sort": "created",
      "order": "desc",
      "per_page": issue_count,
      "page": 1,
  }
  response = requests.get(url, headers=headers, params=params, timeout=60)
  response.raise_for_status()
  json_response = response.json()
  issues = json_response.get("items", None)
  if not issues:
    return []
  for issue in issues:
    if not issue.get("labels", None) or len(issue["labels"]) == 0:
      unlabelled_issues.append(issue)
  return unlabelled_issues


def add_label_to_issue(issue_number: str, label: str):
  """
  Add the specified label to the given issue number.

  Args:
    issue_number: issue number of the Github issue, in string foramt.
    label: label to assign
  """
  print(f"Attempting to add label '{label}' to issue #{issue_number}")
  if label not in ALLOWED_LABELS:
    error_message = (
        f"Error: Label '{label}' is not an allowed label. Will not apply."
    )
    print(error_message)
    return {"status": "error", "message": error_message, "applied_label": None}

  url = f"{BASE_URL}/repos/{OWNER}/{REPO}/issues/{issue_number}/labels"
  payload = [label, BOT_LABEL]
  response = requests.post(url, headers=headers, json=payload, timeout=60)
  response.raise_for_status()
  return response.json()


approval_instruction = (
    "Only label them when the user approves the labeling!"
    if is_interactive()
    else (
        "Do not ask for user approval for labeling! If you can't find a"
        " appropriate labels for the issue, do not label it."
    )
)

root_agent = Agent(
    model="gemini-2.5-pro-preview-05-06",
    name="adk_triaging_assistant",
    description="Triage ADK issues.",
    instruction=f"""
      You are a Github adk-python repo triaging bot. You will help get issues, and recommend a label.
      IMPORTANT: {approval_instruction}
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
      - If you can't find a appropriate labels for the issue, follow the previous instruction that starts with "IMPORTANT:".

      Present the followings in an easy to read format highlighting issue number and your label.
      - the issue summary in a few sentence
      - your label recommendation and justification
    """,
    tools=[
        list_issues,
        add_label_to_issue,
    ],
)
