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
import random
import time

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import requests

# Read the PAT from the environment variable
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Ensure you've set this in your shell
if not GITHUB_TOKEN:
  raise ValueError("GITHUB_TOKEN environment variable not set")

# Repository information
OWNER = "google"
REPO = "adk-python"

# Base URL for the GitHub API
BASE_URL = "https://api.github.com"

# Headers including the Authorization header
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def list_issues(per_page: int):
  """
  Generator to list all issues for the repository by handling pagination.

  Args:
    per_page: number of pages to return per page.

  """
  state = "open"
  # only process the 1st page for testing for now
  page = 1
  results = []
  url = (  # :contentReference[oaicite:16]{index=16}
      f"{BASE_URL}/repos/{OWNER}/{REPO}/issues"
  )
  # Warning: let's only handle max 10 issues at a time to avoid bad results
  params = {"state": state, "per_page": per_page, "page": page}
  response = requests.get(url, headers=headers, params=params)
  response.raise_for_status()  # :contentReference[oaicite:17]{index=17}
  issues = response.json()
  if not issues:
    return []
  for issue in issues:
    # Skip pull requests (issues API returns PRs as well)
    if "pull_request" in issue:
      continue
    results.append(issue)
  return results


def add_label_to_issue(issue_number: str, label: str):
  """
  Add the specified label to the given issue number.

  Args:
    issue_number: issue number of the Github issue, in string foramt.
    label: label to assign
  """
  url = f"{BASE_URL}/repos/{OWNER}/{REPO}/issues/{issue_number}/labels"
  payload = [label]
  response = requests.post(url, headers=headers, json=payload)
  response.raise_for_status()
  return response.json()


root_agent = Agent(
    model="gemini-2.5-pro-preview-05-06",
    name="adk_triaging_assistant",
    description="Triage ADK issues.",
    instruction="""
      You are a Github adk-python repo triaging bot. You will help get issues, and label them.
      Here are the rules for labeling:
      - If the user is asking about documentation-related questions, label it with "documentation".
      - If it's about session, memory services, label it with "services"
      - If it's about UI/web, label it with "question"
      - If it's related to tools, label it with "tools"
      - If it's about agent evalaution, then label it with "eval".
      - If it's about streaming/live, label it with "live".
      - If it's about model support(non-Gemini, like Litellm, Ollama, OpenAI models), label it with "models".
      - If it's about tracing, label it with "tracing".
      - If it's agent orchestration, agent definition, label it with "core".
      - If you can't find a appropriate labels for the issue, return the issues to user to decide.
    """,
    tools=[
        list_issues,
        add_label_to_issue,
    ],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(  # avoid false alarm about rolling dice.
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)
