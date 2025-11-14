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
    "mcp": "seanzhou1023",
    "tools": "seanzhou1023",
    "eval": "ankursharmas",
    "live": "hangfei",
    "models": "genquan9",
    "tracing": "jawoszek",
    "core": "Jacksunwei",
    "web": "wyf7107",
    "a2a": "seanzhou1023",
}

LABEL_GUIDELINES = """
      Label rubric and disambiguation rules:
      - "documentation": Tutorials, README content, reference docs, or samples.
      - "services": Session and memory services, persistence layers, or storage
        integrations.
      - "web": ADK web UI, FastAPI server, dashboards, or browser-based flows.
      - "question": Usage questions without a reproducible problem.
      - "tools": Built-in tools (e.g., SQL utils, code execution) or tool APIs.
      - "mcp": Model Context Protocol features. Apply both "mcp" and "tools".
      - "eval": Evaluation framework, test harnesses, scoring, or datasets.
      - "live": Streaming, bidi, audio, or Gemini Live configuration.
      - "models": Non-Gemini model adapters (LiteLLM, Ollama, OpenAI, etc.).
      - "tracing": Telemetry, observability, structured logs, or spans.
      - "core": Core ADK runtime (Agent definitions, Runner, planners,
        thinking config, CLI commands, GlobalInstructionPlugin, CPU usage, or
        general orchestration). Default to "core" when the topic is about ADK
        behavior and no other label is a better fit.
      - "agent engine": Vertex AI Agent Engine deployment or sandbox topics
        only (e.g., `.agent_engine_config.json`, `ae_ignore`, Agent Engine
        sandbox, `agent_engine_id`). If the issue does not explicitly mention
        Agent Engine concepts, do not use this label—choose "core" instead.
      - "a2a": Agent-to-agent workflows, coordination logic, or A2A protocol.

      When unsure between labels, prefer the most specific match. If a label
      cannot be assigned confidently, do not call the labeling tool.
"""

APPROVAL_INSTRUCTION = (
    "Do not ask for user approval for labeling! If you can't find appropriate"
    " labels for the issue, do not label it."
)
if IS_INTERACTIVE:
  APPROVAL_INSTRUCTION = "Only label them when the user approves the labeling!"


def list_unlabeled_issues(issue_count: int) -> dict[str, Any]:
  """List most recent `issue_count` number of unlabeled issues in the repo.

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
    issue_number: issue number of the GitHub issue.
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
  label_payload = [label]

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
    issue_number: issue number of the GitHub issue, in string format.
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
      You are a triaging bot for the GitHub {REPO} repo with the owner {OWNER}. You will help get issues, and recommend a label.
      IMPORTANT: {APPROVAL_INSTRUCTION}

      {LABEL_GUIDELINES}

      Here are the rules for labeling:
      - If the user is asking about documentation-related questions, label it with "documentation".
      - If it's about session, memory services, label it with "services".
      - If it's about UI/web, label it with "web".
      - If the user is asking about a question, label it with "question".
      - If it's related to tools, label it with "tools".
      - If it's about agent evaluation, then label it with "eval".
      - If it's about streaming/live, label it with "live".
      - If it's about model support (non-Gemini, like Litellm, Ollama, OpenAI models), label it with "models".
      - If it's about tracing, label it with "tracing".
      - If it's agent orchestration, agent definition, Runner behavior, planners, or performance, label it with "core".
      - Use "agent engine" only when the issue clearly references Vertex AI Agent Engine deployment artifacts (for example `.agent_engine_config.json`, `ae_ignore`, `agent_engine_id`, or Agent Engine sandbox errors).
      - If it's about Model Context Protocol (e.g. MCP tool, MCP toolset, MCP session management etc.), label it with both "mcp" and "tools".
      - If it's about A2A integrations or workflows, label it with "a2a".
      - If you can't find an appropriate labels for the issue, follow the previous instruction that starts with "IMPORTANT:".

      Call the `add_label_and_owner_to_issue` tool to label the issue, which will also assign the issue to the owner of the label.

      After you label the issue, call the `change_issue_type` tool to change the issue type:
      - If the issue is a bug report, change the issue type to "Bug".
      - If the issue is a feature request, change the issue type to "Feature".
      - Otherwise, **do not change the issue type**.

      Response quality requirements:
      - Summarize the issue in your own words without leaving template
        placeholders (never output text like "[fill in later]").
      - Justify the chosen label with a short explanation referencing the issue
        details.
      - Mention the assigned owner when a label maps to one.
      - If no label is applied, clearly state why.

      Present the following in an easy to read format highlighting issue number and your label.
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
