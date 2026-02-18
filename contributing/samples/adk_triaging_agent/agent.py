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
    "a2a": "seanzhou1023",
    "agent engine": "yeesian",
    "auth": "seanzhou1023",
    "bq": "shobsi",
    "core": "Jacksunwei",
    "documentation": "joefernandez",
    "eval": "ankursharmas",
    "live": "seanzhou1023",
    "mcp": "seanzhou1023",
    "models": "xuanyang15",
    "services": "DeanChensj",
    "tools": "xuanyang15",
    "tracing": "jawoszek",
    "web": "wyf7107",
    "workflow": "DeanChensj",
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
        general orchestration including agent transfer for multi-agents system).
        Default to "core" when the topic is about ADK behavior and no other
        label is a better fit.
      - "agent engine": Vertex AI Agent Engine deployment or sandbox topics
        only (e.g., `.agent_engine_config.json`, `ae_ignore`, Agent Engine
        sandbox, `agent_engine_id`). If the issue does not explicitly mention
        Agent Engine concepts, do not use this label—choose "core" instead.
      - "a2a": A2A protocol, running agent as a2a agent with "--a2a" option for
        remote agent to talk with. Talking to remote agent via RemoteA2aAgent.
        NOT including those local multi-agent systems.
      - "bq": BigQuery integration or general issues related to BigQuery.
      - "workflow": Workflow agents and workflow execution.
      - "auth": Authentication or authorization issues.

      When unsure between labels, prefer the most specific match. If a label
      cannot be assigned confidently, do not call the labeling tool.
"""

APPROVAL_INSTRUCTION = (
    "Do not ask for user approval for labeling! If you can't find appropriate"
    " labels for the issue, do not label it."
)
if IS_INTERACTIVE:
  APPROVAL_INSTRUCTION = "Only label them when the user approves the labeling!"


def list_untriaged_issues(issue_count: int) -> dict[str, Any]:
  """List open issues that need triaging.

  Returns issues that need any of the following actions:
  1. Issues without component labels (need labeling + type setting)
  2. Issues with 'planned' label but no assignee (need owner assignment)

  Args:
    issue_count: number of issues to return

  Returns:
    The status of this request, with a list of issues when successful.
    Each issue includes flags indicating what actions are needed.
  """
  url = f"{GITHUB_BASE_URL}/search/issues"
  query = f"repo:{OWNER}/{REPO} is:open is:issue"
  params = {
      "q": query,
      "sort": "created",
      "order": "desc",
      "per_page": 100,  # Fetch more to filter
      "page": 1,
  }

  try:
    response = get_request(url, params)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")
  issues = response.get("items", [])

  component_labels = set(LABEL_TO_OWNER.keys())
  untriaged_issues = []
  for issue in issues:
    issue_labels = {label["name"] for label in issue.get("labels", [])}
    assignees = issue.get("assignees", [])

    existing_component_labels = issue_labels & component_labels
    has_component = bool(existing_component_labels)
    has_planned = "planned" in issue_labels

    # Determine what actions are needed
    needs_component_label = not has_component
    needs_owner = has_planned and not assignees

    # Include issue if it needs any action
    if needs_component_label or needs_owner:
      issue["has_planned_label"] = has_planned
      issue["has_component_label"] = has_component
      issue["existing_component_label"] = (
          list(existing_component_labels)[0]
          if existing_component_labels
          else None
      )
      issue["needs_component_label"] = needs_component_label
      issue["needs_owner"] = needs_owner
      untriaged_issues.append(issue)
      if len(untriaged_issues) >= issue_count:
        break
  return {"status": "success", "issues": untriaged_issues}


def add_label_to_issue(issue_number: int, label: str) -> dict[str, Any]:
  """Add the specified component label to the given issue number.

  Args:
    issue_number: issue number of the GitHub issue.
    label: label to assign

  Returns:
    The status of this request, with the applied label when successful.
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

  return {
      "status": "success",
      "message": response,
      "applied_label": label,
  }


def add_owner_to_issue(issue_number: int, label: str) -> dict[str, Any]:
  """Assign an owner to the issue based on the component label.

  This should only be called for issues that have the 'planned' label.

  Args:
    issue_number: issue number of the GitHub issue.
    label: component label that determines the owner to assign

  Returns:
    The status of this request, with the assigned owner when successful.
  """
  print(
      f"Attempting to assign owner for label '{label}' to issue #{issue_number}"
  )
  if label not in LABEL_TO_OWNER:
    return error_response(
        f"Error: Label '{label}' is not a valid component label."
    )

  owner = LABEL_TO_OWNER.get(label, None)
  if not owner:
    return {
        "status": "warning",
        "message": f"Label '{label}' does not have an owner. Will not assign.",
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
      "assigned_owner": owner,
  }


def change_issue_type(issue_number: int, issue_type: str) -> dict[str, Any]:
  """Change the issue type of the given issue number.

  Args:
    issue_number: issue number of the GitHub issue, in string format.
    issue_type: issue type to assign

  Returns:
    The status of this request, with the applied issue type when successful.
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

      ## Triaging Workflow

      Each issue will have flags indicating what actions are needed:
      - `needs_component_label`: true if the issue needs a component label
      - `needs_owner`: true if the issue needs an owner assigned (has 'planned' label but no assignee)

      For each issue, perform ONLY the required actions based on the flags:

      1. **If `needs_component_label` is true**:
         - Use `add_label_to_issue` to add the appropriate component label
         - Use `change_issue_type` to set the issue type:
           - Bug report → "Bug"
           - Feature request → "Feature"
           - Otherwise → do not change the issue type

      2. **If `needs_owner` is true**:
         - Use `add_owner_to_issue` to assign an owner based on the component label
         - Note: If the issue already has a component label (`has_component_label: true`), use that existing label to determine the owner

      Do NOT add a component label if `needs_component_label` is false.
      Do NOT assign an owner if `needs_owner` is false.

      Response quality requirements:
      - Summarize the issue in your own words without leaving template
        placeholders (never output text like "[fill in later]").
      - Justify the chosen label with a short explanation referencing the issue
        details.
      - Mention the assigned owner only when you actually assign one (i.e., when
        the issue has the 'planned' label).
      - If no label is applied, clearly state why.

      Present the following in an easy to read format highlighting issue number and your label.
      - the issue summary in a few sentence
      - your label recommendation and justification
      - the owner of the label if you assign the issue to an owner (only for planned issues)
    """,
    tools=[
        list_untriaged_issues,
        add_label_to_issue,
        add_owner_to_issue,
        change_issue_type,
    ],
)
