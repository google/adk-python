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

from datetime import datetime
from datetime import timezone
import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from adk_stale_agent.settings import CLOSE_HOURS_AFTER_STALE_THRESHOLD
from adk_stale_agent.settings import GITHUB_BASE_URL
from adk_stale_agent.settings import GRAPHQL_COMMENT_LIMIT
from adk_stale_agent.settings import GRAPHQL_EDIT_LIMIT
from adk_stale_agent.settings import GRAPHQL_TIMELINE_LIMIT
from adk_stale_agent.settings import LLM_MODEL_NAME
from adk_stale_agent.settings import OWNER
from adk_stale_agent.settings import REPO
from adk_stale_agent.settings import REQUEST_CLARIFICATION_LABEL
from adk_stale_agent.settings import STALE_HOURS_THRESHOLD
from adk_stale_agent.settings import STALE_LABEL_NAME
from adk_stale_agent.utils import delete_request
from adk_stale_agent.utils import error_response
from adk_stale_agent.utils import get_request
from adk_stale_agent.utils import patch_request
from adk_stale_agent.utils import post_request
import dateutil.parser
from google.adk.agents.llm_agent import Agent
from requests.exceptions import RequestException

logger = logging.getLogger("google_adk." + __name__)

# --- Constants ---
# Used to detect if the bot has already posted an alert to avoid spamming.
BOT_ALERT_SIGNATURE = (
    "**Notification:** The author has updated the issue description"
)
BOT_NAME = "adk-bot"

# --- Global Cache ---
_MAINTAINERS_CACHE: Optional[List[str]] = None


def _get_cached_maintainers() -> List[str]:
  """
  Fetches the list of repository maintainers.

  This function relies on `utils.get_request` for network resilience.
  `get_request` is configured with an HTTPAdapter that automatically performs
  exponential backoff retries (up to 6 times) for 5xx errors and rate limits.

  If the retries are exhausted or the data format is invalid, this function
  raises a RuntimeError to prevent the bot from running with incorrect permissions.

  Returns:
      List[str]: A list of GitHub usernames with push access.

  Raises:
      RuntimeError: If the API fails after all retries or returns invalid data.
  """
  global _MAINTAINERS_CACHE
  if _MAINTAINERS_CACHE is not None:
    return _MAINTAINERS_CACHE

  logger.info("Initializing Maintainers Cache...")

  try:
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/collaborators"
    params = {"permission": "push"}

    data = get_request(url, params)

    if isinstance(data, list):
      _MAINTAINERS_CACHE = [u["login"] for u in data if "login" in u]
      logger.info(f"Cached {len(_MAINTAINERS_CACHE)} maintainers.")
      return _MAINTAINERS_CACHE
    else:
      logger.error(
          f"Invalid API response format: Expected list, got {type(data)}"
      )
      raise ValueError(f"GitHub API returned non-list data: {data}")

  except Exception as e:
    logger.critical(
        f"FATAL: Failed to verify repository maintainers. Error: {e}"
    )
    raise RuntimeError(
        "Maintainer verification failed. processing aborted."
    ) from e


def load_prompt_template(filename: str) -> str:
  """
  Loads the raw text content of a prompt file.

  Args:
      filename (str): The name of the file (e.g., 'PROMPT_INSTRUCTION.txt').

  Returns:
      str: The file content.
  """
  file_path = os.path.join(os.path.dirname(__file__), filename)
  with open(file_path, "r") as f:
    return f.read()


PROMPT_TEMPLATE = load_prompt_template("PROMPT_INSTRUCTION.txt")


def _fetch_graphql_data(item_number: int) -> Dict[str, Any]:
  """
  Executes the GraphQL query to fetch raw issue data, including comments,
  edits, and timeline events.

  Args:
      item_number (int): The GitHub issue number.

  Returns:
      Dict[str, Any]: The raw 'issue' object from the GraphQL response.

  Raises:
      RequestException: If the GraphQL query returns errors or the issue is not found.
  """
  query = """
    query($owner: String!, $name: String!, $number: Int!, $commentLimit: Int!, $timelineLimit: Int!, $editLimit: Int!) {
      repository(owner: $owner, name: $name) {
        issue(number: $number) {
          author { login }
          createdAt
          labels(first: 20) { nodes { name } }

          comments(last: $commentLimit) {
            nodes {
              author { login }
              body
              createdAt
              lastEditedAt
            }
          }

          userContentEdits(last: $editLimit) {
            nodes {
              editor { login }
              editedAt
            }
          }

          timelineItems(itemTypes: [LABELED_EVENT, RENAMED_TITLE_EVENT, REOPENED_EVENT], last: $timelineLimit) {
            nodes {
              __typename
              ... on LabeledEvent {
                createdAt
                actor { login }
                label { name }
              }
              ... on RenamedTitleEvent {
                createdAt
                actor { login }
              }
              ... on ReopenedEvent {
                createdAt
                actor { login }
              }
            }
          }
        }
      }
    }
    """

  variables = {
      "owner": OWNER,
      "name": REPO,
      "number": item_number,
      "commentLimit": GRAPHQL_COMMENT_LIMIT,
      "editLimit": GRAPHQL_EDIT_LIMIT,
      "timelineLimit": GRAPHQL_TIMELINE_LIMIT,
  }

  response = post_request(
      f"{GITHUB_BASE_URL}/graphql", {"query": query, "variables": variables}
  )

  if "errors" in response:
    raise RequestException(f"GraphQL Error: {response['errors'][0]['message']}")

  data = response.get("data", {}).get("repository", {}).get("issue", {})
  if not data:
    raise RequestException(f"Issue #{item_number} not found.")

  return data


def _build_history_timeline(
    data: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[datetime], Optional[datetime]]:
  """
  Parses raw GraphQL data into a unified, chronologically sorted history list.
  Also extracts specific event times needed for logic checks.

  Args:
      data (Dict[str, Any]): The raw issue data from `_fetch_graphql_data`.

  Returns:
      Tuple[List[Dict], List[datetime], Optional[datetime]]:
          - history: A list of normalized event dictionaries sorted by time.
          - label_events: A list of timestamps when the stale label was applied.
          - last_bot_alert_time: Timestamp of the last bot silent-edit alert (if any).
  """
  issue_author = data.get("author", {}).get("login")
  history = []
  label_events = []
  last_bot_alert_time = None

  # 1. Baseline: Issue Creation
  history.append({
      "type": "created",
      "actor": issue_author,
      "time": dateutil.parser.isoparse(data["createdAt"]),
      "data": None,
  })

  # 2. Process Comments
  for c in data.get("comments", {}).get("nodes", []):
    if not c:
      continue

    actor = c.get("author", {}).get("login")
    c_body = c.get("body", "")
    c_time = dateutil.parser.isoparse(c.get("createdAt"))

    # Track bot alerts for spam prevention
    if BOT_ALERT_SIGNATURE in c_body:
      if last_bot_alert_time is None or c_time > last_bot_alert_time:
        last_bot_alert_time = c_time
      continue

    if actor and not actor.endswith("[bot]") and actor != BOT_NAME:
      # Use edit time if available, otherwise creation time
      e_time = c.get("lastEditedAt")
      actual_time = dateutil.parser.isoparse(e_time) if e_time else c_time
      history.append({
          "type": "commented",
          "actor": actor,
          "time": actual_time,
          "data": c_body,
      })

  # 3. Process Body Edits ("Ghost Edits")
  for e in data.get("userContentEdits", {}).get("nodes", []):
    if not e:
      continue
    actor = e.get("editor", {}).get("login")
    if actor and not actor.endswith("[bot]") and actor != BOT_NAME:
      history.append({
          "type": "edited_description",
          "actor": actor,
          "time": dateutil.parser.isoparse(e.get("editedAt")),
          "data": None,
      })

  # 4. Process Timeline Events
  for t in data.get("timelineItems", {}).get("nodes", []):
    if not t:
      continue

    etype = t.get("__typename")
    actor = t.get("actor", {}).get("login")
    time_val = dateutil.parser.isoparse(t.get("createdAt"))

    if etype == "LabeledEvent":
      if t.get("label", {}).get("name") == STALE_LABEL_NAME:
        label_events.append(time_val)
      continue

    if actor and not actor.endswith("[bot]") and actor != BOT_NAME:
      pretty_type = (
          "renamed_title" if etype == "RenamedTitleEvent" else "reopened"
      )
      history.append({
          "type": pretty_type,
          "actor": actor,
          "time": time_val,
          "data": None,
      })

  # Sort chronologically
  history.sort(key=lambda x: x["time"])
  return history, label_events, last_bot_alert_time


def _replay_history_to_find_state(
    history: List[Dict[str, Any]], maintainers: List[str], issue_author: str
) -> Dict[str, Any]:
  """
  Replays the unified event history to determine the absolute last actor and their role.

  Args:
      history (List[Dict]): Chronologically sorted list of events.
      maintainers (List[str]): List of maintainer usernames.
      issue_author (str): Username of the issue author.

  Returns:
      Dict[str, Any]: A dictionary containing the last state of the issue:
          - last_action_role (str): 'author', 'maintainer', or 'other_user'.
          - last_activity_time (datetime): Timestamp of the last human action.
          - last_action_type (str): The type of the last action (e.g., 'commented').
          - last_comment_text (Optional[str]): The text of the last comment.
          - last_actor_name (str): The specific username of the last actor.
  """
  last_action_role = "author"
  last_activity_time = history[0]["time"]
  last_action_type = "created"
  last_comment_text = None
  last_actor_name = issue_author

  for event in history:
    actor = event["actor"]
    etype = event["type"]

    role = "other_user"
    if actor == issue_author:
      role = "author"
    elif actor in maintainers:
      role = "maintainer"

    last_action_role = role
    last_activity_time = event["time"]
    last_action_type = etype
    last_actor_name = actor

    # Only store text if it was a comment (resets on other events like labels/edits)
    if etype == "commented":
      last_comment_text = event["data"]
    else:
      last_comment_text = None

  return {
      "last_action_role": last_action_role,
      "last_activity_time": last_activity_time,
      "last_action_type": last_action_type,
      "last_comment_text": last_comment_text,
      "last_actor_name": last_actor_name,
  }


def get_issue_state(item_number: int) -> Dict[str, Any]:
  """
  Retrieves the comprehensive state of a GitHub issue using GraphQL.

  This function orchestrates the fetching, parsing, and analysis of the issue's
  history to determine if it is stale, active, or pending maintainer review.

  Args:
      item_number (int): The GitHub issue number.

  Returns:
      Dict[str, Any]: A comprehensive state dictionary for the LLM agent.
          Contains keys such as 'last_action_role', 'is_stale', 'days_since_activity',
          and 'maintainer_alert_needed'.
  """
  try:
    maintainers = _get_cached_maintainers()

    # 1. Fetch
    raw_data = _fetch_graphql_data(item_number)

    issue_author = raw_data.get("author", {}).get("login")
    labels_list = [
        l["name"] for l in raw_data.get("labels", {}).get("nodes", [])
    ]

    # 2. Parse & Sort
    history, label_events, last_bot_alert_time = _build_history_timeline(
        raw_data
    )

    # 3. Analyze (Replay)
    state = _replay_history_to_find_state(history, maintainers, issue_author)

    # 4. Final Calculations & Alert Logic
    current_time = datetime.now(timezone.utc)
    days_since_activity = (
        current_time - state["last_activity_time"]
    ).total_seconds() / 86400

    # Stale Checks
    is_stale = STALE_LABEL_NAME in labels_list
    days_since_stale_label = 0.0
    if is_stale and label_events:
      latest_label_time = max(label_events)
      days_since_stale_label = (
          current_time - latest_label_time
      ).total_seconds() / 86400

    # Silent Edit Alert Logic
    maintainer_alert_needed = False
    if (
        state["last_action_role"] in ["author", "other_user"]
        and state["last_action_type"] == "edited_description"
    ):
      if (
          last_bot_alert_time
          and last_bot_alert_time > state["last_activity_time"]
      ):
        logger.info(
            f"#{item_number}: Silent edit detected, but Bot already alerted. No"
            " spam."
        )
      else:
        maintainer_alert_needed = True
        logger.info(f"#{item_number}: Silent edit detected. Alert needed.")

    logger.debug(
        f"#{item_number} VERDICT: Role={state['last_action_role']}, "
        f"Idle={days_since_activity:.2f}d"
    )

    return {
        "status": "success",
        "last_action_role": state["last_action_role"],
        "last_action_type": state["last_action_type"],
        "last_actor_name": state["last_actor_name"],
        "maintainer_alert_needed": maintainer_alert_needed,
        "is_stale": is_stale,
        "days_since_activity": days_since_activity,
        "days_since_stale_label": days_since_stale_label,
        "last_comment_text": state["last_comment_text"],
        "current_labels": labels_list,
        "stale_threshold_days": STALE_HOURS_THRESHOLD / 24,
        "close_threshold_days": CLOSE_HOURS_AFTER_STALE_THRESHOLD / 24,
        "maintainers": maintainers,
        "issue_author": issue_author,
    }

  except RequestException as e:
    return error_response(f"Network Error: {e}")
  except Exception as e:
    logger.error(
        f"Unexpected error analyzing #{item_number}: {e}", exc_info=True
    )
    return error_response(f"Analysis Error: {e}")


# --- Tool Definitions ---


def _format_days(hours: float) -> str:
  """
  Formats a duration in hours into a clean day string.

  Example:
      168.0 -> "7"
      12.0  -> "0.5"
  """
  days = hours / 24
  return f"{days:.1f}" if days % 1 != 0 else f"{int(days)}"


def add_label_to_issue(item_number: int, label_name: str) -> dict[str, Any]:
  """
  Adds a label to the issue.

  Args:
      item_number (int): The GitHub issue number.
      label_name (str): The name of the label to add.
  """
  logger.debug(f"Adding label '{label_name}' to issue #{item_number}.")
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels"
  try:
    post_request(url, [label_name])
    return {"status": "success"}
  except RequestException as e:
    return error_response(f"Error adding label: {e}")


def remove_label_from_issue(
    item_number: int, label_name: str
) -> dict[str, Any]:
  """
  Removes a label from the issue.

  Args:
      item_number (int): The GitHub issue number.
      label_name (str): The name of the label to remove.
  """
  logger.debug(f"Removing label '{label_name}' from issue #{item_number}.")
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels/{label_name}"
  try:
    delete_request(url)
    return {"status": "success"}
  except RequestException as e:
    return error_response(f"Error removing label: {e}")


def add_stale_label_and_comment(item_number: int) -> dict[str, Any]:
  """
  Marks the issue as stale with a comment and label.

  Args:
      item_number (int): The GitHub issue number.
  """
  stale_days_str = _format_days(STALE_HOURS_THRESHOLD)
  close_days_str = _format_days(CLOSE_HOURS_AFTER_STALE_THRESHOLD)

  comment = (
      "This issue has been automatically marked as stale because it has not"
      f" had recent activity for {stale_days_str} days after a maintainer"
      " requested clarification. It will be closed if no further activity"
      f" occurs within {close_days_str} days."
  )
  try:
    post_request(
        f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments",
        {"body": comment},
    )
    post_request(
        f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels",
        [STALE_LABEL_NAME],
    )
    return {"status": "success"}
  except RequestException as e:
    return error_response(f"Error marking issue as stale: {e}")


def alert_maintainer_of_edit(item_number: int) -> dict[str, Any]:
  """
  Posts a comment alerting maintainers of a silent description update.

  Args:
      item_number (int): The GitHub issue number.
  """
  # Uses the constant signature to ensure detection logic in get_issue_state works.
  comment = f"{BOT_ALERT_SIGNATURE}. Maintainers, please review."
  try:
    post_request(
        f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments",
        {"body": comment},
    )
    return {"status": "success"}
  except RequestException as e:
    return error_response(f"Error posting alert: {e}")


def close_as_stale(item_number: int) -> dict[str, Any]:
  """
  Closes the issue as not planned/stale.

  Args:
      item_number (int): The GitHub issue number.
  """
  days_str = _format_days(CLOSE_HOURS_AFTER_STALE_THRESHOLD)

  comment = (
      "This has been automatically closed because it has been marked as stale"
      f" for over {days_str} days."
  )
  try:
    post_request(
        f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments",
        {"body": comment},
    )
    patch_request(
        f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}",
        {"state": "closed"},
    )
    return {"status": "success"}
  except RequestException as e:
    return error_response(f"Error closing issue: {e}")


root_agent = Agent(
    model=LLM_MODEL_NAME,
    name="adk_repository_auditor_agent",
    description="Audits open issues.",
    instruction=PROMPT_TEMPLATE.format(
        OWNER=OWNER,
        REPO=REPO,
        STALE_LABEL_NAME=STALE_LABEL_NAME,
        REQUEST_CLARIFICATION_LABEL=REQUEST_CLARIFICATION_LABEL,
        stale_threshold_days=STALE_HOURS_THRESHOLD / 24,
        close_threshold_days=CLOSE_HOURS_AFTER_STALE_THRESHOLD / 24,
    ),
    tools=[
        add_label_to_issue,
        add_stale_label_and_comment,
        alert_maintainer_of_edit,
        close_as_stale,
        get_issue_state,
        remove_label_from_issue,
    ],
)
