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

import logging
import os
import dateutil.parser
from datetime import datetime, timezone
from typing import Any, List, Dict, Optional

from adk_stale_agent.settings import (
    GITHUB_BASE_URL,
    OWNER,
    REPO,
    LLM_MODEL_NAME,
    STALE_LABEL_NAME,
    REQUEST_CLARIFICATION_LABEL,
    STALE_HOURS_THRESHOLD,
    CLOSE_HOURS_AFTER_STALE_THRESHOLD,
)
from adk_stale_agent.utils import (
    post_request,
    delete_request,
    patch_request,
    error_response,
    get_request,
)
from google.adk.agents.llm_agent import Agent
from requests.exceptions import RequestException

logger = logging.getLogger("google_adk." + __name__)

# --- Constants ---
BOT_ALERT_SIGNATURE = "**Notification:** The author has updated the issue description"

# --- Global Cache ---
_MAINTAINERS_CACHE: Optional[List[str]] = None


def _get_cached_maintainers() -> List[str]:
    """
    Fetches the list of repository maintainers once and caches it.

    Returns:
        List[str]: A list of GitHub usernames with push access.
    """
    global _MAINTAINERS_CACHE
    if _MAINTAINERS_CACHE is not None:
        return _MAINTAINERS_CACHE

    logger.info("Initializing Maintainers Cache...")
    try:
        url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/collaborators"
        params = {"permission": "push"}
        data = get_request(url, params)
        _MAINTAINERS_CACHE = [u["login"] for u in data]
        logger.info(f"Cached {len(_MAINTAINERS_CACHE)} maintainers.")
    except Exception as e:
        logger.error(f"Failed to fetch maintainers: {e}")
        _MAINTAINERS_CACHE = []
    return _MAINTAINERS_CACHE


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


def get_issue_state(item_number: int) -> Dict[str, Any]:
    """
    Retrieves the comprehensive state of a GitHub issue using GraphQL.

    This function constructs a unified timeline of comments, body edits,
    renames, and reopens to determine who the *absolute last* actor was.
    It handles 'Ghost Edits' (description updates without comments) and
    prevents spamming alerts if the bot has already notified maintainers.

    Args:
        item_number (int): The GitHub issue number.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - last_action_role (str): 'author', 'maintainer', or 'other_user'.
            - is_stale (bool): Whether the issue is currently marked stale.
            - maintainer_alert_needed (bool): True if a silent edit needs an alert.
            - days_since_activity (float): Days since the last human action.
            - ... and other metadata for the LLM.
    """
    maintainers = _get_cached_maintainers()

    # GraphQL Query: Fetches Comments, Edits, and Timeline Events in one go.
    query = """
    query($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        issue(number: $number) {
          author { login }
          createdAt
          labels(first: 20) { nodes { name } }
          
          # 1. Comments (Fetch last 30 to scan for previous bot alerts)
          comments(last: 30) {
            nodes {
              author { login }
              body
              createdAt
              lastEditedAt
            }
          }
          
          # 2. Description Edits (Fetch last 10)
          userContentEdits(last: 10) {
            nodes {
              editor { login }
              editedAt
            }
          }
          
          # 3. Timeline Events (Renames, Reopens, Labels)
          timelineItems(itemTypes: [LABELED_EVENT, RENAMED_TITLE_EVENT, REOPENED_EVENT], last: 20) {
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

    variables = {"owner": OWNER, "name": REPO, "number": item_number}

    try:
        response = post_request(
            f"{GITHUB_BASE_URL}/graphql", {"query": query, "variables": variables}
        )

        if "errors" in response:
            msg = response["errors"][0]["message"]
            return error_response(f"GraphQL Error: {msg}")

        data = response.get("data", {}).get("repository", {}).get("issue", {})
        if not data:
            return error_response(f"Issue #{item_number} not found.")

        # --- Data Parsing ---
        issue_author = data.get("author", {}).get("login")
        labels_list = [l["name"] for l in data.get("labels", {}).get("nodes", [])]

        # We build a unified list of ALL events to replay history chronologically.
        history = []
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
            actor = c.get("author", {}).get("login")
            c_body = c.get("body", "")
            c_time = dateutil.parser.isoparse(c.get("createdAt"))

            # Check if the bot has already alerted about a silent edit in this thread
            if BOT_ALERT_SIGNATURE in c_body:
                if last_bot_alert_time is None or c_time > last_bot_alert_time:
                    last_bot_alert_time = c_time

            # Add human comments to history
            if actor and not actor.endswith("[bot]"):
                e_time = c.get("lastEditedAt")
                # Use edit time if available, otherwise creation time
                actual_time = dateutil.parser.isoparse(e_time) if e_time else c_time
                history.append({
                    "type": "commented",
                    "actor": actor,
                    "time": actual_time,
                    "data": c_body,
                })

        # 3. Process Body Edits
        for e in data.get("userContentEdits", {}).get("nodes", []):
            actor = e.get("editor", {}).get("login")
            if actor and not actor.endswith("[bot]"):
                history.append({
                    "type": "edited_description",
                    "actor": actor,
                    "time": dateutil.parser.isoparse(e.get("editedAt")),
                    "data": None,
                })

        # 4. Process Timeline Events (Labels, Renames, Reopens)
        label_events = []
        for t in data.get("timelineItems", {}).get("nodes", []):
            etype = t.get("__typename")
            actor = t.get("actor", {}).get("login")
            time_val = dateutil.parser.isoparse(t.get("createdAt"))

            # Store stale label events separately for timing calculations
            if etype == "LabeledEvent":
                if t.get("label", {}).get("name") == STALE_LABEL_NAME:
                    label_events.append(time_val)
                continue

            if actor and not actor.endswith("[bot]"):
                pretty_type = (
                    "renamed_title" if etype == "RenamedTitleEvent" else "reopened"
                )
                history.append({
                    "type": pretty_type,
                    "actor": actor,
                    "time": time_val,
                    "data": None,
                })

        # --- History Replay (Chronological Sort) ---
        history.sort(key=lambda x: x["time"])

        last_action_role = "author"  # Default start state
        last_activity_time = history[0]["time"]
        last_action_type = "created"
        last_comment_text = None

        logger.debug(f"--- Activity Trace for #{item_number} ---")

        for event in history:
            actor = event["actor"]
            etype = event["type"]

            # Determine Role
            role = "other_user"
            if actor == issue_author:
                role = "author"
            elif actor in maintainers:
                role = "maintainer"

            # Log the event trace for debugging
            logger.debug(
                f"  [{event['time'].strftime('%m-%d %H:%M')}] "
                f"{etype.upper()} by {actor} ({role})"
            )

            # Update State (The last event in the list wins)
            last_action_role = role
            last_activity_time = event["time"]
            last_action_type = etype

            if etype == "commented":
                last_comment_text = event["data"]
            else:
                last_comment_text = None

        # --- Spam Prevention / Alert Logic ---
        maintainer_alert_needed = False
        # If the User edited the description (silent action) and we haven't alerted AFTER that edit...
        if (
            last_action_role in ["author", "other_user"]
            and last_action_type == "edited_description"
        ):
            if last_bot_alert_time and last_bot_alert_time > last_activity_time:
                maintainer_alert_needed = False
                logger.info(
                    f"#{item_number}: Silent edit detected, but Bot already alerted at "
                    f"{last_bot_alert_time.strftime('%m-%d %H:%M')}. No spam."
                )
            else:
                maintainer_alert_needed = True
                logger.info(f"#{item_number}: Silent edit detected. Alert needed.")

        # --- Final Metric Calculations ---
        current_time = datetime.now(timezone.utc)
        days_since_activity = (
            current_time - last_activity_time
        ).total_seconds() / 86400

        is_stale = STALE_LABEL_NAME in labels_list
        days_since_stale_label = 0.0
        if is_stale and label_events:
            # Calculate time from the MOST RECENT application of the stale label
            latest_label_time = max(label_events)
            days_since_stale_label = (
                current_time - latest_label_time
            ).total_seconds() / 86400

        logger.debug(
            f"  -> FINAL VERDICT: Last Actor = {last_action_role.upper()}, "
            f"Idle = {days_since_activity:.2f} days"
        )

        return {
            "status": "success",
            "last_action_role": last_action_role,
            "last_action_type": last_action_type,
            "maintainer_alert_needed": maintainer_alert_needed,
            "is_stale": is_stale,
            "days_since_activity": days_since_activity,
            "days_since_stale_label": days_since_stale_label,
            "last_comment_text": last_comment_text,
            "current_labels": labels_list,
            "stale_threshold_days": STALE_HOURS_THRESHOLD / 24,
            "close_threshold_days": CLOSE_HOURS_AFTER_STALE_THRESHOLD / 24,
        }

    except RequestException as e:
        return error_response(f"Network Error: {e}")


# --- Tool Definitions ---


def add_label_to_issue(item_number: int, label_name: str) -> dict[str, Any]:
    """Adds a label to the issue."""
    logger.debug(f"Adding label '{label_name}' to issue #{item_number}.")
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels"
    try:
        post_request(url, [label_name])
        return {"status": "success"}
    except RequestException as e:
        return error_response(f"Error adding label: {e}")


def remove_label_from_issue(item_number: int, label_name: str) -> dict[str, Any]:
    """Removes a label from the issue."""
    logger.debug(f"Removing label '{label_name}' from issue #{item_number}.")
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels/{label_name}"
    try:
        delete_request(url)
        return {"status": "success"}
    except RequestException as e:
        return error_response(f"Error removing label: {e}")


def add_stale_label_and_comment(item_number: int) -> dict[str, Any]:
    """Marks the issue as stale with a comment and label."""
    comment = (
        "This issue has been automatically marked as stale because it has not"
        " had recent activity after a maintainer requested clarification. It"
        " will be closed if no further activity occurs within"
        f" {CLOSE_HOURS_AFTER_STALE_THRESHOLD / 24:.0f} days."
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
    """Posts a comment alerting maintainers of a silent description update."""
    comment = (
        "**Notification:** The author has updated the issue description. "
        "Maintainers, please review."
    )
    try:
        post_request(
            f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments",
            {"body": comment},
        )
        return {"status": "success"}
    except RequestException as e:
        return error_response(f"Error posting alert: {e}")


def close_as_stale(item_number: int) -> dict[str, Any]:
    """Closes the issue as not planned/stale."""
    comment = (
        "This has been automatically closed because it has been marked as stale"
        f" for over {CLOSE_HOURS_AFTER_STALE_THRESHOLD / 24:.0f} days."
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
        alert_maintainer_of_edit,
        get_issue_state,
        add_label_to_issue,
        remove_label_from_issue,
        add_stale_label_and_comment,
        close_as_stale,
    ],
)