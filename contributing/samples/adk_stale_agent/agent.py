from datetime import datetime
from datetime import timezone
from typing import Any

from adk_stale_agent.settings import CLOSE_HOURS_AFTER_STALE_THRESHOLD
from adk_stale_agent.settings import GITHUB_BASE_URL
from adk_stale_agent.settings import ISSUES_PER_RUN
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

# --- Primary Tools for the Agent ---


def get_repository_maintainers() -> dict[str, Any]:
  """
  Fetches the list of repository collaborators with 'push' (write) access or higher.
  This should only be called once per run.
  """
  print("DEBUG: Fetching repository maintainers with push access...")
  try:
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/collaborators"
    params = {"permission": "push"}
    collaborators_data = get_request(url, params)

    maintainers = [user["login"] for user in collaborators_data]
    print(f"DEBUG: Found {len(maintainers)} maintainers: {maintainers}")

    return {"status": "success", "maintainers": maintainers}
  except Exception as e:
    return error_response(f"Error fetching repository maintainers: {e}")


def get_all_open_issues() -> dict[str, Any]:
  """Fetches a batch of the oldest open issues for an audit."""
  print(
      f"\nDEBUG: Fetching a batch of {ISSUES_PER_RUN} oldest open issues for"
      " audit..."
  )
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues"
  params = {
      "state": "open",
      "sort": "created",
      "direction": "asc",
      "per_page": ISSUES_PER_RUN,
  }
  try:
    items = get_request(url, params)
    print(f"DEBUG: Found {len(items)} open issues to audit.")
    return {"status": "success", "items": items}
  except Exception as e:
    return error_response(f"Error fetching all open issues: {e}")


def get_issue_state(item_number: int, maintainers: list[str]) -> dict[str, Any]:
  """Analyzes an issue's complete history to create a comprehensive state summary.

  This function acts as the primary "detective" for the agent. It performs the
  complex, deterministic work of fetching and parsing an issue's full history,
  allowing the LLM agent to focus on high-level semantic decision-making.

  It is designed to be highly robust by fetching the complete, multi-page history
  from the GitHub `/timeline` API. By handling pagination correctly, it ensures
  that even issues with a very long history (more than 100 events) are analyzed
  in their entirety, preventing incorrect decisions based on incomplete data.

  Args:
      item_number (int): The number of the GitHub issue or pull request to analyze.
      maintainers (list[str]): A dynamically fetched list of GitHub usernames to be
          considered maintainers. This is used to categorize actors found in
          the issue's history.

  Returns:
      A dictionary that serves as a clean, factual report summarizing the
      issue's state. On failure, it returns a dictionary with an 'error' status.
  """
  try:
    # Step 1: Fetch core issue data and prepare for timeline fetching.
    print(f"DEBUG: Fetching full timeline for issue #{item_number}...")
    issue_url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}"
    issue_data = get_request(issue_url)

    # --- PAGINATION LOGIC as suggested by Gemini Code Assist ---
    # Step 2: Fetch ALL pages from the timeline API to build a complete history.
    timeline_url_base = f"{issue_url}/timeline"
    timeline_data = []
    page = 1

    while True:
      # Construct the URL for the current page, requesting 100 items per page.
      paginated_url = f"{timeline_url_base}?per_page=100&page={page}"
      print(f"DEBUG: Fetching timeline page {page}...")

      events_page = get_request(paginated_url)

      # If the API returns an empty list, we have reached the end.
      if not events_page:
        break

      # Add the events from the current page to our master list.
      timeline_data.extend(events_page)

      # Optimization: if the number of events is less than the max page size,
      # we know it must be the final page, so we can stop early.
      if len(events_page) < 100:
        break

      # Prepare to fetch the next page in the next iteration.
      page += 1

    print(
        f"DEBUG: Fetched a total of {len(timeline_data)} timeline events across"
        f" {page} page(s)."
    )
    # --- END PAGINATION LOGIC ---

    # The rest of the function now proceeds with the complete, unabridged timeline data.

    # Step 3: Initialize key variables for the analysis.
    issue_author = issue_data.get("user", {}).get("login")
    current_labels = [label["name"] for label in issue_data.get("labels", [])]

    # Step 4: Filter and sort all events into a clean, chronological history of human activity.
    human_events = []
    for event in timeline_data:
      actor = event.get("actor", {}).get("login")
      timestamp_str = event.get("created_at") or event.get("submitted_at")

      if not actor or not timestamp_str or actor.endswith("[bot]"):
        continue

      event["parsed_time"] = dateutil.parser.isoparse(timestamp_str)
      human_events.append(event)

    human_events.sort(key=lambda e: e["parsed_time"])

    # Step 5: Find the most recent, relevant events by iterating backwards.
    last_maintainer_comment = None
    stale_label_event_time = None

    for event in reversed(human_events):
      if (
          not last_maintainer_comment
          and event.get("actor", {}).get("login") in maintainers
          and event.get("event") == "commented"
      ):
        last_maintainer_comment = event

      if (
          not stale_label_event_time
          and event.get("event") == "labeled"
          and event.get("label", {}).get("name") == STALE_LABEL_NAME
      ):
        stale_label_event_time = event["parsed_time"]

      if last_maintainer_comment and stale_label_event_time:
        break

    last_author_action = next(
        (
            e
            for e in reversed(human_events)
            if e.get("actor", {}).get("login") == issue_author
        ),
        None,
    )

    # Step 6: Build and return the final summary report for the LLM agent.
    last_human_event = human_events[-1] if human_events else None
    last_human_actor = (
        last_human_event.get("actor", {}).get("login")
        if last_human_event
        else None
    )

    return {
        "status": "success",
        "issue_author": issue_author,
        "current_labels": current_labels,
        "last_maintainer_comment_text": (
            last_maintainer_comment.get("body")
            if last_maintainer_comment
            else None
        ),
        "last_maintainer_comment_time": (
            last_maintainer_comment["parsed_time"].isoformat()
            if last_maintainer_comment
            else None
        ),
        "last_author_event_time": (
            last_author_action["parsed_time"].isoformat()
            if last_author_action
            else None
        ),
        "last_author_action_type": (
            last_author_action.get("event") if last_author_action else "unknown"
        ),
        "last_human_commenter_is_maintainer": (
            last_human_actor in maintainers if last_human_actor else False
        ),
        "stale_label_applied_at": (
            stale_label_event_time.isoformat()
            if stale_label_event_time
            else None
        ),
    }

  except Exception as e:
    # Provide a detailed error message if the analysis fails.
    return error_response(
        f"Error getting comprehensive issue state for #{item_number}: {e}"
    )


def calculate_time_difference(timestamp_str: str) -> dict[str, Any]:
  """Calculates the difference in hours between a UTC timestamp string and now."""
  try:
    if not timestamp_str:
      return error_response("Input timestamp is empty.")
    event_time = dateutil.parser.isoparse(timestamp_str)
    current_time_utc = datetime.now(timezone.utc)
    time_difference = current_time_utc - event_time
    hours_passed = time_difference.total_seconds() / 3600
    return {"status": "success", "hours_passed": hours_passed}
  except Exception as e:
    return error_response(f"Error calculating time difference: {e}")


def add_label_to_issue(item_number: int, label_name: str) -> dict[str, Any]:
  """Adds a specific label to an issue."""
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels"
  try:
    post_request(url, [label_name])
    return {"status": "success"}
  except Exception as e:
    return error_response(f"Error adding label: {e}")


def remove_label_from_issue(
    item_number: int, label_name: str
) -> dict[str, Any]:
  """Removes a specific label from an issue or PR."""
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels/{label_name}"
  try:
    delete_request(url)
    return {"status": "success"}
  except Exception as e:
    return error_response(f"Error removing label: {e}")


def add_stale_label_and_comment(item_number: int) -> dict[str, Any]:
  """Adds the 'stale' label to an issue and posts a comment explaining why."""
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
  except Exception as e:
    return error_response(f"Error marking issue as stale: {e}")


def close_as_stale(item_number: int) -> dict[str, Any]:
  """Posts a final comment and closes an issue or PR as stale."""
  comment = (
      f"This has been automatically closed because it has been marked as stale"
      f" for over 7 days."
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
  except Exception as e:
    return error_response(f"Error closing issue: {e}")


# --- Agent Definition ---

root_agent = Agent(
    model="gemini-2.5-flash",
    name="adk_repository_auditor_agent",
    description=(
        "Audits open issues to manage their state based on conversation"
        " history."
    ),
    instruction=f"""
      You are a highly intelligent and transparent repository auditor for '{OWNER}/{REPO}'.
      Your job is to analyze all open issues and report on your findings before taking any action.

      **Primary Directive:** Ignore any events from users ending in `[bot]`.
      **Reporting Directive:** For EVERY issue you analyze, you MUST output a concise, human-readable summary, starting with "Analysis for Issue #[number]:".

      **WORKFLOW:**
      1.  **Context Gathering**: Call `get_repository_maintainers` and `get_all_open_issues`.
      2.  **Per-Issue Analysis**: For each issue, call `get_issue_state`, passing in the maintainers list.
      3.  **Decision & Reporting**: Based on the summary from `get_issue_state`, follow this strict decision tree in order.

      --- **DECISION TREE & REPORTING TEMPLATES** ---

      **STEP 1: CHECK FOR ACTIVITY (IS THE ISSUE ACTIVE?)**
      - **Condition**: Was the last human action NOT from a maintainer? (i.e., `last_human_commenter_is_maintainer` is `False`).
      - **Action**: The author or a third party has acted. The issue is ACTIVE.
        - **Report and Action**: If '{STALE_LABEL_NAME}' is present, report: "Analysis for Issue #[number]: Issue is ACTIVE. The last action was a [action type] by a non-maintainer. To get the [action type], you MUST use the value from the 'last_author_action_type' field in the summary you received from the tool. Action: Removing stale label." and then call `remove_label_from_issue`. Otherwise, report: "Analysis for Issue #[number]: Issue is ACTIVE. No stale label to remove. Action: None."
      - **If this condition is met, stop processing this issue.**

      **STEP 2: IF PENDING, MANAGE THE STALE LIFECYCLE.**
      - **Condition**: The last human action WAS from a maintainer (`last_human_commenter_is_maintainer` is `True`). The issue is PENDING.
      - **Action**: You must now determine the correct state.

        - **First, check if the issue is already STALE.**
          - **Condition**: Is the `'{STALE_LABEL_NAME}'` label present in `current_labels`?
          - **Action**: The issue is STALE. Your only job is to check if it should be closed.
            - **Get Time Difference**: Call `calculate_time_difference` with the `stale_label_applied_at` timestamp.
            - **Decision & Report**: If `hours_passed` > **{CLOSE_HOURS_AFTER_STALE_THRESHOLD}**: Report "Analysis for Issue #[number]: STALE. Close threshold met ({CLOSE_HOURS_AFTER_STALE_THRESHOLD} hours) with no author activity. Action: Closing issue." and then call `close_as_stale`. Otherwise, report "Analysis for Issue #[number]: STALE. Close threshold not yet met. Action: None."

        - **ELSE (the issue is PENDING but not yet stale):**
          - **Analyze Intent**: Semantically analyze the `last_maintainer_comment_text`. Is it a question, a request for information, a suggestion, or a request for changes?
          - **If YES (it is a request)**:
            - **CRITICAL CHECK**: Now, you must verify the author has not already responded. Compare the `last_author_event_time` and the `last_maintainer_comment_time`.
            - **IF the author has NOT responded** (i.e., `last_author_event_time` is older than `last_maintainer_comment_time` or is null):
              - **Get Time Difference**: Call `calculate_time_difference` with the `last_maintainer_comment_time`.
              - **Decision & Report**: If `hours_passed` > **{STALE_HOURS_THRESHOLD}**: Report "Analysis for Issue #[number]: PENDING. Stale threshold met ({STALE_HOURS_THRESHOLD} hours). Action: Marking as stale." and then call `add_stale_label_and_comment` and `add_label_to_issue` for '{REQUEST_CLARIFICATION_LABEL}'. Otherwise, report: "Analysis for Issue #[number]: PENDING. Stale threshold not met. Action: None."
            - **ELSE (the author HAS responded)**:
              - **Report**: "Analysis for Issue #[number]: PENDING, but author has already responded to the last maintainer request. Action: None."
          - **If NO (it is not a request):**
            - **Report**: "Analysis for Issue #[number]: PENDING. Maintainer's last comment was not a request. Action: None."
    """,
    tools=[
        get_all_open_issues,
        get_issue_state,
        get_repository_maintainers,
        calculate_time_difference,
        add_stale_label_and_comment,
        add_label_to_issue,
        remove_label_from_issue,
        close_as_stale,
    ],
)
