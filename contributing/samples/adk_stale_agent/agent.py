
import dateutil.parser
from typing import Any
from datetime import datetime, timezone
from adk_stale_agent.settings import (
    GITHUB_BASE_URL, OWNER, REPO, STALE_LABEL_NAME,
    REQUEST_CLARIFICATION_LABEL, STALE_HOURS_THRESHOLD, 
    CLOSE_HOURS_AFTER_STALE_THRESHOLD, ISSUES_PER_RUN
)
from adk_stale_agent.utils import get_request, post_request, patch_request, delete_request, error_response
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
        
        maintainers = [user['login'] for user in collaborators_data]
        print(f"DEBUG: Found {len(maintainers)} maintainers: {maintainers}")
        
        return {"status": "success", "maintainers": maintainers}
    except Exception as e:
        return error_response(f"Error fetching repository maintainers: {e}")

def get_all_open_issues() -> dict[str, Any]:
    """Fetches a batch of the oldest open issues for an audit."""
    print(f"\nDEBUG: Fetching a batch of {ISSUES_PER_RUN} oldest open issues for audit...")
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues"
    params = {"state": "open", "sort": "created", "direction": "asc", "per_page": ISSUES_PER_RUN}
    try:
        items = get_request(url, params)
        print(f"DEBUG: Found {len(items)} open issues to audit.")
        return {"status": "success", "items": items}
    except Exception as e:
        return error_response(f"Error fetching all open issues: {e}")

def get_issue_state(item_number: int, maintainers: list[str]) -> dict[str, Any]:
    """
    Analyzes an issue's complete timeline to determine its current state.
    Requires the list of maintainers to be passed in. Returns a simple, 
    pre-processed summary for the agent to make a decision on.
    """
    try:
        issue_url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}"
        timeline_url = f"{issue_url}/timeline?per_page=100"
        
        issue_data = get_request(issue_url)
        timeline_data = get_request(timeline_url)

        issue_author = issue_data.get('user', {}).get('login')
        current_labels = [label['name'] for label in issue_data.get('labels', [])]

        last_maintainer_event = None
        last_author_event = None
        last_third_party_event = None
        stale_label_event_time = None

        for event in timeline_data:
            actor = event.get('actor', {}).get('login')
            event_type = event.get('event')
            timestamp_str = event.get('created_at') or event.get('submitted_at')
            
            if not timestamp_str or not actor or actor.endswith('[bot]'):
                continue

            timestamp = dateutil.parser.isoparse(timestamp_str)

            if event_type == 'labeled' and event.get('label', {}).get('name') == STALE_LABEL_NAME:
                stale_label_event_time = timestamp

            comment_text = event.get('body') if event_type == 'commented' else None
            
            if actor in maintainers:
                last_maintainer_event = {"actor": actor, "time": timestamp, "text": comment_text}
            elif actor == issue_author:
                last_author_event = {"actor": actor, "time": timestamp, "type": event_type, "text": comment_text}
            else:
                last_third_party_event = {"actor": actor, "time": timestamp, "type": event_type}

        last_human_event = max(
            [e for e in [last_maintainer_event, last_author_event, last_third_party_event] if e],
            key=lambda x: x['time'],
            default=None
        )

        return {
            "status": "success", "issue_author": issue_author, "current_labels": current_labels,
            "last_maintainer_event_time": last_maintainer_event['time'].isoformat() if last_maintainer_event else None,
            "last_maintainer_comment_text": last_maintainer_event['text'] if last_maintainer_event else None,
            "last_author_event_time": last_author_event['time'].isoformat() if last_author_event else None,
            "last_human_commenter_is_maintainer": last_human_event['actor'] in maintainers if last_human_event else False,
            "stale_label_applied_at": stale_label_event_time.isoformat() if stale_label_event_time else None
        }
    except Exception as e:
        return error_response(f"Error getting issue state for #{item_number}: {e}")

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
    except Exception as e: return error_response(f"Error adding label: {e}")

def remove_label_from_issue(item_number: int, label_name: str) -> dict[str, Any]:
    """Removes a specific label from an issue or PR."""
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels/{label_name}"
    try:
        delete_request(url)
        return {"status": "success"}
    except Exception as e: return error_response(f"Error removing label: {e}")

def add_stale_label_and_comment(item_number: int) -> dict[str, Any]:
    """Adds the 'stale' label to an issue and posts a comment explaining why."""
    comment = (
        f"This issue has been automatically marked as stale because it has not had "
        f"recent activity after a maintainer requested clarification. It will be closed if "
        f"no further activity occurs within {CLOSE_HOURS_AFTER_STALE_THRESHOLD / 24:.0f} days."
    )
    try:
        post_request(f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels", [STALE_LABEL_NAME])
        post_request(f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments", {"body": comment})
        return {"status": "success"}
    except Exception as e:
        return error_response(f"Error marking issue as stale: {e}")

def close_as_stale(item_number: int) -> dict[str, Any]:
    """Posts a final comment and closes an issue or PR as stale."""
    comment = (f"This has been automatically closed because it has been marked as stale...")
    try:
        post_request(f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments", {"body": comment})
        patch_request(f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}", {"state": "closed"})
        return {"status": "success"}
    except Exception as e: return error_response(f"Error closing issue: {e}")

# --- Agent Definition ---

root_agent = Agent(
    model="gemini-2.5-flash",
    name="adk_repository_auditor_agent",
    description="Audits open issues to manage their state based on conversation history.",
    instruction=f"""
      You are a highly intelligent repository auditor for '{OWNER}/{REPO}'.
      Your job is to analyze all open issues by first gathering repository-level context and then analyzing each issue individually.

      **Primary Directive:** Ignore any events from users ending in `[bot]`.

      **WORKFLOW:**

      **Phase 1: Context Gathering (Do this ONLY ONCE)**
      1.  Call the `get_repository_maintainers` tool to get the list of maintainers.
      2.  Call the `get_all_open_issues` tool to get the list of candidate issues.

      **Phase 2: Per-Issue Analysis (Loop through the candidates)**
      For each issue you found in Phase 1, you must perform the following steps:
      1.  Call the `get_issue_state` tool. **Crucially, you must pass the list of maintainers you retrieved in Phase 1 as the `maintainers` argument to this tool.**
      2.  Based on the JSON summary from `get_issue_state`, follow the decision tree below.

      --- **DECISION TREE** ---

      **1. CHECK IF ACTIVE:**
      - **Condition**: Is the `last_human_commenter_is_maintainer` field `False`?
      - **Action**: The issue is active. Call `remove_label_from_issue` to remove the '{STALE_LABEL_NAME}' label if it exists.

      **2. IF PENDING, CHECK IF IT SHOULD BECOME STALE:**
      - **Condition**: `last_human_commenter_is_maintainer` is `True`.
      - **Action**: 
        a. **Analyze Intent**: Semantically analyze the `last_maintainer_comment_text`. Is it a question? Is he requesting more information? Is he asking the author to take action?
        b. **If YES**: Check the time. If the author hasn't responded since the maintainer's question, call `calculate_time_difference` with `last_maintainer_event_time`. If the returned `hours_passed` is greater than **{STALE_HOURS_THRESHOLD}**:
           - **Stale Action**:
             i. Call the `add_stale_label_and_comment` tool.
             ii. If the '{REQUEST_CLARIFICATION_LABEL}' label is missing from `current_labels`, call `add_label_to_issue` to add it.

      **3. CHECK IF STALE ISSUE SHOULD BE CLOSED:**
      - **Condition**: The issue is already stale (`'{STALE_LABEL_NAME}'` is in `current_labels`).
      - **Action**: Call `calculate_time_difference` with `stale_label_applied_at`. If the returned `hours_passed` is greater than **{CLOSE_HOURS_AFTER_STALE_THRESHOLD} hours`, call `close_as_stale`.
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