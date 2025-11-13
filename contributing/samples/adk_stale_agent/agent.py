import dateutil.parser
from typing import Any
from datetime import datetime, timezone
from adk_stale_agent.settings import (
    GITHUB_BASE_URL, OWNER, REPO, STALE_LABEL_NAME,
    REQUEST_CLARIFICATION_LABEL, STALE_HOURS_THRESHOLD, 
    CLOSE_HOURS_AFTER_STALE_THRESHOLD, ISSUES_PER_RUN
)
from adk_stale_agent.utils import get_request, put_request, patch_request, post_request, error_response, delete_request
from google.adk.agents.llm_agent import Agent

# --- Primary Tools for the Auditor Agent ---
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
    """Fetches a batch of the oldest open issues and PRs for an audit."""
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
        # --- 1. Fetch all required issue data ---
        issue_url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}"
        timeline_url = f"{issue_url}/timeline?per_page=100"
        
        issue_data = get_request(issue_url)
        timeline_data = get_request(timeline_url)

        # --- 2. Initialize variables for analysis ---
        issue_author = issue_data.get('user', {}).get('login')
        current_labels = [label['name'] for label in issue_data.get('labels', [])]

        last_maintainer_event = None
        last_author_event = None
        last_third_party_event = None
        stale_label_event_time = None

        # --- 3. Iterate through the timeline to find key events ---
        for event in timeline_data:
            actor = event.get('actor', {}).get('login')
            event_type = event.get('event')
            # Use 'created_at' for most events, 'submitted_at' for reviews, etc.
            timestamp_str = event.get('created_at') or event.get('submitted_at')
            
            # Skip malformed events or events without an actor
            if not timestamp_str or not actor:
                continue

            # Convert API timestamp string to a timezone-aware datetime object
            timestamp = dateutil.parser.isoparse(timestamp_str)

            # Primary Directive: Ignore any activity from bots
            if actor.endswith('[bot]'):
                continue

            # Find the most recent time the 'stale' label was added
            if event_type == 'labeled' and event.get('label', {}).get('name') == STALE_LABEL_NAME:
                stale_label_event_time = timestamp

            # Capture the text of comment events for semantic analysis
            comment_text = event.get('body') if event_type == 'commented' else None
            
            # Track the last event from each type of user
            if actor in maintainers:
                last_maintainer_event = {"actor": actor, "time": timestamp, "text": comment_text}
            elif actor == issue_author:
                last_author_event = {"actor": actor, "time": timestamp, "type": event_type, "text": comment_text}
            else: # Any other human user
                last_third_party_event = {"actor": actor, "time": timestamp, "type": event_type}

        # --- 4. Determine the absolute last human event ---
        last_human_event = max(
            [e for e in [last_maintainer_event, last_author_event, last_third_party_event] if e],
            key=lambda x: x['time'],
            default=None
        )

        # --- 5. Return the clean, simple summary report for the LLM ---
        return {
            "status": "success",
            "issue_author": issue_author,
            "current_labels": current_labels,
            "last_maintainer_event_time": last_maintainer_event['time'].isoformat() if last_maintainer_event else None,
            "last_maintainer_comment_text": last_maintainer_event['text'] if last_maintainer_event else None,
            "last_author_event_time": last_author_event['time'].isoformat() if last_author_event else None,
            "last_author_comment_text": last_author_event['text'] if last_author_event else None,
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

        print(f"\nDEBUG (Python Time Calculation):")
        print(f"  - Event Time (UTC):    {event_time}")
        print(f"  - Current Time (UTC):  {current_time_utc}")
        print(f"  - Hours Passed:        {hours_passed:.2f}\n")
        return {"status": "success", "hours_passed": hours_passed}
    except Exception as e:
        return error_response(f"Error calculating time difference: {e}")

# We need both add and remove for the final logic
def add_label_to_issue(item_number: int, label_name: str) -> dict[str, Any]:
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels"
    try:
        post_request(url, [label_name])
        return {"status": "success"}
    except Exception as e: return error_response(f"Error adding label: {e}")

def remove_label_from_issue(item_number: int, label_name: str) -> dict[str, Any]:
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels/{label_name}"
    try:
        delete_request(url)
        return {"status": "success"}
    except Exception as e: return error_response(f"Error removing label: {e}")

def close_as_stale(item_number: int) -> dict[str, Any]:
    comment = (f"This has been automatically closed because it has been marked as stale...")
    try:
        post_request(f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments", {"body": comment})
        patch_request(f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}", {"state": "closed"})
        return {"status": "success"}
    except Exception as e: return error_response(f"Error closing issue: {e}")


# --- The Final "Auditor" Agent Definition ---

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
        a. **Analyze Intent**: Semantically analyze the `last_maintainer_comment_text`. Is it a question?
        b. **If YES**: Check the time. If the author hasn't responded since the maintainer's question and the `last_maintainer_event_time` is older than **{STALE_HOURS_THRESHOLD} hours**:
           - **Stale Action**: Add both the '{STALE_LABEL_NAME}' and '{REQUEST_CLARIFICATION_LABEL}' labels if they are missing.

      **3. CHECK IF STALE ISSUE SHOULD BE CLOSED:**
      - **Condition**: The issue is already stale (`'{STALE_LABEL_NAME}'` is in `current_labels`).
      - **Action**: If the `stale_label_applied_at` timestamp is older than **{CLOSE_HOURS_AFTER_STALE_THRESHOLD} hours`, call `close_as_stale`.
    """,
    tools=[
        get_all_open_issues,
        get_issue_state,
        calculate_time_difference,
        add_label_to_issue,
        remove_label_from_issue,
        close_as_stale,
        get_repository_maintainers
    ],
)