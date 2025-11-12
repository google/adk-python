from typing import Any
from adk_stale_bot.settings import (
    GITHUB_BASE_URL, OWNER, REPO, MAINTAINERS, STALE_LABEL_NAME,
    STALE_HOURS_THRESHOLD, CLOSE_HOURS_AFTER_STALE_THRESHOLD
)
from adk_stale_bot.utils import get_request, post_request, patch_request, error_response
from google.adk.agents.llm_agent import Agent

# --- Tool Functions for the Stale Agent ---

def get_open_issues_and_prs() -> dict[str, Any]:
    """Fetches all open issues and PRs to check for staleness."""
    url = f"{GITHUB_BASE_URL}/search/issues"
    query = f"repo:{OWNER}/{REPO} is:open"
    params = {"q": query, "sort": "updated", "order": "desc", "per_page": 100}
    try:
        response = get_request(url, params)
        return {"status": "success", "items": response.get("items", [])}
    except Exception as e:
        return error_response(f"Error fetching issues/PRs: {e}")

def get_issue_details(item_number: int) -> dict[str, Any]:
    """Gets comments and labeling events for a specific issue or PR."""
    try:
        issue_url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}"
        comments_url = f"{issue_url}/comments?sort=created&direction=desc&per_page=100"
        events_url = f"{issue_url}/events?per_page=100"
        return {
            "status": "success",
            "issue": get_request(issue_url),
            "comments": get_request(comments_url),
            "events": get_request(events_url),
        }
    except Exception as e:
        return error_response(f"Error fetching details for #{item_number}: {e}")

def add_stale_label(item_number: int) -> dict[str, Any]:
    """Adds the 'stale' label to an issue or PR."""
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels"
    try:
        post_request(url, [STALE_LABEL_NAME])
        return {"status": "success", "message": f"Label '{STALE_LABEL_NAME}' added to #{item_number}."}
    except Exception as e:
        return error_response(f"Error adding label to #{item_number}: {e}")

def close_as_stale(item_number: int) -> dict[str, Any]:
    """Posts a final comment and closes an issue or PR as stale."""
    comment = (
        f"This has been automatically closed because it has been marked as stale "
        f"and received no further activity. If you feel this is in error, please feel free "
        f"to re-open this and provide the requested information."
    )
    try:
        # 1. Add comment
        comments_url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments"
        post_request(comments_url, {"body": comment})
        # 2. Close issue/PR
        issue_url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}"
        patch_request(issue_url, {"state": "closed"})
        return {"status": "success", "message": f"Successfully commented on and closed #{item_number}."}
    except Exception as e:
        return error_response(f"Error closing #{item_number}: {e}")

# --- The Stale Agent Definition ---

root_agent = Agent(
    model="gemini-2.5-pro",
    name="adk_stale_issue_agent",
    description="Manages stale issues and PRs in the ADK repository.",
    instruction=f"""
      You are an autonomous repository maintenance bot for '{OWNER}/{REPO}'.
      Your task is to identify and process items (issues or PRs) that have become stale.

      Here is your logic flow:
      1.  **Get Items**: Call `get_open_issues_and_prs` to get a list of items to analyze.
      2.  **Analyze Each Item**: For each item, call `get_issue_details` to fetch its data.
      3.  **Check for Stale Condition**:
          - An item should be marked `stale` IF ALL of these are true:
            a. It does NOT currently have the '{STALE_LABEL_NAME}' label.
            b. Its most recent comment is from a maintainer (one of: {', '.join(MAINTAINERS)}).
            c. The original author of the item has NOT commented after that last maintainer comment.
            d. It has been more than **{STALE_HOURS_THRESHOLD} hours** since the maintainer's comment.

      4.  **Check for Close Condition**:
          - An item should be closed IF ALL of these are true:
            a. It HAS the '{STALE_LABEL_NAME}' label.
            b. There has been no activity (e.g., new comments from anyone) since the label was applied. You must check the timestamp of the last comment against the timestamp of when the stale label was applied.
            c. It has been more than **{CLOSE_HOURS_AFTER_STALE_THRESHOLD} hours** since the '{STALE_LABEL_NAME}' label was applied. You find this date by looking in the 'events' data for a 'labeled' event for '{STALE_LABEL_NAME}'.

      You must carefully parse dates and authors from the tool outputs and calculate the time difference in **hours** to make your decisions.
    """,
    tools=[
        get_open_issues_and_prs,
        get_issue_details,
        add_stale_label,
        close_as_stale,
    ],
)