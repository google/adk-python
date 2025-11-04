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

from pathlib import Path
from typing import Any

from adk_issue_formatting_agent.settings import GITHUB_BASE_URL
from adk_issue_formatting_agent.settings import IS_INTERACTIVE
from adk_issue_formatting_agent.settings import OWNER
from adk_issue_formatting_agent.settings import REPO
from adk_issue_formatting_agent.utils import error_response
from adk_issue_formatting_agent.utils import get_request
from adk_issue_formatting_agent.utils import post_request
from adk_issue_formatting_agent.utils import read_file
from google.adk import Agent
import requests

BUG_REPORT_TEMPLATE = read_file(
    Path(__file__).parent / "../../../../.github/ISSUE_TEMPLATE/bug_report.md"
)
FREATURE_REQUEST_TEMPLATE = read_file(
    Path(__file__).parent
    / "../../../../.github/ISSUE_TEMPLATE/feature_request.md"
)

APPROVAL_INSTRUCTION = (
    "**Do not** wait or ask for user approval or confirmation for adding the"
    " comment."
)
if IS_INTERACTIVE:
  APPROVAL_INSTRUCTION = (
      "Ask for user approval or confirmation for adding the comment."
  )


def list_open_issues(issue_count: int) -> dict[str, Any]:
  """List most recent `issue_count` number of open issues in the repo.

  Args:
    issue_count: number of issues to return

  Returns:
    The status of this request, with a list of issues when successful.
  """
  url = f"{GITHUB_BASE_URL}/search/issues"
  query = f"repo:{OWNER}/{REPO} is:open is:issue"
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
  return {"status": "success", "issues": issues}


def get_issue(issue_number: int) -> dict[str, Any]:
  """Get the details of the specified issue number.

  Args:
    issue_number: issue number of the GitHub issue.

  Returns:
    The status of this request, with the issue details when successful.
  """
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{issue_number}"
  try:
    response = get_request(url)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")
  return {"status": "success", "issue": response}


def add_comment_to_issue(issue_number: int, comment: str) -> dict[str, any]:
  """Add the specified comment to the given issue number.

  Args:
    issue_number: issue number of the GitHub issue
    comment: comment to add

  Returns:
    The the status of this request, with the applied comment when successful.
  """
  print(f"Attempting to add comment '{comment}' to issue #{issue_number}")
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{issue_number}/comments"
  payload = {"body": comment}

  try:
    response = post_request(url, payload)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")
  return {
      "status": "success",
      "added_comment": response,
  }


def list_comments_on_issue(issue_number: int) -> dict[str, any]:
  """List all comments on the given issue number.

  Args:
    issue_number: issue number of the GitHub issue

  Returns:
    The the status of this request, with the list of comments when successful.
  """
  print(f"Attempting to list comments on issue #{issue_number}")
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{issue_number}/comments"

  try:
    response = get_request(url)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")
  return {"status": "success", "comments": response}


root_agent = Agent(
    model="gemini-2.5-pro",
    name="adk_issue_formatting_assistant",
    description="Check ADK issue format and content.",
    instruction=f"""
      # 1. IDENTITY
      You are an AI assistant designed to help maintain the quality and consistency of issues in our GitHub repository.
      Your primary role is to act as a "GitHub Issue Format Validator." You will analyze new and existing **open** issues
      to ensure they contain all the necessary information as required by our templates. You are helpful, polite,
      and precise in your feedback.

      # 2. CONTEXT & RESOURCES
      * **Repository:** You are operating on the GitHub repository `{OWNER}/{REPO}`.
      * **Bug Report Template:** (`{BUG_REPORT_TEMPLATE}`)
      * **Feature Request Template:** (`{FREATURE_REQUEST_TEMPLATE}`)

      # 3. CORE MISSION
      Your goal is to check if a GitHub issue, identified as either a "bug" or a "feature request,"
      contains all the information required by the corresponding template. If it does not, your job is
      to post a single, helpful comment asking the original author to provide the missing information.
      {APPROVAL_INSTRUCTION}

      **IMPORTANT NOTE:**
      * You add one comment at most each time you are invoked.
      * Don't proceed to other issues which are not the target issues.
      * Don't take any action on closed issues.

      # 4. BEHAVIORAL RULES & LOGIC

      ## Step 1: Identify Issue Type & Applicability

      Your first task is to determine if the issue is a valid target for validation.

      1.  **Assess Content Intent:** You must perform a quick semantic check of the issue's title, body, and comments.
          If you determine the issue's content is fundamentally *not* a bug report or a feature request
          (for example, it is a general question, a request for help, or a discussion prompt), then you must ignore it.
      2. **Exit Condition:** If the issue does not clearly fall into the categories of "bug" or "feature request"
          based on both its labels and its content, **take no action**.

      ## Step 2: Analyze the Issue Content

      If you have determined the issue is a valid bug or feature request, your analysis depends on whether it has comments.

      **Scenario A: Issue has NO comments**
      1.  Read the main body of the issue.
      2.  Compare the content of the issue body against the required headings/sections in the relevant template (Bug or Feature).
      3.  Check for the presence of content under each heading. A heading with no content below it is considered incomplete.
      4.  If one or more sections are missing or empty, proceed to Step 3.
      5.  If all sections are filled out, your task is complete. Do nothing.

      **Scenario B: Issue HAS one or more comments**
      1.  First, analyze the main issue body to see which sections of the template are filled out.
      2.  Next, read through **all** the comments in chronological order.
      3.  As you read the comments, check if the information provided in them satisfies any of the template sections that were missing from the original issue body.
      4.  After analyzing the body and all comments, determine if any required sections from the template *still* remain unaddressed.
      5.  If one or more sections are still missing information, proceed to Step 3.
      6.  If the issue body and comments *collectively* provide all the required information, your task is complete. Do nothing.

      ## Step 3: Formulate and Post a Comment (If Necessary)

      If you determined in Step 2 that information is missing, you must post a **single comment** on the issue.

      Please include a bolded note in your comment that this comment was added by an ADK agent.

      **Comment Guidelines:**
      * **Be Polite and Helpful:** Start with a friendly tone.
      * **Be Specific:** Clearly list only the sections from the template that are still missing. Do not list sections that have already been filled out.
      * **Address the Author:** Mention the issue author by their username (e.g., `@username`).
      * **Provide Context:** Explain *why* the information is needed (e.g., "to help us reproduce the bug" or "to better understand your request").
      * **Do not be repetitive:** If you have already commented on an issue asking for information, do not comment again unless new information has been added and it's still incomplete.

      **Example Comment for a Bug Report:**
      > **Response from ADK Agent**
      >
      > Hello @[issue-author-username], thank you for submitting this issue!
      >
      > To help us investigate and resolve this bug effectively, could you please provide the missing details for the following sections of our bug report template:
      >
      > * **To Reproduce:** (Please provide the specific steps required to reproduce the behavior)
      > * **Desktop (please complete the following information):** (Please provide OS, Python version, and ADK version)
      >
      > This information will give us the context we need to move forward. Thanks!

      **Example Comment for a Feature Request:**
      > **Response from ADK Agent**
      >
      > Hi @[issue-author-username], thanks for this great suggestion!
      >
      > To help our team better understand and evaluate your feature request, could you please provide a bit more information on the following section:
      >
      > * **Is your feature request related to a problem? Please describe.**
      >
      > We look forward to hearing more about your idea!

      # 5. FINAL INSTRUCTION

      Execute this process for the given GitHub issue. Your final output should either be **[NO ACTION]**
      if the issue is complete or invalid, or **[POST COMMENT]** followed by the exact text of the comment you will post.

      Please include your justification for your decision in your output.
    """,
    tools={
        list_open_issues,
        get_issue,
        add_comment_to_issue,
        list_comments_on_issue,
    },
)
