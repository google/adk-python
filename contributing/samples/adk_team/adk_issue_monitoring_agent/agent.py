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

import logging
import os
from typing import Any

from adk_issue_monitoring_agent.settings import BOT_ALERT_SIGNATURE
from adk_issue_monitoring_agent.settings import GITHUB_BASE_URL
from adk_issue_monitoring_agent.settings import LLM_MODEL_NAME
from adk_issue_monitoring_agent.settings import OWNER
from adk_issue_monitoring_agent.settings import REPO
from adk_issue_monitoring_agent.settings import SPAM_LABEL_NAME
from adk_issue_monitoring_agent.utils import error_response
from adk_issue_monitoring_agent.utils import get_issue_comments
from adk_issue_monitoring_agent.utils import get_issue_details
from adk_issue_monitoring_agent.utils import post_request
from google.adk.agents.llm_agent import Agent
from requests.exceptions import RequestException

logger = logging.getLogger("google_adk." + __name__)


def load_prompt_template(filename: str) -> str:
  file_path = os.path.join(os.path.dirname(__file__), filename)
  with open(file_path, "r") as f:
    return f.read()


PROMPT_TEMPLATE = load_prompt_template("PROMPT_INSTRUCTION.txt")

# --- Tools ---


def flag_issue_as_spam(
    item_number: int, detection_reason: str
) -> dict[str, Any]:
  """
  Flags an issue as spam by adding a label and leaving a comment for maintainers.
  Includes idempotency checks to avoid duplicate POST actions.

  Args:
      item_number (int): The GitHub issue number.
      detection_reason (str): The explanation of what the spam is.
  """
  logger.info(f"Flagging #{item_number} as SPAM. Reason: {detection_reason}")

  label_url = (
      f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels"
  )
  comment_url = (
      f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments"
  )

  safe_reason = detection_reason.replace("```", "'''")

  alert_body = (
      f"{BOT_ALERT_SIGNATURE}\n"
      "@maintainers, a suspected spam comment was detected in this thread.\n\n"
      "**Reason:**\n"
      f"```text\n{safe_reason}\n```"
  )

  try:
    # 1. Fetch current state to check what actions are actually needed
    issue = get_issue_details(OWNER, REPO, item_number)
    comments = get_issue_comments(OWNER, REPO, item_number)

    current_labels = [
        label["name"].lower() for label in issue.get("labels", [])
    ]
    is_labeled = SPAM_LABEL_NAME.lower() in current_labels
    is_commented = any(
        BOT_ALERT_SIGNATURE in c.get("body", "") for c in comments
    )

    if is_labeled and is_commented:
      logger.info(f"#{item_number} is already labeled and commented. Skipping.")
    elif is_labeled and not is_commented:
      post_request(comment_url, {"body": alert_body})
      logger.info(f"Successfully posted spam alert comment to #{item_number}.")
    elif not is_labeled and is_commented:
      post_request(label_url, {"labels": [SPAM_LABEL_NAME]})
      logger.info(
          f"Successfully added '{SPAM_LABEL_NAME}' label to #{item_number}."
      )
    else:
      post_request(label_url, {"labels": [SPAM_LABEL_NAME]})
      post_request(comment_url, {"body": alert_body})
      logger.info(f"Successfully fully flagged #{item_number}.")

    return {"status": "success", "message": "Maintainers alerted successfully."}

  except RequestException as e:
    return error_response(f"Error flagging issue: {e}")


root_agent = Agent(
    model=LLM_MODEL_NAME,
    name="spam_auditor_agent",
    description="Audits issue comments for spam.",
    instruction=PROMPT_TEMPLATE.format(
        OWNER=OWNER,
        REPO=REPO,
    ),
    tools=[flag_issue_as_spam],
)
