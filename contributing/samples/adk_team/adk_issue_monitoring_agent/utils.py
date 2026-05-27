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
from typing import Any

from adk_issue_monitoring_agent.settings import GITHUB_TOKEN
from adk_issue_monitoring_agent.settings import INITIAL_FULL_SCAN
from adk_issue_monitoring_agent.settings import SPAM_LABEL_NAME
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("google_adk." + __name__)

_api_call_count = 0


def get_api_call_count() -> int:
  return _api_call_count


def reset_api_call_count() -> None:
  global _api_call_count
  _api_call_count = 0


def _increment_api_call_count() -> None:
  global _api_call_count
  _api_call_count += 1


retry_strategy = Retry(
    total=6,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "DELETE"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
_session = requests.Session()
_session.mount("https://", adapter)
_session.headers.update({
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
})


def get_request(url: str, params: dict[str, Any] | None = None) -> Any:
  _increment_api_call_count()
  response = _session.get(url, params=params or {}, timeout=60)
  response.raise_for_status()
  return response.json()


def post_request(url: str, payload: Any) -> Any:
  _increment_api_call_count()
  response = _session.post(url, json=payload, timeout=60)
  response.raise_for_status()
  return response.json()


def error_response(error_message: str) -> dict[str, Any]:
  return {"status": "error", "message": error_message}


def get_repository_maintainers(owner: str, repo: str) -> list[str]:
  """Fetches all users with push/maintain access."""
  url = f"https://api.github.com/repos/{owner}/{repo}/collaborators"
  data = get_request(url, {"permission": "push"})
  return [user["login"] for user in data]


def get_issue_details(
    owner: str, repo: str, issue_number: int
) -> dict[str, Any]:
  """Fetches the main issue object to get the original description (body)."""
  url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
  return get_request(url)


def get_issue_comments(
    owner: str, repo: str, issue_number: int
) -> list[dict[str, Any]]:
  """Fetches ALL comments for a specific issue, handling pagination."""
  url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
  all_comments = []
  page = 1

  while True:
    data = get_request(url, params={"per_page": 100, "page": page})
    if not data:
      break

    all_comments.extend(data)

    if len(data) < 100:
      break
    page += 1

  return all_comments


def get_target_issues(owner: str, repo: str) -> list[int]:
  """
  Fetches issues.
  If INITIAL_FULL_SCAN is True, fetches ALL open issues.
  If False, fetches only issues updated in the last 24 hours using the 'since' parameter.
  """
  from datetime import datetime
  from datetime import timedelta
  from datetime import timezone

  url = f"https://api.github.com/repos/{owner}/{repo}/issues"
  params = {
      "state": "open",
      "per_page": 100,
  }

  if INITIAL_FULL_SCAN:
    logger.info("INITIAL_FULL_SCAN is True. Fetching ALL open issues...")
  else:
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    params["since"] = yesterday
    logger.info(f"Daily mode: Fetching issues updated since {yesterday}...")

  issue_numbers = []
  page = 1

  while True:
    params["page"] = page
    try:
      items = get_request(url, params=params)

      if not items:
        break

      for item in items:
        if "pull_request" not in item:
          # Extract all the label names on this issue
          current_labels = [label["name"] for label in item.get("labels", [])]

          # Only add the issue if it DOES NOT already have the spam label
          if SPAM_LABEL_NAME not in current_labels:
            issue_numbers.append(item["number"])
          else:
            logger.debug(
                f"Skipping #{item['number']} - already marked as spam."
            )

      if len(items) < 100:
        break

      page += 1
    except requests.exceptions.RequestException as e:
      logger.error(f"Failed to fetch issues on page {page}: {e}")
      break

  return issue_numbers
