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
from datetime import timedelta
from datetime import timezone
import logging
import threading
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from adk_stale_agent.settings import GITHUB_TOKEN
from adk_stale_agent.settings import STALE_HOURS_THRESHOLD
import dateutil.parser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("google_adk." + __name__)

# --- API Call Counter for Monitoring ---
_api_call_count = 0
_counter_lock = threading.Lock()


def get_api_call_count() -> int:
  """
  Returns the total number of API calls made since the last reset.

  Returns:
      int: The global count of API calls.
  """
  with _counter_lock:
    return _api_call_count


def reset_api_call_count() -> None:
  """Resets the global API call counter to zero."""
  global _api_call_count
  with _counter_lock:
    _api_call_count = 0


def _increment_api_call_count() -> None:
  """
  Atomically increments the global API call counter.
  Required because the agent may run tools in parallel threads.
  """
  global _api_call_count
  with _counter_lock:
    _api_call_count += 1


# --- Production-Ready HTTP Session with Exponential Backoff ---

# Configure the retry strategy:
retry_strategy = Retry(
    total=6,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=[
        "HEAD",
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "OPTIONS",
        "TRACE",
        "PATCH",
    ],
)

adapter = HTTPAdapter(max_retries=retry_strategy)

# Create a single, reusable Session object for connection pooling
_session = requests.Session()
_session.mount("https://", adapter)
_session.mount("http://", adapter)

_session.headers.update({
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
})


def get_request(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
  """
  Sends a GET request to the GitHub API with automatic retries.

  Args:
      url (str): The URL endpoint.
      params (Optional[Dict[str, Any]]): Query parameters.

  Returns:
      Any: The JSON response parsed into a dict or list.

  Raises:
      requests.exceptions.RequestException: If retries are exhausted.
  """
  _increment_api_call_count()
  try:
    response = _session.get(url, params=params or {}, timeout=60)
    response.raise_for_status()
    return response.json()
  except requests.exceptions.RequestException as e:
    logger.error(f"GET request failed for {url}: {e}")
    raise


def post_request(url: str, payload: Any) -> Any:
  """
  Sends a POST request to the GitHub API with automatic retries.

  Args:
      url (str): The URL endpoint.
      payload (Any): The JSON payload.

  Returns:
      Any: The JSON response.
  """
  _increment_api_call_count()
  try:
    response = _session.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()
  except requests.exceptions.RequestException as e:
    logger.error(f"POST request failed for {url}: {e}")
    raise


def patch_request(url: str, payload: Any) -> Any:
  """
  Sends a PATCH request to the GitHub API with automatic retries.

  Args:
      url (str): The URL endpoint.
      payload (Any): The JSON payload.

  Returns:
      Any: The JSON response.
  """
  _increment_api_call_count()
  try:
    response = _session.patch(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()
  except requests.exceptions.RequestException as e:
    logger.error(f"PATCH request failed for {url}: {e}")
    raise


def delete_request(url: str) -> Any:
  """
  Sends a DELETE request to the GitHub API with automatic retries.

  Args:
      url (str): The URL endpoint.

  Returns:
      Any: A success dict if 204, else the JSON response.
  """
  _increment_api_call_count()
  try:
    response = _session.delete(url, timeout=60)
    response.raise_for_status()
    if response.status_code == 204:
      return {"status": "success", "message": "Deletion successful."}
    return response.json()
  except requests.exceptions.RequestException as e:
    logger.error(f"DELETE request failed for {url}: {e}")
    raise


def error_response(error_message: str) -> Dict[str, Any]:
  """
  Creates a standardized error response dictionary for tool outputs.

  Args:
      error_message (str): The error details.

  Returns:
      Dict[str, Any]: Standardized error object.
  """
  return {"status": "error", "message": error_message}


def get_old_open_issue_numbers(
    owner: str, repo: str, days_old: Optional[float] = None
) -> List[int]:
  """
  Finds open issues older than the specified threshold using server-side filtering.

  OPTIMIZATION:
  Instead of fetching ALL issues and filtering in Python (which wastes API calls),
  this uses the GitHub Search API `created:<DATE` syntax.

  Args:
      owner (str): Repository owner.
      repo (str): Repository name.
      days_old (Optional[float]): Filter issues older than this many days.
                                  Defaults to STALE_HOURS_THRESHOLD / 24.

  Returns:
      List[int]: A list of issue numbers matching the criteria.
  """
  if days_old is None:
    days_old = STALE_HOURS_THRESHOLD / 24

  now_utc = datetime.now(timezone.utc)
  cutoff_dt = now_utc - timedelta(days=days_old)

  cutoff_str = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

  query = f"repo:{owner}/{repo} is:issue state:open created:<{cutoff_str}"

  logger.info(
      f"Searching for issues in '{owner}/{repo}' created before {cutoff_str}..."
  )

  issue_numbers = []
  page = 1
  url = "https://api.github.com/search/issues"

  while True:
    params = {"q": query, "per_page": 100, "page": page}
    try:
      data = get_request(url, params=params)
      items = data.get("items", [])

      if not items:
        break

      for item in items:
        if "pull_request" not in item:
          issue_numbers.append(item["number"])

      if len(items) < 100:
        break

      page += 1

    except requests.exceptions.RequestException as e:
      logger.error(f"GitHub search failed on page {page}: {e}")
      break

  logger.info(f"Found {len(issue_numbers)} stale issues.")
  return issue_numbers
