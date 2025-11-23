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
from datetime import datetime, timedelta, timezone
import dateutil.parser
from typing import Any, Dict, List, Optional

import requests
from adk_stale_agent.settings import GITHUB_TOKEN, STALE_HOURS_THRESHOLD
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Module-level logger setup ---
logger = logging.getLogger("google_adk." + __name__)

# --- API Call Counter for Monitoring ---
_api_call_count = 0


def get_api_call_count() -> int:
    """Returns the total number of API calls made since the last reset."""
    return _api_call_count


def reset_api_call_count() -> None:
    """Resets the global API call counter to zero."""
    global _api_call_count
    _api_call_count = 0


def _increment_api_call_count() -> None:
    """Atomically increments the global API call counter."""
    global _api_call_count
    _api_call_count += 1


# --- Production-Ready HTTP Session with Exponential Backoff ---

# Configure the retry strategy. This implements exponential backoff automatically.
# - total=6: Allow up to 6 total retries.
# - backoff_factor=2: A key factor for exponential delay. The time between retries
#   will be {backoff_factor} * (2 ** ({number_of_retries} - 1)).
#   e.g., waits for [2s, 4s, 8s, 16s, 32s] between retries.
# - status_forcelist: A set of HTTP status codes that will trigger a retry.
#   These are common codes for temporary server errors or rate limiting.
retry_strategy = Retry(
    total=6,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE", "PATCH"],
)

# Create an adapter with the retry strategy.
adapter = HTTPAdapter(max_retries=retry_strategy)

# Create a single, reusable Session object for the entire application.
# This is crucial for performance as it enables connection pooling.
_session = requests.Session()

# Mount the adapter to the session for both http and https protocols.
_session.mount("https://", adapter)
_session.mount("http://", adapter)

# Set common headers for all requests made with this session.
_session.headers.update({
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
})


def get_request(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Sends a GET request to the GitHub API with configured retries.

    Args:
        url: The URL endpoint to send the request to.
        params: An optional dictionary of URL parameters.

    Returns:
        The JSON response from the API as a dictionary or list.

    Raises:
        requests.exceptions.RequestException: For network errors or HTTP status
                                              codes that are not resolved by retries.
    """
    _increment_api_call_count()
    try:
        response = _session.get(url, params=params or {}, timeout=60)
        response.raise_for_status()  # Raise an exception for HTTP error codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"GET request failed for {url}: {e}")
        raise


def post_request(url: str, payload: Any) -> Any:
    """
    Sends a POST request to the GitHub API with configured retries.

    Args:
        url: The URL endpoint to send the request to.
        payload: The JSON payload to send with the request.

    Returns:
        The JSON response from the API as a dictionary or list.

    Raises:
        requests.exceptions.RequestException: For network errors or HTTP status
                                              codes that are not resolved by retries.
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
    Sends a PATCH request to the GitHub API with configured retries.

    Args:
        url: The URL endpoint to send the request to.
        payload: The JSON payload to send with the request.

    Returns:
        The JSON response from the API as a dictionary or list.

    Raises:
        requests.exceptions.RequestException: For network errors or HTTP status
                                              codes that are not resolved by retries.
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
    Sends a DELETE request to the GitHub API with configured retries.

    Args:
        url: The URL endpoint to send the request to.

    Returns:
        A success dictionary for 204 status, otherwise the JSON response.

    Raises:
        requests.exceptions.RequestException: For network errors or HTTP status
                                              codes that are not resolved by retries.
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
        error_message: A descriptive message of the error that occurred.

    Returns:
        A dictionary containing the error status and message.
    """
    return {"status": "error", "message": error_message}


def get_old_open_issue_numbers(
    owner: str, repo: str, days_old: Optional[float] = None
) -> List[int]:
    """
    Finds open issues older than the precise `days_old` threshold.

    This function first fetches ALL open issues from the repository and then
    applies a precise, client-side filter to find the ones that are
    older than the specified threshold.
    """
    if days_old is None:
        days_old = STALE_HOURS_THRESHOLD / 24

    # 1. Calculate the PRECISE cutoff time in UTC.
    now_utc = datetime.now(timezone.utc)
    precise_cutoff_datetime = now_utc - timedelta(days=days_old)

    # 2. Build a query to get ALL open issues. The date filter is removed.
    query = f"repo:{owner}/{repo} is:issue state:open"
    logger.info(f"Fetching all open issues from '{owner}/{repo}'...")

    all_open_issues = []
    page = 1
    url = "https://api.github.com/search/issues"

    # Stage 1: Fetch all open issues via API
    while True:
        params = {"q": query, "per_page": 100, "page": page}
        try:
            data = get_request(url, params=params)
            items = data.get("items", [])
            if not items:
                break
            
            all_open_issues.extend(items)

            if len(items) < 100:
                break
            page += 1
        except requests.exceptions.RequestException as e:
            logger.error(f"GitHub search failed on page {page}: {e}")
            break

    logger.info(
        f"Fetched {len(all_open_issues)} total open issues. "
        f"Now filtering for those created before: {precise_cutoff_datetime.isoformat()}"
    )

    # Stage 2: Apply the precise time filter in Python
    final_issue_numbers = []
    for item in all_open_issues:
        if "pull_request" in item:
            continue

        issue_creation_time = dateutil.parser.isoparse(item["created_at"])

        if issue_creation_time < precise_cutoff_datetime:
            final_issue_numbers.append(item["number"])

    logger.info(f"Found {len(final_issue_numbers)} issues that are older than the threshold.")
    return final_issue_numbers