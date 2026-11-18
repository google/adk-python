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
from __future__ import annotations

import asyncio
from collections.abc import Callable
from collections.abc import Mapping
import json
import re
import time
from typing import Any

from google.auth.credentials import Credentials
import requests

from .. import _gda_stream_util
from ..tool_context import ToolContext
from .config import DataAgentToolConfig

_GDA_CLIENT_ID = "GOOGLE_ADK"
_GDA_REQUEST_TIMEOUT_SECONDS = 30
_MIN_PAGE_SIZE = 1
_MAX_AUTO_DATA_AGENTS = 1000
_MAX_AUTO_PAGES = 100
# Bounds how long one automatic drain may run. This function is synchronous, so
# the budget is also the worst case for how long it can occupy the caller's
# event loop. It must stay above 2 * _GDA_REQUEST_TIMEOUT_SECONDS plus a backoff
# so that a first attempt which burns its full request timeout can still be
# retried; see the guard in _fetch_page_with_retries.
_MAX_AUTO_DURATION_SECONDS = 90
_MAX_PAGE_ATTEMPTS = 3
_PAGE_RETRY_BACKOFF_SECONDS = 0.5
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_SEGMENT = r"[a-zA-Z0-9][a-zA-Z0-9_.-]*"

_PAGINATION_TIMEOUT_MSG = (
    "timed out while fetching accessible data agents; set 'page_size' and"
    " 'page_token' to page through them instead"
)

_DATA_AGENT_NAME_RE = re.compile(
    rf"\Aprojects/{_SEGMENT}/locations/{_SEGMENT}/dataAgents/{_SEGMENT}\Z"
)
_SEGMENT_RE = re.compile(rf"\A{_SEGMENT}\Z")


def _validate_data_agent_name(data_agent_name: str) -> dict[str, Any] | None:
  """Validates data_agent_name format."""
  if not _DATA_AGENT_NAME_RE.match(data_agent_name):
    return {
        "status": "ERROR",
        "error_details": (
            "Invalid data_agent_name format. Expected format:"
            " projects/{project}/locations/{location}/dataAgents/{agent},"
            f" got: '{data_agent_name}'"
        ),
    }
  return None


def _validate_path_segment(
    value: str, field_name: str
) -> dict[str, Any] | None:
  """Validates a URL path segment format."""
  if not _SEGMENT_RE.match(value):
    return {
        "status": "ERROR",
        "error_details": (
            f"Invalid {field_name} format. Expected alphanumeric characters,"
            f" hyphens, underscores, or periods, got: '{value}'"
        ),
    }
  return None


def _gda_headers() -> dict[str, str]:
  """Returns the standard headers used for all GDA API requests."""
  return {
      "Content-Type": "application/json",
      "X-Goog-API-Client": _GDA_CLIENT_ID,
  }


def _modification_disabled_error(
    settings: DataAgentToolConfig,
) -> dict[str, Any] | None:
  """Returns an error dict if data agent mutation is disabled, else None."""
  if settings.enable_data_agent_modification:
    return None
  return {
      "status": "ERROR",
      "error_details": (
          "Data agent mutation is disabled. Enable it by setting "
          "`enable_data_agent_modification=True` in DataAgentToolConfig."
      ),
  }


def _parse_agent_config(
    agent_config: str | dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """Parses agent_config into a dict.

  The public tool signatures annotate agent_config as `str` so the generated
  function-calling schema stays a plain string (a `str | dict` union emits an
  `anyOf` with a property-less object, which the backend rejects). ADK does not
  coerce primitive argument types at runtime, so a dict can still arrive here
  from a Python caller or a middleware layer that pre-parses tool arguments --
  accept it instead of failing the call.
  """
  try:
    parsed_config = (
        agent_config
        if isinstance(agent_config, dict)
        else json.loads(agent_config)
    )
    if not isinstance(parsed_config, dict):
      raise TypeError(
          "agent_config must be a dictionary or a JSON string representing a"
          f" dictionary, got {type(parsed_config).__name__}"
      )
  except (ValueError, TypeError) as ex:
    return None, {
        "status": "ERROR",
        "error_details": f"Invalid agent_config: {ex}",
    }
  return parsed_config, None


def _mask_field_present(config: dict[str, Any], field: str) -> bool:
  """Checks if a dot-separated field path is present in config."""
  node: Any = config
  for part in field.split("."):
    if not isinstance(node, dict) or part not in node:
      return False
    node = node[part]
  return True


async def _await_lro(
    *,
    session: requests.Session,
    base_url: str,
    headers: dict[str, str],
    resp: requests.Response,
    deadline: float,
    poll_interval: float,
    total_timeout: float,
) -> dict[str, Any]:
  """Interprets a mutation response and polls the LRO until it is done."""
  if not resp.ok:
    return {
        "status": "ERROR",
        "error_details": (
            f"API returned error status: {resp.status_code} {resp.text}"
        ),
    }

  operation = resp.json()
  if operation.get("done"):
    if "error" in operation:
      return {
          "status": "ERROR",
          "error_details": json.dumps(operation["error"]),
      }
    return {
        "status": "SUCCESS",
        "response": operation.get("response", operation),
    }

  operation_name = operation.get("name")
  if not operation_name or "/operations/" not in operation_name:
    if not operation.get("done", True):
      return {
          "status": "ERROR",
          "error_details": (
              "Operation is not completed and does not contain a pollable"
              f" '/operations/' name: {operation}"
          ),
      }
    return {"status": "SUCCESS", "response": operation}

  poll_url = f"{base_url}/{operation_name}"

  while True:
    remaining_budget = deadline - time.monotonic()
    if remaining_budget <= 0.1:
      break

    request_timeout = min(_GDA_REQUEST_TIMEOUT_SECONDS, remaining_budget)
    try:
      poll_resp = await asyncio.to_thread(
          session.get,
          poll_url,
          headers=headers,
          timeout=request_timeout,
      )
    except (requests.ConnectionError, requests.Timeout) as ex:
      remaining_budget = deadline - time.monotonic()
      if remaining_budget <= poll_interval:
        return {
            "status": "ERROR",
            "error_details": f"Polling failed with exception: {ex}",
            "operation_name": operation_name,
        }
      await asyncio.sleep(min(poll_interval, remaining_budget))
      continue
    except Exception as ex:  # pylint: disable=broad-except
      return {
          "status": "ERROR",
          "error_details": f"Polling failed with exception: {ex}",
          "operation_name": operation_name,
      }
    if not poll_resp.ok:
      if poll_resp.status_code in _RETRYABLE_STATUS_CODES:
        remaining_budget = deadline - time.monotonic()
        if remaining_budget > poll_interval:
          await asyncio.sleep(min(poll_interval, remaining_budget))
          continue
      return {
          "status": "ERROR",
          "error_details": (
              f"Polling failed with status: {poll_resp.status_code} "
              f"{poll_resp.text}"
          ),
          "operation_name": operation_name,
      }
    try:
      poll_op = poll_resp.json()
    except ValueError as ex:
      return {
          "status": "ERROR",
          "error_details": f"Polling returned invalid JSON: {ex}",
          "operation_name": operation_name,
      }
    if poll_op.get("done"):
      if "error" in poll_op:
        return {
            "status": "ERROR",
            "error_details": json.dumps(poll_op["error"]),
            "operation_name": operation_name,
        }
      return {
          "status": "SUCCESS",
          "response": poll_op.get("response", poll_op),
      }

    remaining_budget = deadline - time.monotonic()
    if remaining_budget <= 0.1:
      break
    await asyncio.sleep(min(poll_interval, remaining_budget))

  return {
      "status": "ERROR",
      "error_details": (
          f"Operation {operation_name} did not complete within"
          f" {total_timeout} seconds. The operation may still be executing"
          " asynchronously in the background. Do not retry the operation."
      ),
      "operation_name": operation_name,
  }


async def _mutate_data_agent(
    build_url: Callable[[str], str],
    http_method: str,
    *,
    credentials: Credentials,
    settings: DataAgentToolConfig,
    location: str | None = None,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Issues a data agent mutation request and waits for the LRO to finish."""
  kwargs = {}
  loc = location or (
      settings.location
      if settings and isinstance(settings.location, str)
      else None
  )
  if loc:
    kwargs["location"] = loc
  api_endpoint = (
      settings.api_endpoint
      if settings and isinstance(settings.api_endpoint, str)
      else None
  )
  if api_endpoint:
    kwargs["api_endpoint"] = api_endpoint
  session, endpoint = _gda_stream_util.get_gda_session(credentials, **kwargs)
  base_url = f"{endpoint}/v1"
  url = build_url(base_url)

  total_timeout = settings.data_agent_modification_timeout_seconds
  poll_interval = settings.data_agent_modification_poll_interval_seconds
  deadline = time.monotonic() + total_timeout

  request_kwargs: dict[str, Any] = {}
  if params is not None:
    request_kwargs["params"] = params
  if json_body is not None:
    request_kwargs["json"] = json_body

  with session:
    resp = await asyncio.to_thread(
        getattr(session, http_method),
        url,
        **request_kwargs,
        headers=_gda_headers(),
        timeout=min(
            _GDA_REQUEST_TIMEOUT_SECONDS,
            max(0.0, deadline - time.monotonic()),
        ),
    )
    return await _await_lro(
        session=session,
        base_url=base_url,
        headers=_gda_headers(),
        resp=resp,
        deadline=deadline,
        poll_interval=poll_interval,
        total_timeout=total_timeout,
    )


def _extract_location_from_resource_name(resource_name: str) -> str | None:
  """Extracts the location segment from a resource name if present."""
  parts = resource_name.split("/")
  for i, part in enumerate(parts[:-1]):
    if part == "locations" and i + 1 < len(parts):
      return parts[i + 1]
  return None


def _parse_pagination_params(
    page_size: Any,
    page_token: Any,
) -> tuple[int | None, str | None, dict[str, Any] | None]:
  """Validates and normalizes optional page_size and page_token parameters.

  The returned page token is None when the caller did not supply one at all,
  which is what selects the automatic drain. An explicitly supplied empty
  string is preserved as "" and means "the first page", so it must stay
  distinguishable from None.
  """
  parsed_page_size: int | None = None
  if page_size is not None:
    if isinstance(page_size, bool) or not isinstance(page_size, int):
      return (
          None,
          None,
          {
              "status": "ERROR",
              "error_details": (
                  "Invalid type for 'page_size': expected integer, got"
                  f" {type(page_size).__name__}"
              ),
          },
      )
    if page_size < _MIN_PAGE_SIZE:
      return (
          None,
          None,
          {
              "status": "ERROR",
              "error_details": f"'page_size' must be positive, got {page_size}",
          },
      )
    parsed_page_size = page_size

  parsed_page_token: str | None = None
  if page_token is not None:
    if not isinstance(page_token, str):
      return (
          None,
          None,
          {
              "status": "ERROR",
              "error_details": (
                  "Invalid type for 'page_token': expected string, got"
                  f" {type(page_token).__name__}"
              ),
          },
      )
    parsed_page_token = page_token

  return parsed_page_size, parsed_page_token, None


def _parse_data_agents_page(
    body: Any,
) -> tuple[list[Any], str, dict[str, Any]]:
  """Splits a decoded JSON page into data agents, nextPageToken, and extra fields."""
  if not isinstance(body, dict):
    raise ValueError("failed to decode response: response is not a JSON object")

  data_agents: list[Any] = []
  if "dataAgents" in body and body["dataAgents"] is not None:
    raw_agents = body["dataAgents"]
    if not isinstance(raw_agents, list):
      raise ValueError("failed to decode response: invalid 'dataAgents' field")
    data_agents = list(raw_agents)

  next_page_token = ""
  if "nextPageToken" in body and body["nextPageToken"] is not None:
    raw_token = body["nextPageToken"]
    if not isinstance(raw_token, str):
      raise ValueError(
          "failed to decode response: invalid 'nextPageToken' field"
      )
    next_page_token = raw_token

  extras = {
      k: v for k, v in body.items() if k not in ("dataAgents", "nextPageToken")
  }
  return data_agents, next_page_token, extras


def _concat_extra_values(prev: Any, next_val: Any) -> Any:
  """Joins two JSON values for an extra top-level field across pages.

  When both values are lists (such as `unreachable`), concatenates them so
  entries from every page are preserved. When the values cannot be merged, keeps
  whichever side holds real data: a non-list `prev` is overwritten by `next_val`
  (last page wins for scalars), while an unmergeable `next_val` leaves an
  already aggregated `prev` list untouched so earlier pages are never dropped.
  """
  if not isinstance(prev, list):
    if next_val is None:
      return prev
    return next_val
  if not isinstance(next_val, list):
    return prev
  return prev + next_val


def _merge_extras(
    extras: dict[str, Any], page_extras: Mapping[str, Any]
) -> None:
  """Folds a page's extra top-level fields into the ones collected so far."""
  for field, value in page_extras.items():
    if field not in extras:
      extras[field] = value
    else:
      extras[field] = _concat_extra_values(extras[field], value)


def _build_list_result(
    extras: Mapping[str, Any],
    data_agents: list[Any],
    next_page_token: str,
) -> dict[str, Any]:
  """Assembles a list_accessible_data_agents response dictionary."""
  result: dict[str, Any] = {}
  for field, value in extras.items():
    if field not in ("status", "response", "error_details"):
      result[field] = value
  result["status"] = "SUCCESS"
  result["response"] = data_agents
  if next_page_token:
    result["nextPageToken"] = next_page_token
  return result


def _fetch_data_agents_page(
    session: requests.Session,
    list_url: str,
    *,
    page_size: int | None = None,
    page_token: str = "",
    timeout: float = _GDA_REQUEST_TIMEOUT_SECONDS,
) -> Any:
  """Sends a GET request for a single page of accessible data agents."""
  query_params: dict[str, Any] = {}
  if page_size is not None:
    query_params["pageSize"] = page_size
  if page_token:
    query_params["pageToken"] = page_token

  request_kwargs: dict[str, Any] = {}
  if query_params:
    request_kwargs["params"] = query_params

  resp = session.get(
      list_url,
      **request_kwargs,
      headers=_gda_headers(),
      timeout=timeout,
  )
  # raise_for_status() omits the response body, which is where the API puts the
  # actual error message, so raise a richer error here instead. Mirrors the
  # polling path above.
  if not resp.ok:
    raise requests.HTTPError(
        f"API returned non-200 status: {resp.status_code} {resp.text}",
        response=resp,
    )
  try:
    return resp.json()
  except ValueError as ex:
    raise ValueError(f"failed to decode response: {ex}") from ex


def _is_retryable_page_error(ex: Exception) -> bool:
  """Returns whether a failed page request is worth retrying.

  Connection-level failures never got a reply and the GET is idempotent, so they
  are always safe to retry. HTTP failures are retried only for the transient
  status codes the polling path already treats as retryable.
  """
  if isinstance(ex, requests.ConnectionError):
    return True
  status = getattr(getattr(ex, "response", None), "status_code", None)
  return status in _RETRYABLE_STATUS_CODES


def _pagination_timeout_msg(last_error: Exception | None) -> str:
  """Returns the timeout guidance, keeping the underlying cause if any."""
  if last_error is None:
    return _PAGINATION_TIMEOUT_MSG
  return f"{_PAGINATION_TIMEOUT_MSG} (last error: {last_error})"


def _page_failure(last_error: Exception) -> Exception:
  """Returns the error to raise once a page has run out of attempts.

  A timeout that survives every attempt still means the caller should switch to
  manual pagination, so it keeps the guidance rather than surfacing the bare
  transport error. Other failures are surfaced unchanged.
  """
  if isinstance(last_error, requests.Timeout):
    return TimeoutError(_pagination_timeout_msg(last_error))
  return last_error


def _fetch_page_with_retries(
    session: requests.Session,
    list_url: str,
    *,
    page_token: str,
    deadline: float,
) -> Any:
  """Fetches one page, retrying transient failures within the time budget.

  The session from `get_gda_session` has no retry adapter, and one automatic
  drain can issue many sequential requests, so a single transient 503 would
  otherwise fail the whole call. Retrying is done by hand rather than with a
  urllib3 Retry adapter because that adapter is shared by the whole session (it
  would silently change the polling path's semantics too), because its backoff
  sleeps are invisible to the deadline budget enforced here, and because its
  exhaustion surfaces as requests.RetryError rather than requests.HTTPError.
  """
  last_error: Exception | None = None
  attempt = 0
  while True:
    attempt += 1
    remaining_budget = deadline - time.monotonic()
    if remaining_budget <= 0:
      raise TimeoutError(_pagination_timeout_msg(last_error))

    try:
      return _fetch_data_agents_page(
          session,
          list_url,
          page_token=page_token,
          timeout=min(_GDA_REQUEST_TIMEOUT_SECONDS, remaining_budget),
      )
    except requests.ConnectTimeout as ex:
      # Most specific first: ConnectTimeout inherits from both ConnectionError
      # and Timeout. The connection was never established, so it is safe to
      # retry.
      error: Exception = ex
    except requests.Timeout as ex:
      # A read timeout means the server did receive the request; retrying only
      # re-runs the same slow call, so surface the guidance immediately.
      raise TimeoutError(_pagination_timeout_msg(ex)) from ex
    except (requests.HTTPError, requests.ConnectionError) as ex:
      if not _is_retryable_page_error(ex):
        raise
      error = ex

    # Kept for the budget-exhausted message at the top of the next iteration.
    last_error = error
    if attempt >= _MAX_PAGE_ATTEMPTS:
      raise _page_failure(error) from error

    # Only spend budget on a retry that still has room to finish; otherwise the
    # retry would merely consume the remainder and fail anyway. Note there is no
    # Retry-After handling: a fixed linear backoff is enough for one caller.
    # list_accessible_data_agents is synchronous (matching get_data_agent_info
    # and ask_data_agent in this module for backward compatibility); per-request
    # timeouts are clamped via min(_GDA_REQUEST_TIMEOUT_SECONDS,
    # remaining_budget) above, and this bounded sleep (0.5s / 1.0s) is accounted
    # for against deadline before sleeping.
    backoff = _PAGE_RETRY_BACKOFF_SECONDS * attempt
    if deadline - time.monotonic() <= backoff + _GDA_REQUEST_TIMEOUT_SECONDS:
      raise _page_failure(error) from error
    time.sleep(backoff)


def _list_all_accessible_data_agents(
    session: requests.Session,
    list_url: str,
) -> dict[str, Any]:
  """Fetches pages of accessible data agents until completion or safety limits."""
  deadline = time.monotonic() + _MAX_AUTO_DURATION_SECONDS
  data_agents: list[Any] = []
  page_token = ""
  extras: dict[str, Any] = {}
  seen_tokens: set[str] = set()
  pages_fetched = 0

  for _ in range(_MAX_AUTO_PAGES):
    try:
      body = _fetch_page_with_retries(
          session, list_url, page_token=page_token, deadline=deadline
      )
    except TimeoutError as ex:
      if pages_fetched == 0:
        # Nothing was fetched, so there is no partial list to hand back.
        # Returning SUCCESS with an empty response would tell the caller they
        # have no accessible data agents, which is a worse lie than an error.
        raise
      # `page_token` still holds the token for the page that just failed, so
      # the caller can resume from exactly where this left off.
      extras["paginationWarning"] = (
          "timed out before every page was fetched, so this list is incomplete;"
          f" pass the returned nextPageToken to resume ({ex})"
      )
      break
    pages_fetched += 1

    page_agents, next_page_token, page_extras = _parse_data_agents_page(body)
    data_agents.extend(page_agents)
    _merge_extras(extras, page_extras)

    page_token = next_page_token

    if page_token and page_token in seen_tokens:
      # The API handed back a token it has already issued, so it is cycling
      # rather than advancing. Drop the token -- resuming from it would just
      # replay pages forever -- but flag the result so the truncation is not
      # read as the end of the list.
      extras["paginationWarning"] = (
          "the API returned a nextPageToken it had already returned, so this"
          " list is incomplete and cannot be resumed from it; retry the call or"
          " use a narrower request"
      )
      page_token = ""
      break
    if page_token:
      seen_tokens.add(page_token)

    if not page_token or len(data_agents) >= _MAX_AUTO_DATA_AGENTS:
      break

  return _build_list_result(extras, data_agents, page_token)


def list_accessible_data_agents(
    project_id: str,
    credentials: Credentials,
    settings: DataAgentToolConfig | None = None,
    *,
    location: str | None = None,
    page_size: int | None = None,
    page_token: str | None = None,
) -> dict[str, Any]:
  """Lists accessible data agents in a project.

  Every page is fetched automatically by default, so a single call returns all
  accessible data agents unless a safety limit stops the drain early. Whenever
  the response contains a `nextPageToken` there are more: pass that value back
  as `page_token` to step through the remaining pages one page at a time, or
  set `page_size` or `page_token` from the start to take over pagination
  manually.

  Args:
      project_id: The project to list agents in.
      credentials: The credentials to use for the request.
      settings: Optional tool settings containing location or custom endpoint.
      location: Optional Google Cloud location to list agents from (e.g. "eu" or
        "us"). If omitted, uses the toolset's configured location, falling back
        to "global".
      page_size: Optional. The maximum number of data agents to return in this
        call. Must be a positive integer. Only set this to page through the
        results manually; when both `page_size` and `page_token` are omitted,
        every page is fetched automatically and all accessible data agents are
        returned.
      page_token: Optional. A page token returned as `nextPageToken` by a
        previous call, used to fetch the next page. Only set this to page
        through the results manually; omit it to fetch all accessible data
        agents at once. Passing an empty string explicitly requests the first
        page manually, which is not the same as omitting it.

  Returns:
      A dictionary containing the status and a list of data agents with their
      detailed information, including name, display_name, description (if
      available), create_time, update_time, and data_analytics_agent context,
      plus `nextPageToken` when more data agents remain, or error details if
      the request fails.

      An automatic drain also stops early once it reaches a safety limit, so
      `nextPageToken` is the authoritative signal: whenever it is present more
      data agents remain and passing it back continues the list, whether the
      drain stopped on a limit or for any other reason.

      A `paginationWarning` field is added only when the drain stopped for a
      reason worth reading rather than by simply reaching a limit: it ran out
      of time, in which case `nextPageToken` is present and resuming works, or
      the API cycled its page tokens, in which case `nextPageToken` is omitted
      because resuming from it would replay pages forever.

  Examples:
      >>> list_accessible_data_agents(
      ...     project_id="my-gcp-project",
      ...     credentials=credentials,
      ... )
      {
        "status": "SUCCESS",
        "response": [
          {
            "name": "projects/my-project/locations/global/dataAgents/agent1",
            "displayName": "My Test Agent",
            "createTime": "2025-10-01T22:44:22.473927629Z",
            "updateTime": "2025-10-01T22:44:23.094541325Z",
            "dataAnalyticsAgent": {
              "publishedContext": {
                "datasourceReferences": {
                  "bq": {
                    "tableReferences": [{
                      "projectId": "my-project",
                      "datasetId": "dataset1",
                      "tableId": "table1"
                    }]
                  }
                }
              }
            }
          },
          {
            "name": "projects/my-project/locations/global/dataAgents/agent2",
            "displayName": "",
            "description": "Description for Agent 2.",
            "createTime": "2025-06-23T20:23:48.650597312Z",
            "updateTime": "2025-06-23T20:23:49.437095391Z",
            "dataAnalyticsAgent": {
              "publishedContext": {
                "datasourceReferences": {
                  "bq": {
                    "tableReferences": [{
                      "projectId": "another-project",
                      "datasetId": "dataset2",
                      "tableId": "table2"
                    }]
                  }
                },
                "systemInstruction": "You are a helpful assistant.",
                "options": {"analysis": {"python": {"enabled": True}}}
              }
            }
          }
        ]
      }
  """
  try:
    config_location = (
        settings.location
        if settings and isinstance(settings.location, str)
        else None
    )
    effective_location = location or config_location or "global"
    for val, name in (
        (project_id, "project_id"),
        (effective_location, "location"),
    ):
      invalid_segment_error = _validate_path_segment(val, name)
      if invalid_segment_error:
        return invalid_segment_error

    parsed_page_size, parsed_page_token, pagination_error = (
        _parse_pagination_params(page_size, page_token)
    )
    if pagination_error:
      return pagination_error

    api_endpoint = (
        settings.api_endpoint
        if settings and isinstance(settings.api_endpoint, str)
        else None
    )

    kwargs: dict[str, str] = {}
    if effective_location:
      kwargs["location"] = effective_location
    if api_endpoint:
      kwargs["api_endpoint"] = api_endpoint

    session, endpoint = _gda_stream_util.get_gda_session(credentials, **kwargs)
    base_url = f"{endpoint}/v1"

    list_url = f"{base_url}/projects/{project_id}/locations/{effective_location}/dataAgents:listAccessible"
    with session:
      if parsed_page_size is not None or parsed_page_token is not None:
        # Caller-driven pagination is a single request the caller can simply
        # reissue, so it deliberately skips the retry layer that the automatic
        # drain below needs to survive a transient failure mid-sequence.
        # Note the `is not None` test: an explicit page_token="" asks for the
        # first page manually, so only the absence of BOTH arguments selects
        # the automatic drain.
        body = _fetch_data_agents_page(
            session,
            list_url,
            page_size=parsed_page_size,
            page_token=parsed_page_token or "",
            timeout=_GDA_REQUEST_TIMEOUT_SECONDS,
        )
        page_agents, next_page_token, page_extras = _parse_data_agents_page(
            body
        )
        return _build_list_result(page_extras, page_agents, next_page_token)
      return _list_all_accessible_data_agents(session, list_url)
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def _get_data_agent_info(
    data_agent_name: str,
    credentials: Credentials,
    session: requests.Session | None = None,
    settings: DataAgentToolConfig | None = None,
) -> dict[str, Any]:
  try:
    real_session: requests.Session | None = session
    real_settings: DataAgentToolConfig | None = settings

    extracted_location = _extract_location_from_resource_name(data_agent_name)
    location = extracted_location or (
        real_settings.location
        if real_settings and isinstance(real_settings.location, str)
        else None
    )
    api_endpoint = (
        real_settings.api_endpoint
        if real_settings and isinstance(real_settings.api_endpoint, str)
        else None
    )

    kwargs: dict[str, str] = {}
    if location:
      kwargs["location"] = location
    if api_endpoint:
      kwargs["api_endpoint"] = api_endpoint

    endpoint = _gda_stream_util.get_gda_endpoint(**kwargs)
    base_url = f"{endpoint}/v1"
    get_url = f"{base_url}/{data_agent_name}"
    if real_session:
      resp = real_session.get(
          get_url,
          headers=_gda_headers(),
          timeout=_GDA_REQUEST_TIMEOUT_SECONDS,
      )
    else:
      local_session, _ = _gda_stream_util.get_gda_session(credentials, **kwargs)
      with local_session:
        resp = local_session.get(
            get_url,
            headers=_gda_headers(),
            timeout=_GDA_REQUEST_TIMEOUT_SECONDS,
        )

    resp.raise_for_status()
    return {
        "status": "SUCCESS",
        "response": resp.json(),
    }
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def get_data_agent_info(
    data_agent_name: str,
    credentials: Credentials,
    settings: DataAgentToolConfig | None = None,
) -> dict[str, Any]:
  """Gets a data agent by name.

  Args:
      data_agent_name: The name of the agent to get, in format
        projects/{project}/locations/{location}/dataAgents/{agent}.
      credentials: The credentials to use for the request.
      settings: Optional tool settings containing location or custom endpoint.

  Returns:
      A dictionary containing the status and details of a data agent,
      including name, display_name, description (if available),
      create_time, update_time, and data_analytics_agent context,
      or error details if the request fails.

  Examples:
      >>> get_data_agent_info(
      ...     data_agent_name="projects/p/locations/g/dataAgents/agent-1",
      ...     credentials=credentials,
      ... )
      {
          "status": "SUCCESS",
          "response": {
              "name": "projects/p/locations/g/dataAgents/agent-1",
              "description": "Description for Agent 1.",
              "createTime": "2025-06-23T20:23:48.650597312Z",
              "updateTime": "2025-06-23T20:23:49.437095391Z",
              "dataAnalyticsAgent": {
                  "publishedContext": {
                      "systemInstruction": "You are a helpful assistant.",
                      "options": {"analysis": {"python": {"enabled": True}}},
                      "datasourceReferences": {
                          "bq": {
                              "tableReferences": [{
                                  "projectId": "my-gcp-project",
                                  "datasetId": "dataset1",
                                  "tableId": "table1"
                              }]
                          }
                      },
                  }
              }
          }
      }
  """
  return _get_data_agent_info(data_agent_name, credentials, settings=settings)


def ask_data_agent(
    data_agent_name: str,
    query: str,
    *,
    credentials: Credentials,
    settings: DataAgentToolConfig,
    tool_context: ToolContext,
) -> dict[str, Any]:
  r"""Asks a question to a data agent.

  Args:
      data_agent_name: The resource name of an existing data agent to ask, in
        format projects/{project}/locations/{location}/dataAgents/{agent}.
      query: The question to ask the agent.
      credentials: The credentials to use for the request.
      settings: Tool configuration including max rows and optional endpoint.
      tool_context: The context for the tool.

  Returns:
      A dictionary with two keys:
      - 'status': A string indicating the final status (e.g., "SUCCESS").
      - 'response': A list of dictionaries, where each dictionary
        represents a step in the agent's execution process and can
        contain keys like 'text', 'data', or 'Data Retrieved' indicating
        thought process, SQL generation, data retrieval, or final answer.

  Examples:
      A query to a data agent, showing the full return structure.
      The original question: "What is the average tree height in San
      Francisco?"

      >>> ask_data_agent(
      ...     data_agent_name="projects/p/locations/g/dataAgents/agent-1",
      ...     query="What is the average tree height in San Francisco?",
      ...     credentials=credentials,
      ...     settings=settings,
      ...     tool_context=tool_context,
      ... )
      {
        "status": "SUCCESS",
        "response": [
          {
            "text": {
              "parts": [
                "Analyzing context",
                "Retrieved context for 1 table."
              ],
              "textType": "THOUGHT"
            }
          },
          {
            "data": {
              "generatedSql": "SELECT\n AVG(SAFE_CAST(street_trees.dbh AS\n
              FLOAT64)) AS average_height\nFROM\n
              bigquery-public-data.san_francisco.street_trees AS street_trees;"
            }
          },
          {
            "Data Retrieved": {
              "headers": [
                "average_height"
              ],
              "rows": [
                [
                  10.073475670972512
                ]
              ],
              "summary": "Showing all 1 rows."
            }
          },
          {
            "text": {
              "parts": [
                "### Summary\nBased on the street tree data for San Francisco,\n
                the average height (recorded in the dbh column) is
                approximately\n                10.07."
              ],
              "textType": "FINAL_RESPONSE"
            }
          }
        ]
      }
  """
  try:
    location = (
        settings.location
        if settings and isinstance(settings.location, str)
        else None
    )
    api_endpoint = (
        settings.api_endpoint
        if settings and isinstance(settings.api_endpoint, str)
        else None
    )

    if not location and not api_endpoint and data_agent_name:
      location = _extract_location_from_resource_name(data_agent_name)

    kwargs: dict[str, str] = {}
    if location:
      kwargs["location"] = location
    if api_endpoint:
      kwargs["api_endpoint"] = api_endpoint

    session, endpoint = _gda_stream_util.get_gda_session(credentials, **kwargs)
    with session:
      base_url = f"{endpoint}/v1"

      agent_info = _get_data_agent_info(
          data_agent_name, credentials, session=session, settings=settings
      )

      if agent_info.get("status") == "ERROR":
        return agent_info
      parent = data_agent_name.rsplit("/", 2)[0]
      chat_url = f"{base_url}/{parent}:chat"
      chat_payload = {
          "messages": [{"userMessage": {"text": query}}],
          "dataAgentContext": {
              "dataAgent": data_agent_name,
          },
          "clientIdEnum": _GDA_CLIENT_ID,
      }
      resp = _gda_stream_util.get_stream(
          session,
          chat_url,
          chat_payload,
          _gda_headers(),
          settings.max_query_result_rows,
      )

    return {"status": "SUCCESS", "response": resp}
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


async def create_data_agent(
    project_id: str,
    data_agent_id: str,
    agent_config: str,
    location: str | None = None,
    *,
    credentials: Credentials,
    settings: DataAgentToolConfig,
) -> dict[str, Any]:
  r"""Creates a new data agent.

  Args:
      project_id: The project in which to create the agent.
      data_agent_id: The ID to use for the new data agent.
      agent_config: A JSON string representing the DataAgent resource to create.
        For detailed REST resource schema and create documentation, see:
        https://docs.cloud.google.com/gemini/data-agents/reference/rest/v1/projects.locations.dataAgents#DataAgent
        https://docs.cloud.google.com/gemini/data-agents/reference/rest/v1/projects.locations.dataAgents/create
      location: The Google Cloud location for data agent creation. If omitted,
        uses the toolset's configured location, falling back to "global". Only
        specify this when the user explicitly asks for a different region.
      credentials: The credentials to use for the request.
      settings: The configuration for the tool.

  Returns:
      A dictionary containing the status and the newly created data agent's
      details, or error details if the request fails.
      The tool waits for the create operation to finish, polling for up to
      `DataAgentToolConfig.data_agent_modification_timeout_seconds` (60s by
      default) in total. A timeout does not necessarily mean the creation
      failed; the operation may still be processing in the background, and
      `operation_name` is returned so the caller can check status later.

  Examples:
      >>> await create_data_agent(
      ...     project_id="my-gcp-project",
      ...     data_agent_id="my-new-agent",
      ...     agent_config='{"displayName": "My New Agent", "description":'
      ...     ' "An agent that helps with my-new-agent tasks",'
      ...     ' "dataAnalyticsAgent": {"publishedContext":'
      ...     ' {"datasourceReferences": {"bq": {"tableReferences":'
      ...     ' [{"projectId": "my-gcp-project", "datasetId": "dataset1",'
      ...     ' "tableId": "table1"}]}}, "systemInstruction": "You are a'
      ...     ' helpful assistant.", "options": {"analysis": {"python":'
      ...     ' {"enabled": True}}}}}}',
      ...     location="global",
      ...     credentials=credentials,
      ...     settings=DataAgentToolConfig(enable_data_agent_modification=True),
      ... )
      {
        "status": "SUCCESS",
        "response": {
          "@type":
          "type.googleapis.com/google.cloud.geminidataanalytics.v1.DataAgent",
          "name":
          "projects/my-gcp-project/locations/global/dataAgents/my-new-agent",
          "displayName": "My New Agent",
          "description": "An agent that helps with my-new-agent tasks",
          "createTime": "2025-10-01T22:44:22.473927629Z",
          "updateTime": "2025-10-01T22:44:22.473927629Z",
          "dataAnalyticsAgent": {
            "publishedContext": {
              "datasourceReferences": {
                "bq": {
                  "tableReferences": [{
                    "projectId": "my-gcp-project",
                    "datasetId": "dataset1",
                    "tableId": "table1"
                  }]
                }
              },
              "systemInstruction": "You are a helpful assistant.",
              "options": {"analysis": {"python": {"enabled": True}}}
            }
          }
        }
      }

      Example showing an error response if the Gemini Data Analytics API is
      disabled:
      >>> await create_data_agent(
      ...     project_id="my-gcp-project",
      ...     data_agent_id="my-new-agent",
      ...     agent_config={"displayName": "My New Agent"},
      ...     credentials=credentials,
      ...     settings=DataAgentToolConfig(enable_data_agent_modification=True),
      ... )
      {
        "status": "ERROR",
        "error_details": "API returned error status: 403 {\n  \"error\": {\n
        \"code\": 403,\n    \"message\": \"Data Analytics API with Gemini has
        not been used in project my-gcp-project before or it is disabled.\",\n
        \"status\": \"PERMISSION_DENIED\"\n  }\n}"
      }
  """
  try:
    disabled_error = _modification_disabled_error(settings)
    if disabled_error:
      return disabled_error

    config_location = (
        settings.location
        if settings and isinstance(settings.location, str)
        else None
    )
    effective_location = location or config_location or "global"
    for val, name in (
        (project_id, "project_id"),
        (effective_location, "location"),
        (data_agent_id, "data_agent_id"),
    ):
      invalid_segment_error = _validate_path_segment(val, name)
      if invalid_segment_error:
        return invalid_segment_error

    parsed_config, config_error = _parse_agent_config(agent_config)
    if config_error:
      return config_error

    return await _mutate_data_agent(
        lambda base_url: (
            f"{base_url}/projects/{project_id}/locations/{effective_location}/dataAgents"
        ),
        "post",
        location=effective_location,
        credentials=credentials,
        settings=settings,
        params={"dataAgentId": data_agent_id},
        json_body=parsed_config,
    )
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


async def update_data_agent(
    data_agent_name: str,
    agent_config: str,
    update_mask: str,
    *,
    credentials: Credentials,
    settings: DataAgentToolConfig,
) -> dict[str, Any]:
  r"""Updates an existing data agent.

  Args:
      data_agent_name: The name of the data agent to update, in format
        projects/{project}/locations/{location}/dataAgents/{agent}.
      agent_config: A JSON string representing the DataAgent resource. For
        detailed REST resource schema and patch documentation, see:
        https://docs.cloud.google.com/gemini/data-agents/reference/rest/v1/projects.locations.dataAgents#DataAgent
        https://docs.cloud.google.com/gemini/data-agents/reference/rest/v1/projects.locations.dataAgents/patch
      update_mask: Comma-separated list of fields to update, using the API's
        camelCase JSON field names (e.g. "displayName,description" or
        "dataAnalyticsAgent.publishedContext.systemInstruction"). Every field
        listed here MUST also be present in agent_config; fields listed in
        update_mask but absent from agent_config are rejected to prevent
        accidental data loss.
      credentials: The credentials to use for the request.
      settings: The configuration for the tool.

  Returns:
      A dictionary containing the status and the updated data agent's details,
      or error details if the request fails.
      The tool waits for the update operation to finish, polling for up to
      `DataAgentToolConfig.data_agent_modification_timeout_seconds` (60s by
      default) in total.
      Note that a polling timeout does not necessarily mean the update
      failed; the operation may still be completing in the background.

  Examples:
      >>> update_data_agent(
      ...     "projects/my-project/locations/global/dataAgents/my-agent",
      ...     agent_config='{"displayName": "Updated Agent"}',
      ...     update_mask="displayName",
      ...     credentials=creds,
      ...     settings=settings,
      ... )
      {'status': 'SUCCESS', 'response': {'name':
      'projects/my-project/locations/global/dataAgents/my-agent', 'displayName':
      'Updated Agent'}}
  """
  try:
    disabled_error = _modification_disabled_error(settings)
    if disabled_error:
      return disabled_error

    invalid_name_error = _validate_data_agent_name(data_agent_name)
    if invalid_name_error:
      return invalid_name_error

    fields = [f.strip() for f in update_mask.split(",") if f.strip()]
    if not fields:
      return {
          "status": "ERROR",
          "error_details": (
              "update_mask must be a non-empty comma-separated list of fields,"
              ' e.g. "displayName,description".'
          ),
      }
    update_mask = ",".join(fields)

    parsed_config, config_error = _parse_agent_config(agent_config)
    if config_error or parsed_config is None:
      return config_error or {
          "status": "ERROR",
          "error_details": "Failed to parse agent_config.",
      }

    # Check that every field in the update_mask exists in the provided agent_config.
    # Example Pass: update_mask="displayName", agent_config={"displayName": "New"}
    # Example Fail: update_mask="displayName,description", agent_config={"displayName": "New"}
    # This prevents the API from wiping out fields that the user forgot to include in agent_config.
    missing_fields = [
        f for f in fields if not _mask_field_present(parsed_config, f)
    ]

    if missing_fields:
      return {
          "status": "ERROR",
          "error_details": (
              f"update_mask fields {missing_fields} are not present in"
              " agent_config. Fields listed in update_mask but absent from"
              " agent_config will be cleared; include them explicitly or remove"
              " them from the mask."
          ),
      }

    return await _mutate_data_agent(
        lambda base_url: f"{base_url}/{data_agent_name}",
        "patch",
        credentials=credentials,
        settings=settings,
        location=_extract_location_from_resource_name(data_agent_name),
        params={"updateMask": update_mask},
        json_body=parsed_config,
    )
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


async def delete_data_agent(
    data_agent_name: str,
    *,
    credentials: Credentials,
    settings: DataAgentToolConfig,
) -> dict[str, Any]:
  r"""Deletes an existing data agent.

  Args:
      data_agent_name: The name of the data agent to delete, in format
        projects/{project}/locations/{location}/dataAgents/{agent}.
      credentials: The credentials to use for the request.
      settings: The configuration for the tool.

  Returns:
      A dictionary containing the status and response, or error details if the
      request fails.
      The tool waits for the delete operation to finish, polling for up to
      `DataAgentToolConfig.data_agent_modification_timeout_seconds` (60s by
      default) in total.
      Note that a polling timeout does not necessarily mean the deletion
      failed; the operation may still be completing in the background.

  Examples:
      >>> await delete_data_agent(
      ...     "projects/my-project/locations/global/dataAgents/my-agent",
      ...     credentials=creds,
      ...     settings=settings,
      ... )
      {'status': 'SUCCESS', 'response': {'name':
      'projects/my-project/locations/global/dataAgents/my-agent', 'deleteTime':
      '2026-07-31T21:26:41.877Z', 'purgeTime': '2026-08-30T21:26:41.877Z'}}
  """
  try:
    disabled_error = _modification_disabled_error(settings)
    if disabled_error:
      return disabled_error

    invalid_name_error = _validate_data_agent_name(data_agent_name)
    if invalid_name_error:
      return invalid_name_error

    return await _mutate_data_agent(
        lambda base_url: f"{base_url}/{data_agent_name}",
        "delete",
        credentials=credentials,
        settings=settings,
        location=_extract_location_from_resource_name(data_agent_name),
    )
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }
