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

from google.genai import types

from ._client_labels_utils import get_client_labels

# TODO: Callers that can be pointed at a custom endpoint still send the runtime
# tag there: google_llm (custom base_url or client), lite_llm (custom api_base
# or litellm_proxy), the GCP skill registry (AGENT_REGISTRY_ENDPOINT), and the
# Vertex session and memory services when _api_client_http_options_override
# sets a base_url.


def get_tracking_headers(
    framework_label: str | None = None, *, exclude_runtime_tag: bool = False
) -> dict[str, str]:
  """Returns a dictionary of HTTP headers for tracking API requests.

  These headers are used to identify HTTP calls made by ADK towards
   Vertex AI LLM APIs.

  Args:
    framework_label: Optional SemVer build-metadata suffix appended to the
      google-adk framework token (e.g. "managed_agent"), used to distinguish a
      specific ADK surface in Google's server-side usage pipeline.
    exclude_runtime_tag: Whether to leave out the hosting runtime suffix. Pass
      True for requests that do not go to a Google API endpoint.
  """
  labels = get_client_labels(
      framework_label=framework_label, exclude_runtime_tag=exclude_runtime_tag
  )
  header_value = " ".join(labels)
  return {
      "x-goog-api-client": header_value,
      "user-agent": header_value,
  }


def get_tracking_http_options(
    *, exclude_runtime_tag: bool = False
) -> types.HttpOptions:
  """Returns HttpOptions carrying ADK tracking headers for a genai Client.

  Use this when constructing a google.genai Client so its outbound calls are
  attributable to ADK by Google's server-side usage pipeline, matching
  models/google_llm.py.

  Args:
    exclude_runtime_tag: Whether to leave out the hosting runtime suffix (see
      get_tracking_headers).
  """
  return types.HttpOptions(
      headers=get_tracking_headers(exclude_runtime_tag=exclude_runtime_tag)
  )


def merge_tracking_headers(
    headers: dict[str, str] | None,
    framework_label: str | None = None,
    *,
    exclude_runtime_tag: bool = False,
) -> dict[str, str]:
  """Merge tracking headers to the given headers.

  Args:
    headers: headers to merge tracking headers into.
    framework_label: Optional SemVer build-metadata suffix appended to the
      google-adk framework token (e.g. "managed_agent"), used to distinguish a
      specific ADK surface in Google's server-side usage pipeline.
    exclude_runtime_tag: Whether to leave out the hosting runtime suffix (see
      get_tracking_headers).

  Returns:
    A dictionary of HTTP headers with tracking headers merged.
  """
  tracking_headers = get_tracking_headers(
      framework_label=framework_label, exclude_runtime_tag=exclude_runtime_tag
  )
  new_headers = {
      key: value
      for key, value in (headers or {}).items()
      if key.lower() not in tracking_headers
  }
  for key, tracking_header_value in tracking_headers.items():
    # Merge tracking headers with existing headers and avoid duplicates.
    value_parts = tracking_header_value.split(" ")
    for header_key, custom_value in (headers or {}).items():
      if header_key.lower() != key or not custom_value:
        continue
      for custom_value_part in custom_value.split(" "):
        if custom_value_part not in value_parts:
          value_parts.append(custom_value_part)
    new_headers[key] = " ".join(value_parts)
  return new_headers


def merge_tracking_http_options(
    http_options: types.HttpOptions | types.HttpOptionsDict | None = None,
    framework_label: str | None = None,
    *,
    exclude_runtime_tag: bool = False,
) -> types.HttpOptions | types.HttpOptionsDict:
  """Merges tracking headers into the given HttpOptions or HttpOptionsDict.

  Args:
    http_options: Optional HttpOptions or HttpOptionsDict to merge tracking
      headers into. If None or empty, a new HttpOptions instance is returned.
    framework_label: Optional SemVer build-metadata suffix appended to the
      google-adk framework token (e.g. "managed_agent"), used to distinguish a
      specific ADK surface in Google's server-side usage pipeline.
    exclude_runtime_tag: Whether to leave out the hosting runtime suffix (see
      get_tracking_headers).

  Returns:
    HttpOptions or HttpOptionsDict with tracking headers merged.
  """
  if not http_options:
    http_options = types.HttpOptions()
  if isinstance(http_options, dict):
    headers = http_options.get("headers")
    http_options["headers"] = merge_tracking_headers(
        headers,
        framework_label=framework_label,
        exclude_runtime_tag=exclude_runtime_tag,
    )
  else:
    http_options.headers = merge_tracking_headers(
        http_options.headers,
        framework_label=framework_label,
        exclude_runtime_tag=exclude_runtime_tag,
    )
  return http_options
