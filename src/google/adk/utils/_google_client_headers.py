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

from __future__ import annotations

from ._client_labels_utils import get_client_labels


def get_tracking_headers() -> dict[str, str]:
  """Returns a dictionary of HTTP headers for tracking API requests.

  These headers are used to identify HTTP calls made by ADK towards
   Vertex AI LLM APIs.
  """
  labels = get_client_labels()
  header_value = " ".join(labels)
  return {
      "x-goog-api-client": header_value,
      "user-agent": header_value,
  }


def merge_tracking_headers(headers: dict[str, str] | None) -> dict[str, str]:
  """Merge tracking headers to the given headers.

  Args:
    headers: headers to merge tracking headers into.

  Returns:
    A dictionary of HTTP headers with tracking headers merged.
  """
  new_headers = (headers or {}).copy()
  for key, tracking_header_value in get_tracking_headers().items():
    custom_value = new_headers.get(key, None)
    if not custom_value:
      new_headers[key] = tracking_header_value
      continue

    # Merge tracking headers with existing headers and avoid duplicates.
    value_parts = tracking_header_value.split(" ")
    for custom_value_part in custom_value.split(" "):
      if custom_value_part not in value_parts:
        value_parts.append(custom_value_part)
    new_headers[key] = " ".join(value_parts)
  return new_headers
