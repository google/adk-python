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

import json
from typing import Any

import requests


def get_stream(
    url: str,
    ca_payload: dict[str, Any],
    headers: dict[str, str],
    max_query_result_rows: int,
) -> list[dict[str, Any]]:
  """Sends a JSON request to a streaming API and returns a list of messages."""
  with requests.Session() as s:
    accumulator = ""
    messages = []
    data_msg_idx = -1

    with s.post(url, json=ca_payload, headers=headers, stream=True) as resp:
      resp.raise_for_status()
      for line in resp.iter_lines():
        if not line:
          continue

        decoded_line = line.decode("utf-8")

        if decoded_line == "[{":
          accumulator = "{"
        elif decoded_line == "}]":
          accumulator += "}"
        elif decoded_line == ",":
          continue
        else:
          accumulator += decoded_line

        try:
          data_json = json.loads(accumulator)
        except ValueError:
          continue

        accumulator = ""

        if not isinstance(data_json, dict):
          messages.append(data_json)
          continue

        processed_msg = None
        data_result = _extract_data_result(data_json)
        if data_result is not None:
          processed_msg = _format_data_retrieved(
              data_result, max_query_result_rows
          )
          if data_msg_idx >= 0:
            messages[data_msg_idx] = {
                "Data Retrieved": "Intermediate result omitted"
            }
          data_msg_idx = len(messages)
        elif isinstance(data_json.get("systemMessage"), dict):
          processed_msg = data_json["systemMessage"]
        else:
          processed_msg = data_json

        if processed_msg is not None:
          messages.append(processed_msg)

    return messages


def _extract_data_result(msg: dict[str, Any]) -> dict[str, Any] | None:
  """Attempts to find the result.data deep inside the generic dict."""
  sm = msg.get("systemMessage")
  if not isinstance(sm, dict):
    return None
  data = sm.get("data")
  if not isinstance(data, dict):
    return None
  result = data.get("result")
  if not isinstance(result, dict):
    return None
  if "data" in result and isinstance(result["data"], list):
    return result
  return None


def _format_data_retrieved(
    result: dict[str, Any], max_rows: int
) -> dict[str, Any]:
  """Transforms the raw result dict into the simplified Toolbox format."""
  raw_data = result.get("data", [])

  fields = []
  schema = result.get("schema")
  if isinstance(schema, dict):
    schema_fields = schema.get("fields")
    if isinstance(schema_fields, list):
      fields = schema_fields

  headers = []
  for f in fields:
    if isinstance(f, dict):
      name = f.get("name")
      if isinstance(name, str):
        headers.append(name)

  if not headers and raw_data:
    first_row = raw_data[0]
    if isinstance(first_row, dict):
      headers = list(first_row.keys())

  total_rows = len(raw_data)
  num_to_display = min(total_rows, max_rows)

  rows = []
  for r in raw_data[:num_to_display]:
    if isinstance(r, dict):
      row = [r.get(h) for h in headers]
      rows.append(row)

  summary = f"Showing all {total_rows} rows."
  if total_rows > max_rows:
    summary = f"Showing the first {num_to_display} of {total_rows} total rows."

  return {
      "Data Retrieved": {
          "headers": headers,
          "rows": rows,
          "summary": summary,
      }
  }
