#!/usr/bin/env python3
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

"""Export Cloud Trace data to BigQuery.

This script demonstrates the EXTRA EFFORT required to get Cloud Trace data
into BigQuery for analytics, compared to the BigQuery Agent Analytics Plugin
which does this automatically.

COMPARISON:
  BigQuery Plugin:     Automatic - data flows directly to BigQuery
  This approach:       Manual ETL script that must be run periodically

Usage:
  python export_traces_to_bigquery.py [--hours=1] [--trace-ids=id1,id2,...]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import google.auth
from google.cloud import bigquery
import requests


PROJECT_ID = os.environ.get("PROJECT_ID", "test-project-0728-467323")
DATASET_ID = "cloud_trace_export"
TABLE_ID = "agent_traces"


def get_access_token() -> str:
  """Get access token for Cloud Trace API."""
  credentials, _ = google.auth.default()
  credentials.refresh(google.auth.transport.requests.Request())
  return credentials.token


def fetch_trace(trace_id: str, token: str) -> dict[str, Any] | None:
  """Fetch a single trace from Cloud Trace API."""
  url = f"https://cloudtrace.googleapis.com/v1/projects/{PROJECT_ID}/traces/{trace_id}"
  headers = {"Authorization": f"Bearer {token}"}
  response = requests.get(url, headers=headers)
  if response.status_code == 200:
    return response.json()
  print(f"  Warning: Could not fetch trace {trace_id}: {response.status_code}")
  return None


def fetch_recent_traces(hours: int, token: str) -> list[str]:
  """Fetch trace IDs from the last N hours.

  Note: Cloud Trace list API is limited - may not return all traces.
  This is another limitation compared to BigQuery plugin.
  """
  end_time = datetime.now(timezone.utc)
  start_time = end_time - timedelta(hours=hours)

  url = (
      f"https://cloudtrace.googleapis.com/v1/projects/{PROJECT_ID}/traces"
      f"?startTime={start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
      f"&endTime={end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
      f"&pageSize=100"
  )
  headers = {"Authorization": f"Bearer {token}"}

  trace_ids = []
  while url:
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
      print(f"  Warning: List traces failed: {response.status_code}")
      break

    data = response.json()
    for trace in data.get("traces", []):
      trace_ids.append(trace.get("traceId"))

    # Handle pagination
    next_page = data.get("nextPageToken")
    if next_page:
      url = (
          f"https://cloudtrace.googleapis.com/v1/projects/{PROJECT_ID}/traces"
          f"?startTime={start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
          f"&endTime={end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
          f"&pageSize=100"
          f"&pageToken={next_page}"
      )
    else:
      url = None

  return trace_ids


def parse_span_to_row(trace_id: str, span: dict[str, Any]) -> dict[str, Any]:
  """Parse a Cloud Trace span into a BigQuery row.

  This is the complex transformation logic needed because Cloud Trace
  stores data in span labels, not structured fields like BigQuery plugin.
  """
  labels = span.get("labels", {})

  # Parse timestamps
  start_time = span.get("startTime", "")
  end_time = span.get("endTime", "")

  # Calculate duration
  duration_ms = None
  if start_time and end_time:
    try:
      start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
      end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
      duration_ms = (end_dt - start_dt).total_seconds() * 1000
    except (ValueError, TypeError):
      pass

  # Extract token counts (may be in different label formats)
  input_tokens = None
  output_tokens = None
  try:
    input_tokens = int(labels.get("gen_ai.usage.input_tokens", 0)) or None
    output_tokens = int(labels.get("gen_ai.usage.output_tokens", 0)) or None
  except (ValueError, TypeError):
    pass

  total_tokens = None
  if input_tokens and output_tokens:
    total_tokens = input_tokens + output_tokens

  return {
      "trace_id": trace_id,
      "span_id": span.get("spanId", ""),
      "parent_span_id": span.get("parentSpanId"),
      "span_name": span.get("name", ""),
      "start_time": start_time,
      "end_time": end_time,
      "duration_ms": duration_ms,
      "service_name": labels.get("service.name"),
      "agent_name": labels.get("gen_ai.agent.name"),
      "session_id": labels.get("gen_ai.conversation.id") or labels.get("gcp.vertex.agent.session_id"),
      "operation_name": labels.get("gen_ai.operation.name"),
      "tool_name": labels.get("gen_ai.tool.name"),
      "tool_call_id": labels.get("gen_ai.tool.call.id"),
      "tool_args": labels.get("gcp.vertex.agent.tool_call_args"),
      "tool_response": labels.get("gcp.vertex.agent.tool_response"),
      "input_tokens": input_tokens,
      "output_tokens": output_tokens,
      "total_tokens": total_tokens,
      "llm_request": labels.get("gcp.vertex.agent.llm_request"),
      "llm_response": labels.get("gcp.vertex.agent.llm_response"),
      "model": labels.get("gen_ai.request.model"),
      "finish_reason": json.dumps(labels.get("gen_ai.response.finish_reasons")) if labels.get("gen_ai.response.finish_reasons") else None,
      "labels": json.dumps(labels),
      "exported_at": datetime.now(timezone.utc).isoformat(),
  }


def export_traces_to_bigquery(trace_ids: list[str]) -> int:
  """Export traces to BigQuery."""
  if not trace_ids:
    print("No trace IDs to export")
    return 0

  print(f"Exporting {len(trace_ids)} traces to BigQuery...")

  token = get_access_token()
  rows = []

  for i, trace_id in enumerate(trace_ids):
    print(f"  [{i+1}/{len(trace_ids)}] Fetching trace: {trace_id}")
    trace_data = fetch_trace(trace_id, token)
    if trace_data:
      for span in trace_data.get("spans", []):
        row = parse_span_to_row(trace_id, span)
        rows.append(row)

  if not rows:
    print("No spans found to export")
    return 0

  print(f"  Found {len(rows)} spans, loading to BigQuery...")

  # Load to BigQuery
  client = bigquery.Client(project=PROJECT_ID)
  table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

  errors = client.insert_rows_json(table_ref, rows)
  if errors:
    print(f"  Errors inserting rows: {errors}")
    return 0

  print(f"  Successfully exported {len(rows)} spans to {table_ref}")
  return len(rows)


def main():
  parser = argparse.ArgumentParser(description="Export Cloud Trace to BigQuery")
  parser.add_argument("--hours", type=int, default=1, help="Export traces from last N hours")
  parser.add_argument("--trace-ids", type=str, help="Comma-separated trace IDs to export")
  args = parser.parse_args()

  print("=" * 60)
  print("Cloud Trace to BigQuery Export")
  print("=" * 60)
  print(f"Project: {PROJECT_ID}")
  print(f"Target:  {DATASET_ID}.{TABLE_ID}")
  print("=" * 60)
  print()
  print("NOTE: This manual ETL is required because Cloud Trace")
  print("      does NOT have native BigQuery export.")
  print()
  print("      The BigQuery Agent Analytics Plugin does this")
  print("      AUTOMATICALLY with zero additional code.")
  print()
  print("=" * 60)

  if args.trace_ids:
    trace_ids = [t.strip() for t in args.trace_ids.split(",")]
    print(f"Exporting {len(trace_ids)} specified traces...")
  else:
    print(f"Fetching traces from the last {args.hours} hour(s)...")
    token = get_access_token()
    trace_ids = fetch_recent_traces(args.hours, token)
    print(f"  Found {len(trace_ids)} traces")

  if trace_ids:
    count = export_traces_to_bigquery(trace_ids)
    print()
    print("=" * 60)
    print(f"Export complete! {count} spans exported.")
    print("=" * 60)
    print()
    print("Query your data:")
    print(f"  SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` LIMIT 10")
  else:
    print("No traces found to export")


if __name__ == "__main__":
  main()
