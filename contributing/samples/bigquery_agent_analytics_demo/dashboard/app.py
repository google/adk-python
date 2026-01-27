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

"""FastAPI Agent Analytics Dashboard.

A comprehensive monitoring dashboard for agent behavior analytics stored in
BigQuery. Provides real-time insights into agent performance, tool usage,
error tracking, and session analysis.

Usage:
    uvicorn dashboard.app:app --reload --port 8080

Environment Variables:
    BQ_AGENT_ANALYTICS_PROJECT: Google Cloud project ID
    BQ_AGENT_ANALYTICS_DATASET: BigQuery dataset ID
    BQ_AGENT_ANALYTICS_TABLE: BigQuery table name (default: agent_events_v2)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import bigquery
from pydantic import BaseModel
from pydantic import Field


# Configuration from environment
PROJECT_ID = os.getenv("BQ_AGENT_ANALYTICS_PROJECT")
DATASET_ID = os.getenv("BQ_AGENT_ANALYTICS_DATASET")
TABLE_ID = os.getenv("BQ_AGENT_ANALYTICS_TABLE", "agent_events_v2")

app = FastAPI(
    title="Agent Analytics Dashboard",
    description="Real-time monitoring dashboard for ADK agent behavior",
    version="1.0.0",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_bq_client() -> bigquery.Client:
  """Get or create BigQuery client."""
  if not PROJECT_ID:
    raise HTTPException(
        status_code=500,
        detail="BQ_AGENT_ANALYTICS_PROJECT environment variable not set",
    )
  return bigquery.Client(project=PROJECT_ID)


def get_full_table_id() -> str:
  """Get full BigQuery table ID."""
  if not PROJECT_ID or not DATASET_ID:
    raise HTTPException(
        status_code=500,
        detail="BigQuery configuration not set. Please set "
        "BQ_AGENT_ANALYTICS_PROJECT and BQ_AGENT_ANALYTICS_DATASET",
    )
  return f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"


# =============================================================================
# Pydantic Models
# =============================================================================


class TimeRange(BaseModel):
  """Time range for queries."""

  start: datetime = Field(
      default_factory=lambda: datetime.now(timezone.utc) - timedelta(hours=24)
  )
  end: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OverviewMetrics(BaseModel):
  """Overview dashboard metrics."""

  total_sessions: int
  total_invocations: int
  total_events: int
  error_rate: float
  avg_latency_ms: float
  total_tokens: int
  unique_users: int
  unique_agents: int


class EventTypeCount(BaseModel):
  """Event type count."""

  event_type: Optional[str]
  count: int
  percentage: float


class ToolMetrics(BaseModel):
  """Tool usage metrics."""

  tool_name: str
  invocations: int
  success_count: int
  error_count: int
  success_rate: float
  avg_latency_ms: float


class SessionSummary(BaseModel):
  """Session summary."""

  session_id: Optional[str]
  user_id: Optional[str]
  agent: Optional[str]
  start_time: datetime
  end_time: Optional[datetime]
  duration_seconds: Optional[float]
  event_count: int
  error_count: int
  status: str


class EventDetail(BaseModel):
  """Detailed event information."""

  timestamp: datetime
  event_type: Optional[str]
  agent: Optional[str]
  session_id: Optional[str]
  invocation_id: Optional[str]
  span_id: Optional[str]
  parent_span_id: Optional[str]
  content: Optional[dict]
  latency_ms: Optional[dict]
  status: str
  error_message: Optional[str]


class ErrorSummary(BaseModel):
  """Error summary."""

  event_type: str
  error_message: str
  count: int
  last_occurrence: datetime
  affected_sessions: int


class LLMMetrics(BaseModel):
  """LLM performance metrics."""

  total_requests: int
  total_responses: int
  error_count: int
  avg_latency_ms: float
  avg_time_to_first_token_ms: Optional[float]
  total_prompt_tokens: int
  total_completion_tokens: int
  total_tokens: int
  avg_tokens_per_request: float


class TimeSeriesPoint(BaseModel):
  """Time series data point."""

  timestamp: datetime
  value: float
  label: Optional[str] = None


class LatencyPercentiles(BaseModel):
  """Latency percentile metrics."""

  min_ms: float
  max_ms: float
  avg_ms: float
  median_ms: float
  p90_ms: float
  p95_ms: float
  p99_ms: float
  std_dev_ms: float
  sample_count: int


class LatencyByCategory(BaseModel):
  """Latency breakdown by category."""

  category: str
  count: int
  avg_ms: float
  min_ms: float
  max_ms: float
  p50_ms: float
  p90_ms: float
  p95_ms: float


class SlowestRequest(BaseModel):
  """Details of a slow request."""

  timestamp: datetime
  event_type: Optional[str]
  agent: Optional[str]
  session_id: Optional[str]
  latency_ms: float
  content_preview: Optional[str]


class LatencyDistributionBucket(BaseModel):
  """Latency distribution histogram bucket."""

  bucket_start_ms: int
  bucket_end_ms: int
  count: int
  percentage: float


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/api/health")
async def health_check():
  """Health check endpoint."""
  return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}


@app.get("/api/config")
async def get_config():
  """Get current configuration."""
  return {
      "project_id": PROJECT_ID,
      "dataset_id": DATASET_ID,
      "table_id": TABLE_ID,
      "configured": bool(PROJECT_ID and DATASET_ID),
  }


@app.get("/api/overview", response_model=OverviewMetrics)
async def get_overview_metrics(
    hours: int = Query(default=24, ge=1, le=720),
) -> OverviewMetrics:
  """Get overview dashboard metrics."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  WITH base_data AS (
    SELECT
      session_id,
      invocation_id,
      event_type,
      user_id,
      agent,
      status,
      JSON_VALUE(latency_ms, '$.total_ms') AS latency,
      COALESCE(
        SAFE_CAST(JSON_VALUE(attributes, '$.usage_metadata.total_token_count') AS INT64),
        SAFE_CAST(JSON_VALUE(content, '$.usage.total') AS INT64)
      ) AS tokens
    FROM {table}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
      AND event_type IN (
        'USER_MESSAGE_RECEIVED', 'INVOCATION_STARTING', 'INVOCATION_COMPLETED',
        'AGENT_STARTING', 'AGENT_COMPLETED', 'LLM_REQUEST', 'LLM_RESPONSE',
        'LLM_ERROR', 'TOOL_STARTING', 'TOOL_COMPLETED', 'TOOL_ERROR'
      )
  )
  SELECT
    COUNT(DISTINCT session_id) AS total_sessions,
    COUNT(DISTINCT invocation_id) AS total_invocations,
    COUNT(*) AS total_events,
    SAFE_DIVIDE(
      COUNTIF(status = 'ERROR'),
      COUNT(*)
    ) * 100 AS error_rate,
    AVG(SAFE_CAST(latency AS FLOAT64)) AS avg_latency_ms,
    SUM(tokens) AS total_tokens,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(DISTINCT agent) AS unique_agents
  FROM base_data
  """

  result = client.query(query).result()
  row = list(result)[0]

  return OverviewMetrics(
      total_sessions=row.total_sessions or 0,
      total_invocations=row.total_invocations or 0,
      total_events=row.total_events or 0,
      error_rate=row.error_rate or 0.0,
      avg_latency_ms=row.avg_latency_ms or 0.0,
      total_tokens=row.total_tokens or 0,
      unique_users=row.unique_users or 0,
      unique_agents=row.unique_agents or 0,
  )


@app.get("/api/events/types", response_model=list[EventTypeCount])
async def get_event_type_counts(
    hours: int = Query(default=24, ge=1, le=720),
) -> list[EventTypeCount]:
  """Get event counts by type."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  WITH counts AS (
    SELECT
      event_type,
      COUNT(*) AS count
    FROM {table}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    GROUP BY event_type
  ),
  total AS (
    SELECT SUM(count) AS total FROM counts
  )
  SELECT
    c.event_type,
    c.count,
    SAFE_DIVIDE(c.count, t.total) * 100 AS percentage
  FROM counts c, total t
  ORDER BY c.count DESC
  """

  result = client.query(query).result()
  return [
      EventTypeCount(
          event_type=row.event_type,
          count=row.count,
          percentage=row.percentage or 0.0,
      )
      for row in result
  ]


@app.get("/api/events/timeline", response_model=list[TimeSeriesPoint])
async def get_event_timeline(
    hours: int = Query(default=24, ge=1, le=720),
    interval_minutes: int = Query(default=60, ge=5, le=1440),
) -> list[TimeSeriesPoint]:
  """Get event count timeline."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  SELECT
    TIMESTAMP_TRUNC(timestamp, MINUTE) AS bucket,
    COUNT(*) AS count
  FROM {table}
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
  GROUP BY bucket
  ORDER BY bucket
  """

  result = client.query(query).result()
  return [
      TimeSeriesPoint(timestamp=row.bucket, value=float(row.count))
      for row in result
  ]


@app.get("/api/tools/metrics", response_model=list[ToolMetrics])
async def get_tool_metrics(
    hours: int = Query(default=24, ge=1, le=720),
) -> list[ToolMetrics]:
  """Get tool usage metrics."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  WITH tool_events AS (
    SELECT
      JSON_VALUE(content, '$.tool') AS tool_name,
      event_type,
      status,
      SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64) AS latency
    FROM {table}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
      AND event_type IN ('TOOL_STARTING', 'TOOL_COMPLETED', 'TOOL_ERROR')
  )
  SELECT
    tool_name,
    COUNTIF(event_type = 'TOOL_STARTING') AS invocations,
    COUNTIF(event_type = 'TOOL_COMPLETED') AS success_count,
    COUNTIF(event_type = 'TOOL_ERROR') AS error_count,
    SAFE_DIVIDE(
      COUNTIF(event_type = 'TOOL_COMPLETED'),
      COUNTIF(event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR'))
    ) * 100 AS success_rate,
    AVG(CASE WHEN event_type = 'TOOL_COMPLETED' THEN latency END) AS avg_latency_ms
  FROM tool_events
  WHERE tool_name IS NOT NULL
  GROUP BY tool_name
  ORDER BY invocations DESC
  """

  result = client.query(query).result()
  return [
      ToolMetrics(
          tool_name=row.tool_name,
          invocations=row.invocations or 0,
          success_count=row.success_count or 0,
          error_count=row.error_count or 0,
          success_rate=row.success_rate or 0.0,
          avg_latency_ms=row.avg_latency_ms or 0.0,
      )
      for row in result
  ]


@app.get("/api/sessions", response_model=list[SessionSummary])
async def get_sessions(
    hours: int = Query(default=24, ge=1, le=720),
    limit: int = Query(default=50, ge=1, le=500),
    status_filter: Optional[str] = Query(default=None),
) -> list[SessionSummary]:
  """Get session summaries."""
  client = get_bq_client()
  table = get_full_table_id()

  status_clause = ""
  if status_filter:
    status_clause = f"HAVING MAX(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) = {'1' if status_filter == 'ERROR' else '0'}"

  query = f"""
  SELECT
    session_id,
    ANY_VALUE(user_id) AS user_id,
    ANY_VALUE(agent) AS agent,
    MIN(timestamp) AS start_time,
    MAX(timestamp) AS end_time,
    TIMESTAMP_DIFF(MAX(timestamp), MIN(timestamp), SECOND) AS duration_seconds,
    COUNT(*) AS event_count,
    COUNTIF(status = 'ERROR') AS error_count,
    CASE
      WHEN MAX(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) = 1 THEN 'ERROR'
      ELSE 'OK'
    END AS status
  FROM {table}
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    AND session_id IS NOT NULL
    AND event_type IN (
      'USER_MESSAGE_RECEIVED', 'INVOCATION_STARTING', 'INVOCATION_COMPLETED',
      'AGENT_STARTING', 'AGENT_COMPLETED', 'LLM_REQUEST', 'LLM_RESPONSE',
      'LLM_ERROR', 'TOOL_STARTING', 'TOOL_COMPLETED', 'TOOL_ERROR'
    )
  GROUP BY session_id
  {status_clause}
  ORDER BY start_time DESC
  LIMIT {limit}
  """

  result = client.query(query).result()
  return [
      SessionSummary(
          session_id=row.session_id,
          user_id=row.user_id,
          agent=row.agent,
          start_time=row.start_time,
          end_time=row.end_time,
          duration_seconds=row.duration_seconds,
          event_count=row.event_count,
          error_count=row.error_count,
          status=row.status,
      )
      for row in result
  ]


@app.get("/api/sessions/{session_id}/events", response_model=list[EventDetail])
async def get_session_events(session_id: str) -> list[EventDetail]:
  """Get all events for a specific session."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  SELECT
    timestamp,
    event_type,
    agent,
    session_id,
    invocation_id,
    span_id,
    parent_span_id,
    content,
    latency_ms,
    status,
    error_message
  FROM {table}
  WHERE session_id = @session_id
  ORDER BY timestamp ASC
  """

  job_config = bigquery.QueryJobConfig(
      query_parameters=[
          bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
      ]
  )

  result = client.query(query, job_config=job_config).result()
  events = []
  for row in result:
    content = None
    if row.content:
      try:
        content = (
            json.loads(row.content)
            if isinstance(row.content, str)
            else row.content
        )
      except (json.JSONDecodeError, TypeError):
        content = {"raw": str(row.content)}

    latency = None
    if row.latency_ms:
      try:
        latency = (
            json.loads(row.latency_ms)
            if isinstance(row.latency_ms, str)
            else row.latency_ms
        )
      except (json.JSONDecodeError, TypeError):
        latency = None

    events.append(
        EventDetail(
            timestamp=row.timestamp,
            event_type=row.event_type,
            agent=row.agent,
            session_id=row.session_id,
            invocation_id=row.invocation_id,
            span_id=row.span_id,
            parent_span_id=row.parent_span_id,
            content=content,
            latency_ms=latency,
            status=row.status or "OK",
            error_message=row.error_message,
        )
    )
  return events


@app.get("/api/errors", response_model=list[ErrorSummary])
async def get_error_summary(
    hours: int = Query(default=24, ge=1, le=720),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ErrorSummary]:
  """Get error summary."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  SELECT
    event_type,
    error_message,
    COUNT(*) AS count,
    MAX(timestamp) AS last_occurrence,
    COUNT(DISTINCT session_id) AS affected_sessions
  FROM {table}
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    AND status = 'ERROR'
    AND error_message IS NOT NULL
  GROUP BY event_type, error_message
  ORDER BY count DESC
  LIMIT {limit}
  """

  result = client.query(query).result()
  return [
      ErrorSummary(
          event_type=row.event_type,
          error_message=row.error_message,
          count=row.count,
          last_occurrence=row.last_occurrence,
          affected_sessions=row.affected_sessions,
      )
      for row in result
  ]


@app.get("/api/llm/metrics", response_model=LLMMetrics)
async def get_llm_metrics(
    hours: int = Query(default=24, ge=1, le=720),
) -> LLMMetrics:
  """Get LLM performance metrics."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  WITH llm_events AS (
    SELECT
      event_type,
      status,
      SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64) AS latency,
      SAFE_CAST(JSON_VALUE(latency_ms, '$.time_to_first_token_ms') AS FLOAT64) AS ttft,
      COALESCE(
        SAFE_CAST(JSON_VALUE(attributes, '$.usage_metadata.prompt_token_count') AS INT64),
        SAFE_CAST(JSON_VALUE(content, '$.usage.prompt') AS INT64)
      ) AS prompt_tokens,
      COALESCE(
        SAFE_CAST(JSON_VALUE(attributes, '$.usage_metadata.candidates_token_count') AS INT64),
        SAFE_CAST(JSON_VALUE(content, '$.usage.completion') AS INT64)
      ) AS completion_tokens,
      COALESCE(
        SAFE_CAST(JSON_VALUE(attributes, '$.usage_metadata.total_token_count') AS INT64),
        SAFE_CAST(JSON_VALUE(content, '$.usage.total') AS INT64)
      ) AS total_tokens_per_event
    FROM {table}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
      AND event_type IN ('LLM_REQUEST', 'LLM_RESPONSE', 'LLM_ERROR')
  )
  SELECT
    COUNTIF(event_type = 'LLM_REQUEST') AS total_requests,
    COUNTIF(event_type = 'LLM_RESPONSE') AS total_responses,
    COUNTIF(event_type = 'LLM_ERROR') AS error_count,
    AVG(CASE WHEN event_type = 'LLM_RESPONSE' THEN latency END) AS avg_latency_ms,
    AVG(CASE WHEN event_type = 'LLM_RESPONSE' THEN ttft END) AS avg_ttft_ms,
    SUM(CASE WHEN event_type = 'LLM_RESPONSE' THEN prompt_tokens ELSE 0 END) AS total_prompt_tokens,
    SUM(CASE WHEN event_type = 'LLM_RESPONSE' THEN completion_tokens ELSE 0 END) AS total_completion_tokens,
    SUM(CASE WHEN event_type = 'LLM_RESPONSE' THEN total_tokens_per_event ELSE 0 END) AS total_tokens
  FROM llm_events
  """

  result = client.query(query).result()
  row = list(result)[0]

  total_requests = row.total_requests or 0
  total_tokens = row.total_tokens or 0

  return LLMMetrics(
      total_requests=total_requests,
      total_responses=row.total_responses or 0,
      error_count=row.error_count or 0,
      avg_latency_ms=row.avg_latency_ms or 0.0,
      avg_time_to_first_token_ms=row.avg_ttft_ms,
      total_prompt_tokens=row.total_prompt_tokens or 0,
      total_completion_tokens=row.total_completion_tokens or 0,
      total_tokens=total_tokens,
      avg_tokens_per_request=(
          total_tokens / total_requests if total_requests > 0 else 0.0
      ),
  )


@app.get("/api/llm/latency-timeline", response_model=list[TimeSeriesPoint])
async def get_llm_latency_timeline(
    hours: int = Query(default=24, ge=1, le=720),
) -> list[TimeSeriesPoint]:
  """Get LLM latency timeline."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  SELECT
    TIMESTAMP_TRUNC(timestamp, HOUR) AS bucket,
    AVG(SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64)) AS avg_latency
  FROM {table}
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    AND event_type = 'LLM_RESPONSE'
  GROUP BY bucket
  ORDER BY bucket
  """

  result = client.query(query).result()
  return [
      TimeSeriesPoint(
          timestamp=row.bucket,
          value=row.avg_latency or 0.0,
          label="Avg Latency (ms)",
      )
      for row in result
  ]


@app.get("/api/agents", response_model=list[dict])
async def get_agent_summary(
    hours: int = Query(default=24, ge=1, le=720),
) -> list[dict]:
  """Get summary metrics per agent."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  SELECT
    agent,
    COUNT(*) AS total_events,
    COUNT(DISTINCT session_id) AS total_sessions,
    COUNTIF(status = 'ERROR') AS error_count,
    SAFE_DIVIDE(COUNTIF(status = 'ERROR'), COUNT(*)) * 100 AS error_rate,
    AVG(SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64)) AS avg_latency_ms
  FROM {table}
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
  GROUP BY agent
  ORDER BY total_events DESC
  """

  result = client.query(query).result()
  return [
      {
          "agent": row.agent,
          "total_events": row.total_events,
          "total_sessions": row.total_sessions,
          "error_count": row.error_count,
          "error_rate": row.error_rate or 0.0,
          "avg_latency_ms": row.avg_latency_ms or 0.0,
      }
      for row in result
  ]


@app.get("/api/latency/percentiles", response_model=LatencyPercentiles)
async def get_latency_percentiles(
    hours: int = Query(default=24, ge=1, le=720),
    event_type: Optional[str] = Query(default=None),
) -> LatencyPercentiles:
  """Get latency percentile metrics."""
  client = get_bq_client()
  table = get_full_table_id()

  event_filter = ""
  if event_type:
    event_filter = f"AND event_type = '{event_type}'"

  query = f"""
  WITH latencies AS (
    SELECT
      SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64) AS latency
    FROM {table}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
      AND JSON_VALUE(latency_ms, '$.total_ms') IS NOT NULL
      {event_filter}
  )
  SELECT
    MIN(latency) AS min_ms,
    MAX(latency) AS max_ms,
    AVG(latency) AS avg_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(50)] AS median_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(90)] AS p90_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(95)] AS p95_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(99)] AS p99_ms,
    STDDEV(latency) AS std_dev_ms,
    COUNT(*) AS sample_count
  FROM latencies
  WHERE latency IS NOT NULL
  """

  result = client.query(query).result()
  row = list(result)[0]

  return LatencyPercentiles(
      min_ms=row.min_ms or 0.0,
      max_ms=row.max_ms or 0.0,
      avg_ms=row.avg_ms or 0.0,
      median_ms=row.median_ms or 0.0,
      p90_ms=row.p90_ms or 0.0,
      p95_ms=row.p95_ms or 0.0,
      p99_ms=row.p99_ms or 0.0,
      std_dev_ms=row.std_dev_ms or 0.0,
      sample_count=row.sample_count or 0,
  )


@app.get("/api/latency/by-event-type", response_model=list[LatencyByCategory])
async def get_latency_by_event_type(
    hours: int = Query(default=24, ge=1, le=720),
) -> list[LatencyByCategory]:
  """Get latency breakdown by event type."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  WITH latencies AS (
    SELECT
      event_type,
      SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64) AS latency
    FROM {table}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
      AND JSON_VALUE(latency_ms, '$.total_ms') IS NOT NULL
  )
  SELECT
    event_type AS category,
    COUNT(*) AS count,
    AVG(latency) AS avg_ms,
    MIN(latency) AS min_ms,
    MAX(latency) AS max_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(50)] AS p50_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(90)] AS p90_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(95)] AS p95_ms
  FROM latencies
  WHERE latency IS NOT NULL AND event_type IS NOT NULL
  GROUP BY event_type
  ORDER BY avg_ms DESC
  """

  result = client.query(query).result()
  return [
      LatencyByCategory(
          category=row.category,
          count=row.count,
          avg_ms=row.avg_ms or 0.0,
          min_ms=row.min_ms or 0.0,
          max_ms=row.max_ms or 0.0,
          p50_ms=row.p50_ms or 0.0,
          p90_ms=row.p90_ms or 0.0,
          p95_ms=row.p95_ms or 0.0,
      )
      for row in result
  ]


@app.get("/api/latency/by-agent", response_model=list[LatencyByCategory])
async def get_latency_by_agent(
    hours: int = Query(default=24, ge=1, le=720),
) -> list[LatencyByCategory]:
  """Get latency breakdown by agent."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  WITH latencies AS (
    SELECT
      agent,
      SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64) AS latency
    FROM {table}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
      AND JSON_VALUE(latency_ms, '$.total_ms') IS NOT NULL
  )
  SELECT
    agent AS category,
    COUNT(*) AS count,
    AVG(latency) AS avg_ms,
    MIN(latency) AS min_ms,
    MAX(latency) AS max_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(50)] AS p50_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(90)] AS p90_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(95)] AS p95_ms
  FROM latencies
  WHERE latency IS NOT NULL AND agent IS NOT NULL
  GROUP BY agent
  ORDER BY avg_ms DESC
  """

  result = client.query(query).result()
  return [
      LatencyByCategory(
          category=row.category,
          count=row.count,
          avg_ms=row.avg_ms or 0.0,
          min_ms=row.min_ms or 0.0,
          max_ms=row.max_ms or 0.0,
          p50_ms=row.p50_ms or 0.0,
          p90_ms=row.p90_ms or 0.0,
          p95_ms=row.p95_ms or 0.0,
      )
      for row in result
  ]


@app.get("/api/latency/by-tool", response_model=list[LatencyByCategory])
async def get_latency_by_tool(
    hours: int = Query(default=24, ge=1, le=720),
) -> list[LatencyByCategory]:
  """Get latency breakdown by tool."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  WITH latencies AS (
    SELECT
      JSON_VALUE(content, '$.tool') AS tool_name,
      SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64) AS latency
    FROM {table}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
      AND event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR')
      AND JSON_VALUE(latency_ms, '$.total_ms') IS NOT NULL
  )
  SELECT
    tool_name AS category,
    COUNT(*) AS count,
    AVG(latency) AS avg_ms,
    MIN(latency) AS min_ms,
    MAX(latency) AS max_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(50)] AS p50_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(90)] AS p90_ms,
    APPROX_QUANTILES(latency, 100)[OFFSET(95)] AS p95_ms
  FROM latencies
  WHERE latency IS NOT NULL AND tool_name IS NOT NULL
  GROUP BY tool_name
  ORDER BY avg_ms DESC
  """

  result = client.query(query).result()
  return [
      LatencyByCategory(
          category=row.category,
          count=row.count,
          avg_ms=row.avg_ms or 0.0,
          min_ms=row.min_ms or 0.0,
          max_ms=row.max_ms or 0.0,
          p50_ms=row.p50_ms or 0.0,
          p90_ms=row.p90_ms or 0.0,
          p95_ms=row.p95_ms or 0.0,
      )
      for row in result
  ]


@app.get("/api/latency/distribution", response_model=list[LatencyDistributionBucket])
async def get_latency_distribution(
    hours: int = Query(default=24, ge=1, le=720),
    bucket_size_ms: int = Query(default=1000, ge=100, le=10000),
) -> list[LatencyDistributionBucket]:
  """Get latency distribution histogram."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  WITH latencies AS (
    SELECT
      SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64) AS latency
    FROM {table}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
      AND JSON_VALUE(latency_ms, '$.total_ms') IS NOT NULL
  ),
  bucketed AS (
    SELECT
      CAST(FLOOR(latency / {bucket_size_ms}) * {bucket_size_ms} AS INT64) AS bucket_start
    FROM latencies
    WHERE latency IS NOT NULL
  ),
  counts AS (
    SELECT bucket_start, COUNT(*) AS count
    FROM bucketed
    GROUP BY bucket_start
  ),
  total AS (
    SELECT SUM(count) AS total FROM counts
  )
  SELECT
    c.bucket_start AS bucket_start_ms,
    c.bucket_start + {bucket_size_ms} AS bucket_end_ms,
    c.count,
    SAFE_DIVIDE(c.count, t.total) * 100 AS percentage
  FROM counts c, total t
  ORDER BY bucket_start_ms
  LIMIT 50
  """

  result = client.query(query).result()
  return [
      LatencyDistributionBucket(
          bucket_start_ms=row.bucket_start_ms,
          bucket_end_ms=row.bucket_end_ms,
          count=row.count,
          percentage=row.percentage or 0.0,
      )
      for row in result
  ]


@app.get("/api/latency/slowest", response_model=list[SlowestRequest])
async def get_slowest_requests(
    hours: int = Query(default=24, ge=1, le=720),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[SlowestRequest]:
  """Get slowest requests."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  SELECT
    timestamp,
    event_type,
    agent,
    session_id,
    SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64) AS latency_ms,
    LEFT(COALESCE(
      JSON_VALUE(content, '$.summary'),
      JSON_VALUE(content, '$.tool'),
      TO_JSON_STRING(content)
    ), 100) AS content_preview
  FROM {table}
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    AND JSON_VALUE(latency_ms, '$.total_ms') IS NOT NULL
  ORDER BY latency_ms DESC
  LIMIT {limit}
  """

  result = client.query(query).result()
  return [
      SlowestRequest(
          timestamp=row.timestamp,
          event_type=row.event_type,
          agent=row.agent,
          session_id=row.session_id,
          latency_ms=row.latency_ms or 0.0,
          content_preview=row.content_preview,
      )
      for row in result
  ]


@app.get("/api/latency/timeline", response_model=list[TimeSeriesPoint])
async def get_latency_timeline(
    hours: int = Query(default=24, ge=1, le=720),
    metric: str = Query(default="avg", regex="^(avg|p50|p90|p95|p99)$"),
) -> list[TimeSeriesPoint]:
  """Get latency timeline with different percentile options."""
  client = get_bq_client()
  table = get_full_table_id()

  metric_expr = {
      "avg": "AVG(latency)",
      "p50": "APPROX_QUANTILES(latency, 100)[OFFSET(50)]",
      "p90": "APPROX_QUANTILES(latency, 100)[OFFSET(90)]",
      "p95": "APPROX_QUANTILES(latency, 100)[OFFSET(95)]",
      "p99": "APPROX_QUANTILES(latency, 100)[OFFSET(99)]",
  }

  query = f"""
  WITH latencies AS (
    SELECT
      TIMESTAMP_TRUNC(timestamp, HOUR) AS bucket,
      SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64) AS latency
    FROM {table}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
      AND JSON_VALUE(latency_ms, '$.total_ms') IS NOT NULL
  )
  SELECT
    bucket,
    {metric_expr[metric]} AS value
  FROM latencies
  WHERE latency IS NOT NULL
  GROUP BY bucket
  ORDER BY bucket
  """

  result = client.query(query).result()
  return [
      TimeSeriesPoint(
          timestamp=row.bucket,
          value=row.value or 0.0,
          label=f"{metric.upper()} Latency (ms)",
      )
      for row in result
  ]


@app.get("/api/trace/{trace_id}", response_model=list[EventDetail])
async def get_trace_events(trace_id: str) -> list[EventDetail]:
  """Get all events for a specific trace."""
  client = get_bq_client()
  table = get_full_table_id()

  query = f"""
  SELECT
    timestamp,
    event_type,
    agent,
    session_id,
    invocation_id,
    span_id,
    parent_span_id,
    content,
    latency_ms,
    status,
    error_message
  FROM {table}
  WHERE trace_id = @trace_id
  ORDER BY timestamp ASC
  """

  job_config = bigquery.QueryJobConfig(
      query_parameters=[
          bigquery.ScalarQueryParameter("trace_id", "STRING", trace_id),
      ]
  )

  result = client.query(query, job_config=job_config).result()
  events = []
  for row in result:
    content = None
    if row.content:
      try:
        content = (
            json.loads(row.content)
            if isinstance(row.content, str)
            else row.content
        )
      except (json.JSONDecodeError, TypeError):
        content = {"raw": str(row.content)}

    latency = None
    if row.latency_ms:
      try:
        latency = (
            json.loads(row.latency_ms)
            if isinstance(row.latency_ms, str)
            else row.latency_ms
        )
      except (json.JSONDecodeError, TypeError):
        latency = None

    events.append(
        EventDetail(
            timestamp=row.timestamp,
            event_type=row.event_type,
            agent=row.agent,
            session_id=row.session_id,
            invocation_id=row.invocation_id,
            span_id=row.span_id,
            parent_span_id=row.parent_span_id,
            content=content,
            latency_ms=latency,
            status=row.status or "OK",
            error_message=row.error_message,
        )
    )
  return events


# =============================================================================
# Frontend HTML Dashboard
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def dashboard():
  """Serve the main dashboard HTML."""
  return get_dashboard_html()


def get_dashboard_html() -> str:
  """Return the dashboard HTML with embedded JavaScript."""
  return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agent Analytics Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1"></script>
  <style>
    .metric-card {
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .status-ok { color: #10b981; }
    .status-error { color: #ef4444; }
    .event-row:hover { background-color: #f3f4f6; }
    .tab-active {
      border-bottom: 2px solid #3b82f6;
      color: #3b82f6;
    }
    .loading {
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
  </style>
</head>
<body class="bg-gray-100 min-h-screen">
  <div id="app">
    <!-- Header -->
    <header class="bg-white shadow-sm border-b">
      <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-gray-800">Agent Analytics Dashboard</h1>
        <div class="flex items-center space-x-4">
          <select id="timeRange" class="border rounded-lg px-3 py-2 text-sm">
            <option value="1">Last 1 hour</option>
            <option value="6">Last 6 hours</option>
            <option value="24" selected>Last 24 hours</option>
            <option value="72">Last 3 days</option>
            <option value="168">Last 7 days</option>
          </select>
          <button onclick="refreshData()" class="bg-blue-500 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-600">
            Refresh
          </button>
        </div>
      </div>
    </header>

    <!-- Navigation Tabs -->
    <nav class="bg-white border-b">
      <div class="max-w-7xl mx-auto px-4">
        <div class="flex space-x-8">
          <button onclick="showTab('overview')" class="tab-btn py-4 px-2 text-sm font-medium tab-active" data-tab="overview">
            Overview
          </button>
          <button onclick="showTab('sessions')" class="tab-btn py-4 px-2 text-sm font-medium text-gray-500 hover:text-gray-700" data-tab="sessions">
            Sessions
          </button>
          <button onclick="showTab('tools')" class="tab-btn py-4 px-2 text-sm font-medium text-gray-500 hover:text-gray-700" data-tab="tools">
            Tools
          </button>
          <button onclick="showTab('llm')" class="tab-btn py-4 px-2 text-sm font-medium text-gray-500 hover:text-gray-700" data-tab="llm">
            LLM Performance
          </button>
          <button onclick="showTab('errors')" class="tab-btn py-4 px-2 text-sm font-medium text-gray-500 hover:text-gray-700" data-tab="errors">
            Errors
          </button>
          <button onclick="showTab('latency')" class="tab-btn py-4 px-2 text-sm font-medium text-gray-500 hover:text-gray-700" data-tab="latency">
            Latency Analysis
          </button>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 py-6">
      <!-- Overview Tab -->
      <div id="tab-overview" class="tab-content">
        <!-- Metric Cards -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Total Sessions</div>
            <div id="metric-sessions" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Total Events</div>
            <div id="metric-events" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Error Rate</div>
            <div id="metric-error-rate" class="text-2xl font-bold loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Avg Latency</div>
            <div id="metric-latency" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Total Tokens</div>
            <div id="metric-tokens" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Unique Users</div>
            <div id="metric-users" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Unique Agents</div>
            <div id="metric-agents" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Invocations</div>
            <div id="metric-invocations" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
        </div>

        <!-- Charts -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div class="bg-white rounded-lg shadow p-4">
            <h3 class="text-lg font-semibold mb-4">Event Types Distribution</h3>
            <canvas id="eventTypesChart" height="250"></canvas>
          </div>
          <div class="bg-white rounded-lg shadow p-4">
            <h3 class="text-lg font-semibold mb-4">Events Timeline</h3>
            <canvas id="timelineChart" height="250"></canvas>
          </div>
        </div>
      </div>

      <!-- Sessions Tab -->
      <div id="tab-sessions" class="tab-content hidden">
        <div class="bg-white rounded-lg shadow">
          <div class="p-4 border-b flex justify-between items-center">
            <h3 class="text-lg font-semibold">Recent Sessions</h3>
            <select id="sessionFilter" onchange="loadSessions()" class="border rounded px-2 py-1 text-sm">
              <option value="">All Sessions</option>
              <option value="ERROR">With Errors</option>
              <option value="OK">Successful</option>
            </select>
          </div>
          <div id="sessions-list" class="divide-y">
            <div class="p-4 text-center text-gray-500 loading">Loading sessions...</div>
          </div>
        </div>

        <!-- Session Detail Modal -->
        <div id="session-modal" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div class="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] overflow-hidden">
            <div class="p-4 border-b flex justify-between items-center">
              <h3 class="text-lg font-semibold">Session Events</h3>
              <button onclick="closeSessionModal()" class="text-gray-500 hover:text-gray-700">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
              </button>
            </div>
            <div id="session-events" class="p-4 overflow-auto max-h-[60vh]">
              <!-- Events will be loaded here -->
            </div>
          </div>
        </div>
      </div>

      <!-- Tools Tab -->
      <div id="tab-tools" class="tab-content hidden">
        <div class="bg-white rounded-lg shadow">
          <div class="p-4 border-b">
            <h3 class="text-lg font-semibold">Tool Performance</h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full">
              <thead class="bg-gray-50">
                <tr>
                  <th class="px-4 py-3 text-left text-sm font-medium text-gray-500">Tool Name</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Invocations</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Success</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Errors</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Success Rate</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Avg Latency</th>
                </tr>
              </thead>
              <tbody id="tools-table" class="divide-y">
                <tr><td colspan="6" class="p-4 text-center text-gray-500 loading">Loading tools...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- LLM Performance Tab -->
      <div id="tab-llm" class="tab-content hidden">
        <!-- LLM Metrics -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Total Requests</div>
            <div id="llm-requests" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Avg Latency (ms)</div>
            <div id="llm-latency" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Avg TTFT (ms)</div>
            <div id="llm-ttft" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Error Count</div>
            <div id="llm-errors" class="text-2xl font-bold loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Prompt Tokens</div>
            <div id="llm-prompt-tokens" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Completion Tokens</div>
            <div id="llm-completion-tokens" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Total Tokens</div>
            <div id="llm-total-tokens" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Avg Tokens/Request</div>
            <div id="llm-avg-tokens" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
        </div>

        <!-- LLM Latency Chart -->
        <div class="bg-white rounded-lg shadow p-4">
          <h3 class="text-lg font-semibold mb-4">LLM Latency Over Time</h3>
          <canvas id="llmLatencyChart" height="200"></canvas>
        </div>
      </div>

      <!-- Errors Tab -->
      <div id="tab-errors" class="tab-content hidden">
        <div class="bg-white rounded-lg shadow">
          <div class="p-4 border-b">
            <h3 class="text-lg font-semibold">Error Summary</h3>
          </div>
          <div id="errors-list" class="divide-y">
            <div class="p-4 text-center text-gray-500 loading">Loading errors...</div>
          </div>
        </div>
      </div>

      <!-- Latency Analysis Tab -->
      <div id="tab-latency" class="tab-content hidden">
        <!-- Latency Percentiles -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Median (P50)</div>
            <div id="latency-p50" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">P90</div>
            <div id="latency-p90" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">P95</div>
            <div id="latency-p95" class="text-2xl font-bold text-orange-500 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">P99</div>
            <div id="latency-p99" class="text-2xl font-bold text-red-500 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Average</div>
            <div id="latency-avg" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Min</div>
            <div id="latency-min" class="text-2xl font-bold text-green-500 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Max</div>
            <div id="latency-max" class="text-2xl font-bold text-red-500 loading">-</div>
          </div>
          <div class="metric-card bg-white rounded-lg shadow p-4">
            <div class="text-sm text-gray-500">Std Dev</div>
            <div id="latency-stddev" class="text-2xl font-bold text-gray-800 loading">-</div>
          </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <!-- Latency Distribution -->
          <div class="bg-white rounded-lg shadow p-4">
            <h3 class="text-lg font-semibold mb-4">Latency Distribution</h3>
            <canvas id="latencyDistChart" height="250"></canvas>
          </div>
          <!-- Latency Timeline -->
          <div class="bg-white rounded-lg shadow p-4">
            <div class="flex justify-between items-center mb-4">
              <h3 class="text-lg font-semibold">Latency Over Time</h3>
              <select id="latencyMetricSelect" onchange="loadLatencyTimeline()" class="border rounded px-2 py-1 text-sm">
                <option value="avg">Average</option>
                <option value="p50">Median (P50)</option>
                <option value="p90">P90</option>
                <option value="p95">P95</option>
                <option value="p99">P99</option>
              </select>
            </div>
            <canvas id="latencyTimelineChart" height="250"></canvas>
          </div>
        </div>

        <!-- Latency by Event Type -->
        <div class="bg-white rounded-lg shadow mb-6">
          <div class="p-4 border-b">
            <h3 class="text-lg font-semibold">Latency by Event Type</h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full">
              <thead class="bg-gray-50">
                <tr>
                  <th class="px-4 py-3 text-left text-sm font-medium text-gray-500">Event Type</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Count</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Avg (ms)</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">P50 (ms)</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">P90 (ms)</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">P95 (ms)</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Min (ms)</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Max (ms)</th>
                </tr>
              </thead>
              <tbody id="latency-by-event-table" class="divide-y">
                <tr><td colspan="8" class="p-4 text-center text-gray-500 loading">Loading...</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Latency by Agent -->
        <div class="bg-white rounded-lg shadow mb-6">
          <div class="p-4 border-b">
            <h3 class="text-lg font-semibold">Latency by Agent</h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full">
              <thead class="bg-gray-50">
                <tr>
                  <th class="px-4 py-3 text-left text-sm font-medium text-gray-500">Agent</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Count</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Avg (ms)</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">P50 (ms)</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">P90 (ms)</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">P95 (ms)</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Min (ms)</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Max (ms)</th>
                </tr>
              </thead>
              <tbody id="latency-by-agent-table" class="divide-y">
                <tr><td colspan="8" class="p-4 text-center text-gray-500 loading">Loading...</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Slowest Requests -->
        <div class="bg-white rounded-lg shadow">
          <div class="p-4 border-b">
            <h3 class="text-lg font-semibold">Slowest Requests</h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full">
              <thead class="bg-gray-50">
                <tr>
                  <th class="px-4 py-3 text-left text-sm font-medium text-gray-500">Timestamp</th>
                  <th class="px-4 py-3 text-left text-sm font-medium text-gray-500">Event Type</th>
                  <th class="px-4 py-3 text-left text-sm font-medium text-gray-500">Agent</th>
                  <th class="px-4 py-3 text-right text-sm font-medium text-gray-500">Latency (ms)</th>
                  <th class="px-4 py-3 text-left text-sm font-medium text-gray-500">Preview</th>
                </tr>
              </thead>
              <tbody id="slowest-requests-table" class="divide-y">
                <tr><td colspan="5" class="p-4 text-center text-gray-500 loading">Loading...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </main>
  </div>

  <script>
    // Global state
    let charts = {};
    let currentTab = 'overview';

    // Initialize
    document.addEventListener('DOMContentLoaded', () => {
      refreshData();
    });

    // Tab switching
    function showTab(tab) {
      currentTab = tab;
      document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
      document.getElementById(`tab-${tab}`).classList.remove('hidden');
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('tab-active');
        btn.classList.add('text-gray-500');
      });
      document.querySelector(`[data-tab="${tab}"]`).classList.add('tab-active');
      document.querySelector(`[data-tab="${tab}"]`).classList.remove('text-gray-500');
    }

    // Get selected time range
    function getHours() {
      return parseInt(document.getElementById('timeRange').value);
    }

    // Refresh all data
    async function refreshData() {
      const hours = getHours();
      await Promise.all([
        loadOverview(hours),
        loadEventTypes(hours),
        loadTimeline(hours),
        loadSessions(hours),
        loadTools(hours),
        loadLLMMetrics(hours),
        loadLLMLatency(hours),
        loadErrors(hours),
        loadLatencyPercentiles(hours),
        loadLatencyDistribution(hours),
        loadLatencyTimeline(),
        loadLatencyByEventType(hours),
        loadLatencyByAgent(hours),
        loadSlowestRequests(hours),
      ]);
    }

    // Format numbers
    function formatNumber(n) {
      if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
      if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
      return n.toString();
    }

    // Load overview metrics
    async function loadOverview(hours) {
      try {
        const resp = await fetch(`/api/overview?hours=${hours}`);
        const data = await resp.json();

        document.getElementById('metric-sessions').textContent = formatNumber(data.total_sessions);
        document.getElementById('metric-events').textContent = formatNumber(data.total_events);
        document.getElementById('metric-invocations').textContent = formatNumber(data.total_invocations);
        document.getElementById('metric-tokens').textContent = formatNumber(data.total_tokens);
        document.getElementById('metric-users').textContent = formatNumber(data.unique_users);
        document.getElementById('metric-agents').textContent = formatNumber(data.unique_agents);

        const errorRateEl = document.getElementById('metric-error-rate');
        errorRateEl.textContent = data.error_rate.toFixed(2) + '%';
        errorRateEl.className = `text-2xl font-bold ${data.error_rate > 5 ? 'status-error' : 'status-ok'}`;

        document.getElementById('metric-latency').textContent = data.avg_latency_ms.toFixed(0) + 'ms';

        document.querySelectorAll('.loading').forEach(el => el.classList.remove('loading'));
      } catch (e) {
        console.error('Failed to load overview:', e);
      }
    }

    // Load event types chart
    async function loadEventTypes(hours) {
      try {
        const resp = await fetch(`/api/events/types?hours=${hours}`);
        const data = await resp.json();

        const ctx = document.getElementById('eventTypesChart').getContext('2d');
        if (charts.eventTypes) charts.eventTypes.destroy();

        charts.eventTypes = new Chart(ctx, {
          type: 'doughnut',
          data: {
            labels: data.map(d => d.event_type),
            datasets: [{
              data: data.map(d => d.count),
              backgroundColor: [
                '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
                '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#6366f1'
              ]
            }]
          },
          options: {
            responsive: true,
            plugins: {
              legend: { position: 'right' }
            }
          }
        });
      } catch (e) {
        console.error('Failed to load event types:', e);
      }
    }

    // Load timeline chart
    async function loadTimeline(hours) {
      try {
        const resp = await fetch(`/api/events/timeline?hours=${hours}`);
        const data = await resp.json();

        const ctx = document.getElementById('timelineChart').getContext('2d');
        if (charts.timeline) charts.timeline.destroy();

        charts.timeline = new Chart(ctx, {
          type: 'line',
          data: {
            labels: data.map(d => new Date(d.timestamp)),
            datasets: [{
              label: 'Events',
              data: data.map(d => d.value),
              borderColor: '#3b82f6',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              fill: true,
              tension: 0.4
            }]
          },
          options: {
            responsive: true,
            scales: {
              x: {
                type: 'time',
                time: { unit: 'hour' }
              }
            },
            plugins: {
              legend: { display: false }
            }
          }
        });
      } catch (e) {
        console.error('Failed to load timeline:', e);
      }
    }

    // Load sessions
    async function loadSessions() {
      const hours = getHours();
      const filter = document.getElementById('sessionFilter')?.value || '';
      try {
        const url = `/api/sessions?hours=${hours}&limit=50${filter ? `&status_filter=${filter}` : ''}`;
        const resp = await fetch(url);
        const data = await resp.json();

        const container = document.getElementById('sessions-list');
        if (data.length === 0) {
          container.innerHTML = '<div class="p-4 text-center text-gray-500">No sessions found</div>';
          return;
        }

        container.innerHTML = data.map(s => `
          <div class="p-4 event-row cursor-pointer" onclick="showSessionEvents('${s.session_id}')">
            <div class="flex justify-between items-start">
              <div>
                <div class="font-medium text-gray-800">${s.session_id.substring(0, 20)}...</div>
                <div class="text-sm text-gray-500">Agent: ${s.agent} | User: ${s.user_id || 'N/A'}</div>
                <div class="text-xs text-gray-400">${new Date(s.start_time).toLocaleString()}</div>
              </div>
              <div class="text-right">
                <div class="text-sm ${s.status === 'ERROR' ? 'status-error' : 'status-ok'}">${s.status}</div>
                <div class="text-xs text-gray-500">${s.event_count} events</div>
                ${s.error_count > 0 ? `<div class="text-xs text-red-500">${s.error_count} errors</div>` : ''}
              </div>
            </div>
          </div>
        `).join('');
      } catch (e) {
        console.error('Failed to load sessions:', e);
      }
    }

    // Show session events modal
    async function showSessionEvents(sessionId) {
      document.getElementById('session-modal').classList.remove('hidden');
      const container = document.getElementById('session-events');
      container.innerHTML = '<div class="text-center text-gray-500">Loading events...</div>';

      try {
        const resp = await fetch(`/api/sessions/${sessionId}/events`);
        const events = await resp.json();

        container.innerHTML = events.map(e => `
          <div class="mb-4 p-3 border rounded ${e.status === 'ERROR' ? 'border-red-200 bg-red-50' : 'border-gray-200'}">
            <div class="flex justify-between items-start mb-2">
              <span class="px-2 py-1 text-xs rounded ${getEventBadgeClass(e.event_type)}">${e.event_type}</span>
              <span class="text-xs text-gray-500">${new Date(e.timestamp).toLocaleTimeString()}</span>
            </div>
            ${e.latency_ms?.total_ms ? `<div class="text-xs text-gray-500 mb-1">Latency: ${e.latency_ms.total_ms}ms</div>` : ''}
            ${e.error_message ? `<div class="text-sm text-red-600 mb-1">${e.error_message}</div>` : ''}
            ${e.content ? `<pre class="text-xs bg-gray-100 p-2 rounded overflow-auto max-h-40">${JSON.stringify(e.content, null, 2)}</pre>` : ''}
          </div>
        `).join('');
      } catch (e) {
        container.innerHTML = '<div class="text-center text-red-500">Failed to load events</div>';
      }
    }

    function closeSessionModal() {
      document.getElementById('session-modal').classList.add('hidden');
    }

    function getEventBadgeClass(eventType) {
      const classes = {
        'LLM_REQUEST': 'bg-blue-100 text-blue-800',
        'LLM_RESPONSE': 'bg-green-100 text-green-800',
        'LLM_ERROR': 'bg-red-100 text-red-800',
        'TOOL_STARTING': 'bg-purple-100 text-purple-800',
        'TOOL_COMPLETED': 'bg-indigo-100 text-indigo-800',
        'TOOL_ERROR': 'bg-red-100 text-red-800',
        'USER_MESSAGE_RECEIVED': 'bg-yellow-100 text-yellow-800',
        'AGENT_STARTING': 'bg-cyan-100 text-cyan-800',
        'AGENT_COMPLETED': 'bg-teal-100 text-teal-800',
      };
      return classes[eventType] || 'bg-gray-100 text-gray-800';
    }

    // Load tools
    async function loadTools(hours) {
      try {
        const resp = await fetch(`/api/tools/metrics?hours=${hours}`);
        const data = await resp.json();

        const container = document.getElementById('tools-table');
        if (data.length === 0) {
          container.innerHTML = '<tr><td colspan="6" class="p-4 text-center text-gray-500">No tool data</td></tr>';
          return;
        }

        container.innerHTML = data.map(t => `
          <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm font-medium text-gray-900">${t.tool_name}</td>
            <td class="px-4 py-3 text-sm text-right text-gray-500">${t.invocations}</td>
            <td class="px-4 py-3 text-sm text-right status-ok">${t.success_count}</td>
            <td class="px-4 py-3 text-sm text-right ${t.error_count > 0 ? 'status-error' : 'text-gray-500'}">${t.error_count}</td>
            <td class="px-4 py-3 text-sm text-right ${t.success_rate >= 95 ? 'status-ok' : t.success_rate >= 80 ? 'text-yellow-500' : 'status-error'}">${t.success_rate.toFixed(1)}%</td>
            <td class="px-4 py-3 text-sm text-right text-gray-500">${t.avg_latency_ms.toFixed(0)}ms</td>
          </tr>
        `).join('');
      } catch (e) {
        console.error('Failed to load tools:', e);
      }
    }

    // Load LLM metrics
    async function loadLLMMetrics(hours) {
      try {
        const resp = await fetch(`/api/llm/metrics?hours=${hours}`);
        const data = await resp.json();

        document.getElementById('llm-requests').textContent = formatNumber(data.total_requests);
        document.getElementById('llm-latency').textContent = data.avg_latency_ms.toFixed(0);
        document.getElementById('llm-ttft').textContent = data.avg_time_to_first_token_ms?.toFixed(0) || 'N/A';

        const errEl = document.getElementById('llm-errors');
        errEl.textContent = data.error_count;
        errEl.className = `text-2xl font-bold ${data.error_count > 0 ? 'status-error' : 'status-ok'}`;

        document.getElementById('llm-prompt-tokens').textContent = formatNumber(data.total_prompt_tokens);
        document.getElementById('llm-completion-tokens').textContent = formatNumber(data.total_completion_tokens);
        document.getElementById('llm-total-tokens').textContent = formatNumber(data.total_tokens);
        document.getElementById('llm-avg-tokens').textContent = data.avg_tokens_per_request.toFixed(0);
      } catch (e) {
        console.error('Failed to load LLM metrics:', e);
      }
    }

    // Load LLM latency timeline
    async function loadLLMLatency(hours) {
      try {
        const resp = await fetch(`/api/llm/latency-timeline?hours=${hours}`);
        const data = await resp.json();

        const ctx = document.getElementById('llmLatencyChart').getContext('2d');
        if (charts.llmLatency) charts.llmLatency.destroy();

        charts.llmLatency = new Chart(ctx, {
          type: 'line',
          data: {
            labels: data.map(d => new Date(d.timestamp)),
            datasets: [{
              label: 'Avg Latency (ms)',
              data: data.map(d => d.value),
              borderColor: '#8b5cf6',
              backgroundColor: 'rgba(139, 92, 246, 0.1)',
              fill: true,
              tension: 0.4
            }]
          },
          options: {
            responsive: true,
            scales: {
              x: {
                type: 'time',
                time: { unit: 'hour' }
              },
              y: {
                beginAtZero: true,
                title: { display: true, text: 'Latency (ms)' }
              }
            }
          }
        });
      } catch (e) {
        console.error('Failed to load LLM latency:', e);
      }
    }

    // Load errors
    async function loadErrors(hours) {
      try {
        const resp = await fetch(`/api/errors?hours=${hours}`);
        const data = await resp.json();

        const container = document.getElementById('errors-list');
        if (data.length === 0) {
          container.innerHTML = '<div class="p-4 text-center text-gray-500">No errors found</div>';
          return;
        }

        container.innerHTML = data.map(e => `
          <div class="p-4 border-b hover:bg-gray-50">
            <div class="flex justify-between items-start mb-2">
              <span class="px-2 py-1 text-xs rounded bg-red-100 text-red-800">${e.event_type}</span>
              <span class="text-xs text-gray-500">${e.count} occurrences</span>
            </div>
            <div class="text-sm text-red-600 mb-2 font-mono">${e.error_message}</div>
            <div class="text-xs text-gray-500">
              Last seen: ${new Date(e.last_occurrence).toLocaleString()} |
              Affected sessions: ${e.affected_sessions}
            </div>
          </div>
        `).join('');
      } catch (e) {
        console.error('Failed to load errors:', e);
      }
    }

    // Load latency percentiles
    async function loadLatencyPercentiles(hours) {
      try {
        const resp = await fetch(`/api/latency/percentiles?hours=${hours}`);
        const data = await resp.json();

        document.getElementById('latency-p50').textContent = data.median_ms.toFixed(0) + ' ms';
        document.getElementById('latency-p90').textContent = data.p90_ms.toFixed(0) + ' ms';
        document.getElementById('latency-p95').textContent = data.p95_ms.toFixed(0) + ' ms';
        document.getElementById('latency-p99').textContent = data.p99_ms.toFixed(0) + ' ms';
        document.getElementById('latency-avg').textContent = data.avg_ms.toFixed(0) + ' ms';
        document.getElementById('latency-min').textContent = data.min_ms.toFixed(0) + ' ms';
        document.getElementById('latency-max').textContent = data.max_ms.toFixed(0) + ' ms';
        document.getElementById('latency-stddev').textContent = data.std_dev_ms.toFixed(0) + ' ms';
      } catch (e) {
        console.error('Failed to load latency percentiles:', e);
      }
    }

    // Load latency distribution histogram
    async function loadLatencyDistribution(hours) {
      try {
        const resp = await fetch(`/api/latency/distribution?hours=${hours}&bucket_size_ms=2000`);
        const data = await resp.json();

        const ctx = document.getElementById('latencyDistChart').getContext('2d');
        if (charts.latencyDist) charts.latencyDist.destroy();

        charts.latencyDist = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: data.map(d => `${(d.bucket_start_ms/1000).toFixed(0)}-${(d.bucket_end_ms/1000).toFixed(0)}s`),
            datasets: [{
              label: 'Requests',
              data: data.map(d => d.count),
              backgroundColor: 'rgba(59, 130, 246, 0.7)',
              borderColor: '#3b82f6',
              borderWidth: 1
            }]
          },
          options: {
            responsive: true,
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: (ctx) => `${ctx.raw} requests (${data[ctx.dataIndex].percentage.toFixed(1)}%)`
                }
              }
            },
            scales: {
              x: { title: { display: true, text: 'Latency Range' } },
              y: { title: { display: true, text: 'Count' }, beginAtZero: true }
            }
          }
        });
      } catch (e) {
        console.error('Failed to load latency distribution:', e);
      }
    }

    // Load latency timeline
    async function loadLatencyTimeline() {
      const hours = getHours();
      const metric = document.getElementById('latencyMetricSelect')?.value || 'avg';
      try {
        const resp = await fetch(`/api/latency/timeline?hours=${hours}&metric=${metric}`);
        const data = await resp.json();

        const ctx = document.getElementById('latencyTimelineChart').getContext('2d');
        if (charts.latencyTimeline) charts.latencyTimeline.destroy();

        const colors = {
          'avg': '#3b82f6',
          'p50': '#10b981',
          'p90': '#f59e0b',
          'p95': '#f97316',
          'p99': '#ef4444'
        };

        charts.latencyTimeline = new Chart(ctx, {
          type: 'line',
          data: {
            labels: data.map(d => new Date(d.timestamp)),
            datasets: [{
              label: data[0]?.label || 'Latency (ms)',
              data: data.map(d => d.value),
              borderColor: colors[metric] || '#3b82f6',
              backgroundColor: `${colors[metric]}20` || 'rgba(59, 130, 246, 0.1)',
              fill: true,
              tension: 0.4
            }]
          },
          options: {
            responsive: true,
            scales: {
              x: { type: 'time', time: { unit: 'hour' } },
              y: { title: { display: true, text: 'Latency (ms)' }, beginAtZero: true }
            }
          }
        });
      } catch (e) {
        console.error('Failed to load latency timeline:', e);
      }
    }

    // Load latency by event type
    async function loadLatencyByEventType(hours) {
      try {
        const resp = await fetch(`/api/latency/by-event-type?hours=${hours}`);
        const data = await resp.json();

        const container = document.getElementById('latency-by-event-table');
        if (data.length === 0) {
          container.innerHTML = '<tr><td colspan="8" class="p-4 text-center text-gray-500">No data</td></tr>';
          return;
        }

        container.innerHTML = data.map(d => `
          <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm font-medium text-gray-900">${d.category}</td>
            <td class="px-4 py-3 text-sm text-right text-gray-500">${d.count}</td>
            <td class="px-4 py-3 text-sm text-right text-gray-900 font-medium">${d.avg_ms.toFixed(0)}</td>
            <td class="px-4 py-3 text-sm text-right text-gray-500">${d.p50_ms.toFixed(0)}</td>
            <td class="px-4 py-3 text-sm text-right text-orange-500">${d.p90_ms.toFixed(0)}</td>
            <td class="px-4 py-3 text-sm text-right text-orange-600">${d.p95_ms.toFixed(0)}</td>
            <td class="px-4 py-3 text-sm text-right text-green-500">${d.min_ms.toFixed(0)}</td>
            <td class="px-4 py-3 text-sm text-right text-red-500">${d.max_ms.toFixed(0)}</td>
          </tr>
        `).join('');
      } catch (e) {
        console.error('Failed to load latency by event type:', e);
      }
    }

    // Load latency by agent
    async function loadLatencyByAgent(hours) {
      try {
        const resp = await fetch(`/api/latency/by-agent?hours=${hours}`);
        const data = await resp.json();

        const container = document.getElementById('latency-by-agent-table');
        if (data.length === 0) {
          container.innerHTML = '<tr><td colspan="8" class="p-4 text-center text-gray-500">No data</td></tr>';
          return;
        }

        container.innerHTML = data.map(d => `
          <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm font-medium text-gray-900">${d.category || 'Unknown'}</td>
            <td class="px-4 py-3 text-sm text-right text-gray-500">${d.count}</td>
            <td class="px-4 py-3 text-sm text-right text-gray-900 font-medium">${d.avg_ms.toFixed(0)}</td>
            <td class="px-4 py-3 text-sm text-right text-gray-500">${d.p50_ms.toFixed(0)}</td>
            <td class="px-4 py-3 text-sm text-right text-orange-500">${d.p90_ms.toFixed(0)}</td>
            <td class="px-4 py-3 text-sm text-right text-orange-600">${d.p95_ms.toFixed(0)}</td>
            <td class="px-4 py-3 text-sm text-right text-green-500">${d.min_ms.toFixed(0)}</td>
            <td class="px-4 py-3 text-sm text-right text-red-500">${d.max_ms.toFixed(0)}</td>
          </tr>
        `).join('');
      } catch (e) {
        console.error('Failed to load latency by agent:', e);
      }
    }

    // Load slowest requests
    async function loadSlowestRequests(hours) {
      try {
        const resp = await fetch(`/api/latency/slowest?hours=${hours}&limit=15`);
        const data = await resp.json();

        const container = document.getElementById('slowest-requests-table');
        if (data.length === 0) {
          container.innerHTML = '<tr><td colspan="5" class="p-4 text-center text-gray-500">No data</td></tr>';
          return;
        }

        container.innerHTML = data.map(d => `
          <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-xs text-gray-500">${new Date(d.timestamp).toLocaleString()}</td>
            <td class="px-4 py-3 text-sm">
              <span class="px-2 py-1 text-xs rounded ${getEventBadgeClass(d.event_type)}">${d.event_type || 'N/A'}</span>
            </td>
            <td class="px-4 py-3 text-sm text-gray-900">${d.agent || 'N/A'}</td>
            <td class="px-4 py-3 text-sm text-right font-bold ${d.latency_ms > 10000 ? 'text-red-600' : d.latency_ms > 5000 ? 'text-orange-500' : 'text-gray-900'}">${d.latency_ms.toFixed(0)} ms</td>
            <td class="px-4 py-3 text-xs text-gray-500 truncate max-w-xs" title="${d.content_preview || ''}">${d.content_preview || '-'}</td>
          </tr>
        `).join('');
      } catch (e) {
        console.error('Failed to load slowest requests:', e);
      }
    }

    // Auto-refresh every 30 seconds
    setInterval(refreshData, 30000);
  </script>
</body>
</html>"""


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8080)
