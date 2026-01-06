-- BigQuery Agent Analytics - Sample Analysis Queries
-- ==================================================
-- These queries demonstrate analytics capabilities unique to the BQ approach.
-- Using project: test-project-0728-467323, dataset: agent_analytics_demo
--
-- JSON field paths discovered from actual data:
--   Token usage: $.usage.total, $.usage.prompt, $.usage.completion
--   Latency: $.total_ms, $.time_to_first_token_ms
--   Tool info: $.tool, $.result, $.args

-- ============================================================================
-- 1. TOKEN USAGE ANALYSIS
-- ============================================================================

-- 1a. Token usage per LLM response
SELECT
  timestamp,
  session_id,
  agent,
  CAST(JSON_EXTRACT_SCALAR(content, '$.usage.total') AS INT64) AS total_tokens,
  CAST(JSON_EXTRACT_SCALAR(content, '$.usage.prompt') AS INT64) AS prompt_tokens,
  CAST(JSON_EXTRACT_SCALAR(content, '$.usage.completion') AS INT64) AS response_tokens
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE event_type = 'LLM_RESPONSE'
ORDER BY timestamp DESC
LIMIT 100;

-- 1b. Aggregate token usage by agent (daily)
SELECT
  DATE(timestamp) AS date,
  agent,
  COUNT(*) AS llm_calls,
  SUM(CAST(JSON_EXTRACT_SCALAR(content, '$.usage.total') AS INT64)) AS total_tokens,
  ROUND(AVG(CAST(JSON_EXTRACT_SCALAR(content, '$.usage.total') AS INT64)), 1) AS avg_tokens_per_call,
  SUM(CAST(JSON_EXTRACT_SCALAR(content, '$.usage.prompt') AS INT64)) AS total_prompt_tokens,
  SUM(CAST(JSON_EXTRACT_SCALAR(content, '$.usage.completion') AS INT64)) AS total_response_tokens
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE event_type = 'LLM_RESPONSE'
GROUP BY date, agent
ORDER BY date DESC, total_tokens DESC;

-- 1c. Token usage percentiles (for capacity planning)
SELECT
  agent,
  APPROX_QUANTILES(
    CAST(JSON_EXTRACT_SCALAR(content, '$.usage.total') AS INT64),
    100
  )[OFFSET(50)] AS p50_tokens,
  APPROX_QUANTILES(
    CAST(JSON_EXTRACT_SCALAR(content, '$.usage.total') AS INT64),
    100
  )[OFFSET(95)] AS p95_tokens,
  APPROX_QUANTILES(
    CAST(JSON_EXTRACT_SCALAR(content, '$.usage.total') AS INT64),
    100
  )[OFFSET(99)] AS p99_tokens
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE event_type = 'LLM_RESPONSE'
GROUP BY agent;


-- ============================================================================
-- 2. TOOL FAILURE RATE ANALYSIS
-- ============================================================================

-- 2a. Tool failure rates (overall)
SELECT
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  COUNTIF(event_type = 'TOOL_COMPLETED') AS successful_calls,
  COUNTIF(event_type = 'TOOL_ERROR') AS failed_calls,
  COUNT(*) AS total_calls,
  ROUND(100.0 * COUNTIF(event_type = 'TOOL_ERROR') / NULLIF(COUNT(*), 0), 2) AS failure_rate_pct
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR')
GROUP BY tool_name
ORDER BY failure_rate_pct DESC;

-- 2b. Tool failure rates over time (hourly trend)
SELECT
  TIMESTAMP_TRUNC(timestamp, HOUR) AS hour,
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  COUNTIF(event_type = 'TOOL_COMPLETED') AS successes,
  COUNTIF(event_type = 'TOOL_ERROR') AS failures,
  ROUND(100.0 * COUNTIF(event_type = 'TOOL_ERROR') / NULLIF(COUNT(*), 0), 2) AS failure_rate_pct
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR')
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
GROUP BY hour, tool_name
ORDER BY hour DESC, tool_name;

-- 2c. Tool error details (for debugging)
SELECT
  timestamp,
  session_id,
  invocation_id,
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  error_message,
  JSON_EXTRACT_SCALAR(content, '$.args') AS tool_args
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE event_type = 'TOOL_ERROR'
ORDER BY timestamp DESC
LIMIT 50;

-- 2d. Tool latency analysis
SELECT
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  COUNT(*) AS call_count,
  ROUND(AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)), 1) AS avg_latency_ms,
  MIN(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS INT64)) AS min_latency_ms,
  MAX(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS INT64)) AS max_latency_ms,
  APPROX_QUANTILES(
    CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64),
    100
  )[OFFSET(95)] AS p95_latency_ms
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE event_type = 'TOOL_COMPLETED'
  AND latency_ms IS NOT NULL
GROUP BY tool_name
ORDER BY avg_latency_ms DESC;


-- ============================================================================
-- 3. MULTI-MODAL CONTENT ANALYSIS
-- ============================================================================

-- 3a. Count of multimodal content by type
SELECT
  part.mime_type,
  COUNT(*) AS count,
  COUNT(DISTINCT session_id) AS unique_sessions
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`,
UNNEST(content_parts) AS part
WHERE part.mime_type IS NOT NULL
GROUP BY part.mime_type
ORDER BY count DESC;

-- 3b. Image content analysis (with GCS references)
SELECT
  timestamp,
  session_id,
  event_type,
  part.mime_type,
  part.storage_mode,
  part.object_ref.uri AS gcs_uri,
  part.object_ref.size_bytes AS size_bytes
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`,
UNNEST(content_parts) AS part
WHERE part.mime_type LIKE 'image/%'
ORDER BY timestamp DESC
LIMIT 100;

-- 3c. Large content offloaded to GCS
SELECT
  DATE(timestamp) AS date,
  COUNT(*) AS offloaded_items,
  SUM(part.object_ref.size_bytes) AS total_bytes_offloaded,
  ROUND(SUM(part.object_ref.size_bytes) / 1024 / 1024, 2) AS total_mb_offloaded
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`,
UNNEST(content_parts) AS part
WHERE part.storage_mode = 'GCS_REFERENCE'
GROUP BY date
ORDER BY date DESC;


-- ============================================================================
-- 4. SESSION AND USER ANALYTICS
-- ============================================================================

-- 4a. Sessions per user
SELECT
  user_id,
  COUNT(DISTINCT session_id) AS session_count,
  COUNT(DISTINCT invocation_id) AS total_invocations,
  MIN(timestamp) AS first_activity,
  MAX(timestamp) AS last_activity
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
GROUP BY user_id
ORDER BY session_count DESC
LIMIT 100;

-- 4b. Session timeline (conversation flow)
SELECT
  timestamp,
  event_type,
  agent,
  CASE
    WHEN event_type = 'USER_MESSAGE_RECEIVED' THEN
      JSON_EXTRACT_SCALAR(content, '$.text')
    WHEN event_type LIKE 'TOOL_%' THEN
      JSON_EXTRACT_SCALAR(content, '$.tool')
    WHEN event_type = 'LLM_RESPONSE' THEN
      LEFT(JSON_EXTRACT_SCALAR(content, '$.text'), 100)
    ELSE NULL
  END AS summary
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE session_id = 'YOUR_SESSION_ID'
ORDER BY timestamp;

-- 4c. Average session duration
SELECT
  DATE(MIN(timestamp)) AS date,
  COUNT(DISTINCT session_id) AS sessions,
  AVG(TIMESTAMP_DIFF(
    MAX(timestamp),
    MIN(timestamp),
    SECOND
  )) AS avg_session_duration_sec
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
GROUP BY session_id
HAVING COUNT(*) > 1;


-- ============================================================================
-- 5. LLM PERFORMANCE ANALYSIS
-- ============================================================================

-- 5a. LLM latency analysis
SELECT
  agent,
  COUNT(*) AS llm_calls,
  AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)) AS avg_total_ms,
  AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.time_to_first_token_ms') AS FLOAT64)) AS avg_ttft_ms,
  APPROX_QUANTILES(
    CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64),
    100
  )[OFFSET(95)] AS p95_total_ms
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE event_type = 'LLM_RESPONSE'
  AND latency_ms IS NOT NULL
GROUP BY agent
ORDER BY avg_total_ms DESC;

-- 5b. LLM error analysis
SELECT
  DATE(timestamp) AS date,
  agent,
  COUNT(*) AS error_count,
  ARRAY_AGG(DISTINCT error_message LIMIT 5) AS sample_errors
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE event_type = 'LLM_ERROR'
GROUP BY date, agent
ORDER BY date DESC, error_count DESC;


-- ============================================================================
-- 6. COST ESTIMATION (based on token usage)
-- ============================================================================

-- Note: Adjust pricing based on your model and pricing tier
-- Example: Gemini 2.0 Flash pricing (approximate)
SELECT
  DATE(timestamp) AS date,
  agent,
  SUM(CAST(JSON_EXTRACT_SCALAR(content, '$.usage.prompt') AS INT64)) AS prompt_tokens,
  SUM(CAST(JSON_EXTRACT_SCALAR(content, '$.usage.completion') AS INT64)) AS response_tokens,
  -- Approximate cost calculation (adjust rates as needed)
  ROUND(
    SUM(CAST(JSON_EXTRACT_SCALAR(content, '$.usage.prompt') AS INT64)) * 0.000001 +
    SUM(CAST(JSON_EXTRACT_SCALAR(content, '$.usage.completion') AS INT64)) * 0.000004,
    4
  ) AS estimated_cost_usd
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE event_type = 'LLM_RESPONSE'
GROUP BY date, agent
ORDER BY date DESC;


-- ============================================================================
-- 7. OPENTELEMETRY TRACE CORRELATION (when using OTel bridge)
-- ============================================================================

-- 7a. Events with trace context
SELECT
  timestamp,
  event_type,
  trace_id,
  span_id,
  parent_span_id,
  agent,
  session_id
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
WHERE trace_id IS NOT NULL
ORDER BY timestamp DESC
LIMIT 100;

-- 7b. Reconstruct trace hierarchy
WITH trace_events AS (
  SELECT
    timestamp,
    event_type,
    trace_id,
    span_id,
    parent_span_id,
    agent,
    CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64) AS duration_ms
  FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
  WHERE trace_id = 'YOUR_TRACE_ID'
)
SELECT
  e1.event_type AS event,
  e1.span_id,
  e2.event_type AS parent_event,
  e1.duration_ms
FROM trace_events e1
LEFT JOIN trace_events e2
  ON e1.parent_span_id = e2.span_id
ORDER BY e1.timestamp;
