-- Cloud Trace Export to BigQuery - Analytics Queries
-- =====================================================
-- These queries work with Cloud Trace data exported to BigQuery via
-- the export_traces_to_bigquery.py ETL script.
--
-- IMPORTANT: This requires MANUAL ETL - Cloud Trace does NOT have
-- native BigQuery export like the BigQuery Agent Analytics Plugin.
--
-- Project: test-project-0728-467323
-- Dataset: cloud_trace_export
-- Table: agent_traces
-- =====================================================


-- ============================================================================
-- 1. TOKEN USAGE ANALYSIS
-- ============================================================================
-- STATUS: AVAILABLE (via manual ETL)
-- NOTE: Requires parsing from span labels, not native structured fields

-- Token usage summary
SELECT
  COUNT(*) AS total_llm_calls,
  SUM(input_tokens) AS total_input_tokens,
  SUM(output_tokens) AS total_output_tokens,
  SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) AS total_tokens,
  ROUND(AVG(input_tokens + output_tokens), 1) AS avg_tokens_per_call
FROM `test-project-0728-467323.cloud_trace_export.agent_traces`
WHERE input_tokens IS NOT NULL;

-- RESULT:
-- +-----------------+--------------------+---------------------+--------------+---------------------+
-- | total_llm_calls | total_input_tokens | total_output_tokens | total_tokens | avg_tokens_per_call |
-- +-----------------+--------------------+---------------------+--------------+---------------------+
-- |               2 |                590 |                  28 |          618 |               309.0 |
-- +-----------------+--------------------+---------------------+--------------+---------------------+


-- ============================================================================
-- 2. TOOL ANALYTICS
-- ============================================================================
-- STATUS: AVAILABLE (via manual ETL)
-- NOTE: Tool args/results available but must be parsed from span labels

-- Tool usage summary
SELECT
  tool_name,
  COUNT(*) AS call_count,
  ROUND(AVG(duration_ms), 2) AS avg_duration_ms
FROM `test-project-0728-467323.cloud_trace_export.agent_traces`
WHERE tool_name IS NOT NULL
GROUP BY tool_name
ORDER BY call_count DESC;

-- Tool details with arguments and responses
SELECT
  tool_name,
  tool_args,
  tool_response,
  duration_ms
FROM `test-project-0728-467323.cloud_trace_export.agent_traces`
WHERE tool_name IS NOT NULL;


-- ============================================================================
-- 3. SESSION ANALYTICS
-- ============================================================================
-- STATUS: PARTIAL - Session ID available but not user_id

SELECT
  session_id,
  COUNT(*) AS span_count,
  COUNTIF(operation_name = 'execute_tool') AS tool_calls,
  SUM(input_tokens) AS total_input_tokens,
  SUM(output_tokens) AS total_output_tokens,
  ROUND(SUM(duration_ms), 0) AS total_duration_ms
FROM `test-project-0728-467323.cloud_trace_export.agent_traces`
WHERE session_id IS NOT NULL
GROUP BY session_id;


-- ============================================================================
-- 4. LATENCY ANALYSIS
-- ============================================================================
-- STATUS: AVAILABLE

-- Overall latency stats
SELECT
  span_name,
  COUNT(*) AS call_count,
  ROUND(AVG(duration_ms), 2) AS avg_ms,
  ROUND(MIN(duration_ms), 2) AS min_ms,
  ROUND(MAX(duration_ms), 2) AS max_ms,
  ROUND(STDDEV(duration_ms), 2) AS stddev_ms
FROM `test-project-0728-467323.cloud_trace_export.agent_traces`
WHERE duration_ms IS NOT NULL
GROUP BY span_name
ORDER BY avg_ms DESC;


-- ============================================================================
-- 5. AI-POWERED ANALYTICS (with Gemini)
-- ============================================================================
-- STATUS: AVAILABLE after setting up Gemini model connection
-- NOTE: This is what makes BigQuery powerful - same capability as BQ plugin

-- 5a. Analyze tool responses with AI
SELECT
  tool_name,
  tool_args,
  tool_response,
  AI.GENERATE_TEXT(
    MODEL `test-project-0728-467323.cloud_trace_export.gemini_model`,
    CONCAT(
      'Analyze this tool execution. What was requested and what was returned? ',
      'Tool: ', tool_name,
      ' Args: ', COALESCE(tool_args, 'N/A'),
      ' Response: ', COALESCE(tool_response, 'N/A')
    ),
    STRUCT(0.2 AS temperature, 200 AS max_output_tokens)
  ).ml_generate_text_llm_result AS ai_analysis
FROM `test-project-0728-467323.cloud_trace_export.agent_traces`
WHERE tool_name IS NOT NULL
LIMIT 10;


-- ============================================================================
-- 6. COMPARISON: WHAT'S MISSING vs BIGQUERY PLUGIN
-- ============================================================================
/*
+---------------------------+-------------------+------------------------+
| Capability                | Trace Export      | BQ Analytics Plugin    |
+---------------------------+-------------------+------------------------+
| Token usage               | YES (after ETL)   | YES (automatic)        |
| Tool args/results         | YES (after ETL)   | YES (automatic)        |
| Session tracking          | PARTIAL (no user) | YES (session + user)   |
| Latency metrics           | YES (after ETL)   | YES (automatic)        |
| Event types               | NO                | YES (10+ types)        |
| User ID                   | NO                | YES                    |
| Invocation ID             | NO                | YES                    |
| Multimodal content        | NO                | YES + GCS offload      |
| Real-time                 | NO (manual ETL)   | YES (1-2 seconds)      |
| AI analytics              | YES (with setup)  | YES (with setup)       |
+---------------------------+-------------------+------------------------+

KEY DIFFERENCES:
1. BigQuery Plugin: Automatic, real-time, zero ETL
2. Cloud Trace Export: Manual ETL script, must run periodically, ~30min setup
*/


-- ============================================================================
-- 7. FULL DATA EXPLORATION
-- ============================================================================

-- View all exported data
SELECT *
FROM `test-project-0728-467323.cloud_trace_export.agent_traces`
ORDER BY start_time DESC
LIMIT 20;

-- View raw labels (all span attributes)
SELECT
  span_name,
  labels
FROM `test-project-0728-467323.cloud_trace_export.agent_traces`
WHERE labels IS NOT NULL;
