-- BigQuery AI-Powered Analytics for Agent Telemetry
-- ==================================================
-- These queries leverage BigQuery's AI functions (AI.GENERATE, AI.GENERATE_TEXT)
-- to perform advanced analytics that are IMPOSSIBLE with Cloud Logging.
--
-- Prerequisites:
--   1. Create a remote model connection to Vertex AI Gemini
--   2. Grant the connection service account Vertex AI User role
--
-- See: https://cloud.google.com/bigquery/docs/generative-ai-overview

-- ============================================================================
-- SETUP: Create Remote Model Connection (run once)
-- ============================================================================

-- Step 1: Create a Cloud resource connection
-- bq mk --connection --location=US --project_id=test-project-0728-467323 \
--        --connection_type=CLOUD_RESOURCE gemini_conn

-- Step 2: Create a remote model pointing to Gemini
CREATE OR REPLACE MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`
REMOTE WITH CONNECTION `test-project-0728-467323.us.gemini_conn`
OPTIONS (endpoint = 'gemini-2.0-flash');


-- ============================================================================
-- 1. LLM-AS-JUDGE: Customer-Facing Response Evaluation
-- ============================================================================
-- Automatically evaluate agent responses for quality, helpfulness, and accuracy

-- 1a. Evaluate response quality with structured scoring
SELECT
  session_id,
  timestamp,
  JSON_EXTRACT_SCALAR(content, '$.response') AS agent_response,
  ai_evaluation.ml_generate_text_llm_result AS evaluation_result
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`,
UNNEST([STRUCT(
  AI.GENERATE_TEXT(
    MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
    CONCAT(
      'Evaluate this agent response on a scale of 1-5 for each criterion. ',
      'Return JSON with keys: helpfulness, accuracy, clarity, completeness. ',
      'Agent response: ', JSON_EXTRACT_SCALAR(content, '$.response')
    ),
    STRUCT(0.2 AS temperature, 500 AS max_output_tokens)
  ) AS ai_evaluation
)])
WHERE event_type = 'LLM_RESPONSE'
  AND JSON_EXTRACT_SCALAR(content, '$.response') IS NOT NULL
ORDER BY timestamp DESC
LIMIT 20;

-- 1b. Batch evaluation with detailed feedback
SELECT
  session_id,
  user_id,
  JSON_EXTRACT_SCALAR(content, '$.response') AS response,
  result.*
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`,
LATERAL (
  SELECT AI.GENERATE_TEXT(
    MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
    CONCAT(
      'You are evaluating an AI agent response. Analyze and provide:\n',
      '1. Quality score (1-10)\n',
      '2. Is the response helpful? (yes/no)\n',
      '3. Any issues detected (hallucination, incomplete, off-topic)\n',
      '4. Suggested improvement\n\n',
      'Response to evaluate:\n',
      JSON_EXTRACT_SCALAR(content, '$.response')
    ),
    STRUCT(0.3 AS temperature)
  ) AS result
)
WHERE event_type = 'LLM_RESPONSE'
ORDER BY timestamp DESC
LIMIT 50;


-- ============================================================================
-- 2. JAILBREAK & SAFETY DETECTION
-- ============================================================================
-- Detect potential jailbreak attempts or safety violations in user inputs

-- 2a. Scan user messages for jailbreak attempts
SELECT
  timestamp,
  session_id,
  user_id,
  JSON_EXTRACT_SCALAR(content, '$.text') AS user_message,
  safety_check.ml_generate_text_llm_result AS safety_analysis
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`,
UNNEST([STRUCT(
  AI.GENERATE_TEXT(
    MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
    CONCAT(
      'Analyze this user message for potential jailbreak attempts or safety violations. ',
      'Return JSON with: {',
      '"is_jailbreak_attempt": boolean, ',
      '"risk_level": "none|low|medium|high", ',
      '"detected_patterns": [list of patterns], ',
      '"explanation": "brief explanation"}\n\n',
      'User message: ', JSON_EXTRACT_SCALAR(content, '$.text')
    ),
    STRUCT(0.1 AS temperature, 300 AS max_output_tokens)
  ) AS safety_check
)])
WHERE event_type = 'USER_MESSAGE_RECEIVED'
ORDER BY timestamp DESC
LIMIT 100;

-- 2b. Aggregate jailbreak detection results by user
WITH safety_analysis AS (
  SELECT
    user_id,
    session_id,
    JSON_EXTRACT_SCALAR(content, '$.text') AS message,
    AI.GENERATE_TEXT(
      MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
      CONCAT(
        'Is this a jailbreak attempt? Reply only "YES" or "NO": ',
        JSON_EXTRACT_SCALAR(content, '$.text')
      ),
      STRUCT(0.0 AS temperature, 10 AS max_output_tokens)
    ).ml_generate_text_llm_result AS is_jailbreak
  FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
  WHERE event_type = 'USER_MESSAGE_RECEIVED'
)
SELECT
  user_id,
  COUNT(*) AS total_messages,
  COUNTIF(UPPER(is_jailbreak) LIKE '%YES%') AS jailbreak_attempts,
  ROUND(100.0 * COUNTIF(UPPER(is_jailbreak) LIKE '%YES%') / COUNT(*), 2) AS jailbreak_rate_pct
FROM safety_analysis
GROUP BY user_id
HAVING jailbreak_attempts > 0
ORDER BY jailbreak_attempts DESC;


-- ============================================================================
-- 3. TOOL FAILURE ROOT CAUSE ANALYSIS
-- ============================================================================
-- Use LLM to analyze tool errors and identify root causes automatically

-- 3a. Analyze individual tool failures
SELECT
  timestamp,
  session_id,
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  error_message,
  root_cause_analysis.ml_generate_text_llm_result AS root_cause
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`,
UNNEST([STRUCT(
  AI.GENERATE_TEXT(
    MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
    CONCAT(
      'Analyze this tool execution failure and provide root cause analysis.\n\n',
      'Tool: ', COALESCE(JSON_EXTRACT_SCALAR(content, '$.tool'), 'unknown'), '\n',
      'Error: ', COALESCE(error_message, 'No error message'), '\n',
      'Tool arguments: ', COALESCE(JSON_EXTRACT_SCALAR(content, '$.args'), 'none'), '\n\n',
      'Provide:\n',
      '1. Root cause category (input_validation, external_api, timeout, permission, other)\n',
      '2. Specific root cause\n',
      '3. Suggested fix\n',
      '4. Is this a transient or permanent error?'
    ),
    STRUCT(0.2 AS temperature, 400 AS max_output_tokens)
  ) AS root_cause_analysis
)])
WHERE event_type = 'TOOL_ERROR'
ORDER BY timestamp DESC
LIMIT 50;

-- 3b. Categorize and aggregate tool failures by root cause
WITH error_analysis AS (
  SELECT
    JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
    error_message,
    AI.GENERATE_TEXT(
      MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
      CONCAT(
        'Categorize this error into ONE category: ',
        'INPUT_ERROR, API_ERROR, TIMEOUT, PERMISSION, RATE_LIMIT, OTHER. ',
        'Reply with only the category name.\n\nError: ',
        COALESCE(error_message, 'unknown error')
      ),
      STRUCT(0.0 AS temperature, 20 AS max_output_tokens)
    ).ml_generate_text_llm_result AS error_category
  FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
  WHERE event_type = 'TOOL_ERROR'
)
SELECT
  tool_name,
  TRIM(error_category) AS root_cause_category,
  COUNT(*) AS occurrence_count,
  ARRAY_AGG(DISTINCT error_message LIMIT 3) AS sample_errors
FROM error_analysis
GROUP BY tool_name, error_category
ORDER BY occurrence_count DESC;


-- ============================================================================
-- 4. AGENT RESPONSE SENTIMENT ANALYSIS
-- ============================================================================
-- Analyze sentiment of agent responses to ensure positive user experience

-- 4a. Sentiment analysis per response
SELECT
  timestamp,
  session_id,
  JSON_EXTRACT_SCALAR(content, '$.response') AS agent_response,
  sentiment_result.ml_generate_text_llm_result AS sentiment_analysis
FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`,
UNNEST([STRUCT(
  AI.GENERATE_TEXT(
    MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
    CONCAT(
      'Analyze the sentiment and tone of this agent response. ',
      'Return JSON: {"sentiment": "positive|neutral|negative", ',
      '"tone": "friendly|professional|apologetic|frustrated|other", ',
      '"confidence": 0.0-1.0, "flags": ["any concerns"]}\n\n',
      'Response: ', JSON_EXTRACT_SCALAR(content, '$.response')
    ),
    STRUCT(0.1 AS temperature, 200 AS max_output_tokens)
  ) AS sentiment_result
)])
WHERE event_type = 'LLM_RESPONSE'
  AND JSON_EXTRACT_SCALAR(content, '$.response') IS NOT NULL
ORDER BY timestamp DESC
LIMIT 100;

-- 4b. Aggregate sentiment trends over time
WITH sentiment_data AS (
  SELECT
    DATE(timestamp) AS date,
    session_id,
    AI.GENERATE_TEXT(
      MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
      CONCAT(
        'Rate sentiment: POSITIVE, NEUTRAL, or NEGATIVE. Reply with one word only.\n',
        'Text: ', LEFT(JSON_EXTRACT_SCALAR(content, '$.response'), 500)
      ),
      STRUCT(0.0 AS temperature, 15 AS max_output_tokens)
    ).ml_generate_text_llm_result AS sentiment
  FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
  WHERE event_type = 'LLM_RESPONSE'
    AND JSON_EXTRACT_SCALAR(content, '$.response') IS NOT NULL
)
SELECT
  date,
  COUNT(*) AS total_responses,
  COUNTIF(UPPER(sentiment) LIKE '%POSITIVE%') AS positive_count,
  COUNTIF(UPPER(sentiment) LIKE '%NEUTRAL%') AS neutral_count,
  COUNTIF(UPPER(sentiment) LIKE '%NEGATIVE%') AS negative_count,
  ROUND(100.0 * COUNTIF(UPPER(sentiment) LIKE '%POSITIVE%') / COUNT(*), 1) AS positive_pct
FROM sentiment_data
GROUP BY date
ORDER BY date DESC;


-- ============================================================================
-- 5. MEMORY & KEY INFORMATION EXTRACTION
-- ============================================================================
-- Extract important facts, preferences, and entities from conversations

-- 5a. Extract user preferences and key facts from sessions
SELECT
  session_id,
  user_id,
  extracted_info.ml_generate_text_llm_result AS extracted_memory
FROM (
  SELECT
    session_id,
    user_id,
    STRING_AGG(
      CONCAT(event_type, ': ', COALESCE(JSON_EXTRACT_SCALAR(content, '$.text'),
             JSON_EXTRACT_SCALAR(content, '$.response'), '')),
      '\n' ORDER BY timestamp
    ) AS conversation
  FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
  WHERE event_type IN ('USER_MESSAGE_RECEIVED', 'LLM_RESPONSE')
  GROUP BY session_id, user_id
) sessions,
UNNEST([STRUCT(
  AI.GENERATE_TEXT(
    MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
    CONCAT(
      'Extract key information from this conversation for long-term memory.\n',
      'Return JSON with:\n',
      '- user_preferences: [list of preferences mentioned]\n',
      '- entities: [people, places, products mentioned]\n',
      '- facts_learned: [factual information about the user]\n',
      '- action_items: [things the user wants to do]\n',
      '- sentiment_overall: positive/neutral/negative\n\n',
      'Conversation:\n', LEFT(conversation, 3000)
    ),
    STRUCT(0.3 AS temperature, 600 AS max_output_tokens)
  ) AS extracted_info
)])
LIMIT 50;

-- 5b. Build user profiles from historical data
SELECT
  user_id,
  COUNT(DISTINCT session_id) AS session_count,
  user_profile.ml_generate_text_llm_result AS profile
FROM (
  SELECT
    user_id,
    STRING_AGG(
      JSON_EXTRACT_SCALAR(content, '$.text'),
      ' | ' ORDER BY timestamp LIMIT 20
    ) AS all_messages
  FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
  WHERE event_type = 'USER_MESSAGE_RECEIVED'
  GROUP BY user_id
) user_data,
UNNEST([STRUCT(
  AI.GENERATE_TEXT(
    MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
    CONCAT(
      'Based on these user messages, create a brief user profile:\n',
      '- Primary interests\n',
      '- Communication style\n',
      '- Expertise level (beginner/intermediate/expert)\n',
      '- Key topics of interest\n\n',
      'Messages: ', LEFT(all_messages, 2000)
    ),
    STRUCT(0.4 AS temperature, 400 AS max_output_tokens)
  ) AS user_profile
)]) profile_gen
CROSS JOIN (
  SELECT user_id, COUNT(DISTINCT session_id) AS session_count
  FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
  GROUP BY user_id
) counts
WHERE user_data.user_id = counts.user_id
GROUP BY user_id, session_count, user_profile.ml_generate_text_llm_result;


-- ============================================================================
-- 6. CONVERSATION QUALITY SCORING (AI.SCORE pattern)
-- ============================================================================
-- Score entire conversations for quality assurance

SELECT
  session_id,
  user_id,
  conversation_score.ml_generate_text_llm_result AS quality_assessment
FROM (
  SELECT
    session_id,
    user_id,
    STRING_AGG(
      CASE
        WHEN event_type = 'USER_MESSAGE_RECEIVED' THEN CONCAT('User: ', JSON_EXTRACT_SCALAR(content, '$.text'))
        WHEN event_type = 'LLM_RESPONSE' THEN CONCAT('Agent: ', LEFT(JSON_EXTRACT_SCALAR(content, '$.response'), 200))
        WHEN event_type = 'TOOL_COMPLETED' THEN CONCAT('Tool [', JSON_EXTRACT_SCALAR(content, '$.tool'), ']: success')
        WHEN event_type = 'TOOL_ERROR' THEN CONCAT('Tool [', JSON_EXTRACT_SCALAR(content, '$.tool'), ']: FAILED')
      END,
      '\n' ORDER BY timestamp
    ) AS conversation
  FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
  WHERE event_type IN ('USER_MESSAGE_RECEIVED', 'LLM_RESPONSE', 'TOOL_COMPLETED', 'TOOL_ERROR')
  GROUP BY session_id, user_id
) sessions,
UNNEST([STRUCT(
  AI.GENERATE_TEXT(
    MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
    CONCAT(
      'Score this agent conversation (1-10) on:\n',
      '1. Task completion - Did the agent help the user achieve their goal?\n',
      '2. Efficiency - Were minimal turns needed?\n',
      '3. Accuracy - Were tool results used correctly?\n',
      '4. User experience - Was the interaction pleasant?\n\n',
      'Return JSON: {"task_completion": X, "efficiency": X, "accuracy": X, ',
      '"user_experience": X, "overall": X, "summary": "brief assessment"}\n\n',
      'Conversation:\n', LEFT(conversation, 2500)
    ),
    STRUCT(0.2 AS temperature, 400 AS max_output_tokens)
  ) AS conversation_score
)])
ORDER BY session_id DESC
LIMIT 50;


-- ============================================================================
-- 7. ANOMALY DETECTION WITH AI
-- ============================================================================
-- Detect unusual patterns in agent behavior

SELECT
  DATE(timestamp) AS date,
  agent,
  anomaly_check.ml_generate_text_llm_result AS anomaly_report
FROM (
  SELECT
    DATE(timestamp) AS date,
    agent,
    COUNT(*) AS total_events,
    COUNTIF(event_type = 'TOOL_ERROR') AS tool_errors,
    COUNTIF(event_type = 'LLM_ERROR') AS llm_errors,
    AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)) AS avg_latency,
    MAX(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)) AS max_latency
  FROM `test-project-0728-467323.agent_analytics_demo.agent_events_v2`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  GROUP BY date, agent
) daily_stats,
UNNEST([STRUCT(
  AI.GENERATE_TEXT(
    MODEL `test-project-0728-467323.agent_analytics_demo.gemini_model`,
    CONCAT(
      'Analyze these daily agent metrics for anomalies:\n',
      '- Total events: ', CAST(total_events AS STRING), '\n',
      '- Tool errors: ', CAST(tool_errors AS STRING), '\n',
      '- LLM errors: ', CAST(llm_errors AS STRING), '\n',
      '- Avg latency: ', CAST(ROUND(avg_latency, 0) AS STRING), 'ms\n',
      '- Max latency: ', CAST(ROUND(max_latency, 0) AS STRING), 'ms\n\n',
      'Is this anomalous? Return JSON: {"is_anomaly": boolean, "severity": "none|low|medium|high", ',
      '"concerns": ["list of concerns"], "recommendation": "action to take"}'
    ),
    STRUCT(0.1 AS temperature, 300 AS max_output_tokens)
  ) AS anomaly_check
)])
ORDER BY date DESC;
