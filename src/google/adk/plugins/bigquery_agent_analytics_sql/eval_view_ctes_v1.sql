-- Copyright 2026 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
--
-- Shared CTE chain of the BigQuery Agent Analytics eval session and eval turn
-- views.  It rebuilds the events of each session or turn into an AgentData
-- trace and the metrics beside it; the query templates continue it.
--
-- Substitution contract: strip this header (everything up to the first blank
-- line), then replace each placeholder named below, a name in single braces,
-- with plain text.  Strip it first: the header names the placeholders, and a
-- value that spans lines would break out of its comment.  The template has no
-- doubled braces and declares no query parameters, so Python's str.format and
-- any other textual replacement produce the same SQL.
--
-- Placeholders and their legal values:
--
-- {full_table_id}
--     The events table as project.dataset.table.  The template wraps it in
--     backticks, so it has no backtick, backslash, control or line-break
--     character.
-- {content_parts_column}
--     content_parts, or, when that column is withheld from the reader,
--     ARRAY<STRUCT<part_index INT64, mime_type STRING, text STRING,
--     part_attributes STRING, storage_mode STRING>>[] AS content_parts.
-- {attributes_column}
--     attributes, or CAST(NULL AS JSON) AS attributes when it is withheld.
-- {latency_ms_column}
--     latency_ms, or CAST(NULL AS JSON) AS latency_ms when it is withheld.
-- {scope_columns}
--     session_id for one trace per session, or session_id, invocation_id for
--     one trace per turn.
-- {raw_where_clause}
--     Empty, or a trusted condition that narrows the events table's rows
--     that every CTE reads.  The template puts it right after the last
--     filter, so it starts with a newline and has this shape, with the closing
--     parenthesis on a line of its own so that a trailing line comment in the
--     condition cannot swallow it:
--
--       AND (
--         condition
--       )
--
--     The condition:
--     * is SQL: bind its values as query parameters that the running job
--       supplies, and never splice in untrusted text;
--     * must keep every event of each session or turn to read: its lower
--       bound is no later than the first event of any such session, and its
--       upper bound is after the last event of those sessions plus the delay
--       before the plugin's writes are readable, or the session or turn comes
--       out partial;
--     * never filters on a column that varies within a session, such as agent
--       or event_type.
--     A lookback on timestamp, one that starts well before the window to
--     read, lets BigQuery prune the table's partitions.  To pick the sessions
--     or turns of a time window, keep this lookback wide and filter the
--     output columns with the where_clause of the query templates instead.
--
-- Truncated content: a row that has is_truncated set lacks part of its
-- payload.  When the row puts an event in the trace, has_truncated_content
-- holds for its turn, and neither eval_ready nor session_turn_eval_ready
-- does.  No other row counts, such as an LLM_REQUEST row: it repeats the
-- conversation so far, so one oversized message would hold back every later
-- turn of its session.  has_truncated_content tells a caller that truncation
-- keeps a turn from being ready, but not whether it is the only reason: no
-- column of the views holds the other conditions of eval_ready, such as
-- has_user_message and is_complete, so a truncated turn may also be
-- unfinished or lack user text.
--
-- Nested sub-agent sessions: an AgentTool call runs its sub-agent in a
-- throwaway child session whose rows carry attributes.adk.parent_session_id.
-- SessionEvents excludes those child-session rows because the parent
-- session's trace already records the AgentTool call and response as a tool
-- step.  Rows without that field, or from tables where attributes is
-- withheld, evaluate to NULL and are kept.
--
-- Versioning: the _v1 in the file name is the version of this contract.  A
-- backward-compatible fix, which keeps the placeholders and the output
-- columns, edits the file in place; a change to either needs a new _v2 file.

WITH
  -- The events of the sessions to read, one row per event.  Every later step
  -- reads these rows rather than the table, so a filter here narrows them all.
  RawEvents AS (
    SELECT
      timestamp,
      event_id,
      event_type,
      agent,
      session_id,
      invocation_id,
      user_id,
      content,
      {content_parts_column},
      {attributes_column},
      {latency_ms_column},
      status,
      error_message,
      is_truncated
    FROM
      `{full_table_id}`
    WHERE
      session_id IS NOT NULL
      AND invocation_id IS NOT NULL{raw_where_clause}
  ),
  -- Keeps only top-level session events, excluding throwaway in-memory
  -- sessions created by nested AgentTool calls (which carry
  -- attributes.adk.parent_session_id and already appear as tool calls inside
  -- their parent session's trace).  Rows without that field, or from tables
  -- that withhold attributes, evaluate to NULL and are kept.
  SessionEvents AS (
    SELECT
      *
    FROM
      RawEvents
    WHERE
      JSON_VALUE(attributes, '$.adk.parent_session_id') IS NULL
  ),
  -- Each event with the fields that later steps read parsed out of its JSON
  -- payload: its author, the tool name and args, whether the tool is a
  -- framework tool, the user text, the model response before it is unwrapped,
  -- and the token and latency numbers.
  TypedEvents AS (
    SELECT
      *,
      COALESCE(
        agent, JSON_VALUE(attributes, '$.root_agent_name'), 'root_agent'
      ) AS author,
      JSON_VALUE(content, '$.tool') AS tool_name,
      -- The framework tools are the transfer and the tools that pause the turn
      -- for a credential, a confirmation or input from the user.  The trace
      -- shows their AGENT_TRANSFER and HITL rows instead of their tool rows.
      -- A row that names no tool is not a framework tool.
      COALESCE(JSON_VALUE(content, '$.tool'), '') IN (
        'transfer_to_agent',
        'adk_request_credential',
        'adk_request_confirmation',
        'adk_request_input'
      ) AS is_framework_tool,
      TO_JSON_STRING(JSON_QUERY(content, '$.args')) AS tool_args_json,
      CASE
        WHEN event_type != 'USER_MESSAGE_RECEIVED' THEN NULL
        WHEN COALESCE(ARRAY_LENGTH(content_parts), 0) > 0 THEN (
          SELECT
            STRING_AGG(p.text, '\n' ORDER BY p.part_index)
          FROM
            UNNEST(content_parts) AS p
          WHERE
            p.mime_type = 'text/plain'
            AND COALESCE(p.storage_mode, 'INLINE') = 'INLINE'
            AND COALESCE(
              p.part_attributes, TO_JSON_STRING(JSON_OBJECT())
            ) = TO_JSON_STRING(JSON_OBJECT())
            AND p.text IS NOT NULL
            AND p.text NOT IN ('', '[BINARY DATA]', '[UPLOAD FAILED]')
        )
        WHEN NOT REGEXP_CONTAINS(
          COALESCE(
            JSON_VALUE(content, '$.text_summary'),
            JSON_VALUE(content, '$'),
            ''
          ),
          r'^Function response: '
        ) THEN NULLIF(
          COALESCE(
            JSON_VALUE(content, '$.text_summary'), JSON_VALUE(content, '$')
          ),
          ''
        )
        ELSE NULL
      END AS message_text,
      (
        event_type = 'USER_MESSAGE_RECEIVED'
        AND (
          REGEXP_CONTAINS(
            COALESCE(
              JSON_VALUE(content, '$.text_summary'),
              JSON_VALUE(content, '$'),
              ''
            ),
            r'^Function response: '
          )
          OR EXISTS(
            SELECT 1
            FROM UNNEST(content_parts) AS p
            WHERE
              JSON_QUERY(p.part_attributes, '$.function_response')
              IS NOT NULL
          )
        )
      ) AS is_function_response,
      -- Keep sentinels on $.text_summary and $.response in TypedEvents so
      -- COALESCE fails closed instead of falling back to $.response.
      -- ResponseEvents maps those sentinels to NULL, while top-level string
      -- content excludes them directly below.
      IF(
        event_type = 'AGENT_RESPONSE',
        COALESCE(
          IF(
            JSON_TYPE(JSON_QUERY(content, '$.text_summary')) = 'string'
              AND TRIM(JSON_VALUE(content, '$.text_summary')) != '',
            JSON_VALUE(content, '$.text_summary'),
            NULL
          ),
          IF(
            TRIM(JSON_VALUE(content, '$.response')) != '',
            JSON_VALUE(content, '$.response'),
            NULL
          ),
          IF(
            JSON_TYPE(JSON_QUERY(content, '$.response'))
              IN ('object', 'array'),
            TO_JSON_STRING(JSON_QUERY(content, '$.response')),
            NULL
          ),
          IF(
            JSON_TYPE(content) = 'string'
              AND TRIM(JSON_VALUE(content, '$')) != ''
              AND JSON_VALUE(content, '$') NOT IN (
                '[FORMATTER_FAILED]',
                '[FORMATTING FAILED]',
                '[REDACTED]',
                '[CONTENT_PARSE_FAILED]',
                '[UNPARSEABLE_JSON_BLOB]',
                '[UNSUPPORTED_OBJECT]',
                '[REDACTED_SENSITIVE_TEXT]'
              ),
            JSON_VALUE(content, '$'),
            NULL
          )
        ),
        NULL
      ) AS formatted_response,
      (
        event_type = 'LLM_RESPONSE'
        AND (
          JSON_VALUE(attributes, '$.finish_reason') IS NOT NULL
          OR JSON_VALUE(attributes, '$.cache_type') IS NOT NULL
        )
      ) AS is_final_llm_response,
      SAFE_CAST(JSON_VALUE(content, '$.usage.total') AS INT64) AS usage_total,
      SAFE_CAST(JSON_VALUE(content, '$.usage.prompt') AS INT64)
        AS usage_prompt,
      SAFE_CAST(JSON_VALUE(content, '$.usage.completion') AS INT64)
        AS usage_completion,
      SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms
    FROM
      SessionEvents
  ),
  -- Marks the real tool calls, the calls of tools other than the framework
  -- tools, and unwraps the text of each model response.  It also counts the
  -- HITL requests so far in each turn.
  ResponseEvents AS (
    SELECT
      *,
      -- A TOOL_STARTING row that names no tool, as when a custom
      -- content_formatter fails, is not a tool call, since the trace could not
      -- pair it with its response.
      (
        event_type = 'TOOL_STARTING'
        AND NULLIF(tool_name, '') IS NOT NULL
        AND NOT is_framework_tool
      ) AS is_tool_call,
      -- The number of HITL requests so far in the turn.  A turn that resumes
      -- after one keeps its invocation_id, so the count tells the calls before
      -- a pause from the calls after it.
      COUNTIF(
        event_type IN (
          'HITL_CREDENTIAL_REQUEST',
          'HITL_CONFIRMATION_REQUEST',
          'HITL_INPUT_REQUEST'
        )
      ) OVER (
        PARTITION BY session_id, invocation_id
        ORDER BY timestamp, event_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS hitl_request_count,
      -- The text parts that _format_content renders become plain text, and a
      -- response with any other part has no text.
      CASE
        WHEN
          REGEXP_CONTAINS(formatted_response, r"(?s)^text: '.*'$")
          AND NOT REGEXP_CONTAINS(
            formatted_response,
            r"' [|] (?:call: |resp: |Executable code |Code execution result )"
          )
          AND NOT REGEXP_CONTAINS(
            formatted_response, r"' [|] other(?: [|] |$)"
          )
          THEN IF(
            TRIM(
              REGEXP_REPLACE(
                REGEXP_REPLACE(formatted_response, r"^text: '|'$", ''),
                r"' [|] text: '",
                CHR(10)
              )
            ) != ''
              AND REGEXP_REPLACE(
                REGEXP_REPLACE(formatted_response, r"^text: '|'$", ''),
                r"' [|] text: '",
                CHR(10)
              ) NOT IN (
                '[FORMATTER_FAILED]',
                '[FORMATTING FAILED]',
                '[REDACTED]',
                '[CONTENT_PARSE_FAILED]',
                '[UNPARSEABLE_JSON_BLOB]',
                '[UNSUPPORTED_OBJECT]',
                '[REDACTED_SENSITIVE_TEXT]'
              ),
            REGEXP_REPLACE(
              REGEXP_REPLACE(formatted_response, r"^text: '|'$", ''),
              r"' [|] text: '",
              CHR(10)
            ),
            NULL
          )
        WHEN
          NOT REGEXP_CONTAINS(
            formatted_response,
            r'^(?:call: |resp: |Executable code |Code execution result )'
          )
          AND NOT REGEXP_CONTAINS(
            formatted_response, r"^(?:None$|text: '|other(?: [|] |$))"
          )
          AND TRIM(formatted_response) != ''
          AND formatted_response NOT IN (
            '[FORMATTER_FAILED]',
            '[FORMATTING FAILED]',
            '[REDACTED]',
            '[CONTENT_PARSE_FAILED]',
            '[UNPARSEABLE_JSON_BLOB]',
            '[UNSUPPORTED_OBJECT]',
            '[REDACTED_SENSITIVE_TEXT]'
          )
          THEN formatted_response
        ELSE NULL
      END AS response_text
    FROM
      TypedEvents
  ),
  -- Adds two marks to each event: whether its turn has a final model
  -- response, which picks the rows that count tokens, and the island number
  -- that a run of consecutive identical tool calls shares.  The second
  -- ROW_NUMBER restarts after each HITL request, so the request ends a run,
  -- as a different call does.
  AnnotatedEvents AS (
    SELECT
      *,
      LOGICAL_OR(is_final_llm_response) OVER (
        PARTITION BY session_id, invocation_id
      ) AS turn_has_final_llm_response,
      ROW_NUMBER() OVER (
        PARTITION BY session_id, invocation_id, is_tool_call
        ORDER BY timestamp, event_id
      ) - ROW_NUMBER() OVER (
        PARTITION BY
          session_id,
          invocation_id,
          is_tool_call,
          tool_name,
          tool_args_json,
          hitl_request_count
        ORDER BY timestamp, event_id
      ) AS tool_call_island
    FROM
      ResponseEvents
  ),
  -- Each event as a step of the AgentData trace: its author, its place among
  -- the events that share its timestamp, and its content as user text, a
  -- function call, a function response or model text, if it has any.
  TraceEvents AS (
    SELECT
      session_id,
      invocation_id,
      timestamp,
      event_id,
      event_type,
      user_id,
      agent,
      JSON_VALUE(attributes, '$.root_agent_name') AS root_agent_name,
      status,
      error_message,
      is_truncated,
      message_text,
      response_text,
      is_function_response,
      is_tool_call,
      is_framework_tool,
      IF(is_tool_call, tool_name, NULL) AS invoked_tool_name,
      IF(
        is_tool_call,
        COUNT(*) OVER (
          PARTITION BY
            session_id,
            invocation_id,
            is_tool_call,
            tool_name,
            tool_args_json,
            tool_call_island
        ),
        NULL
      ) AS tool_call_run_length,
      (
        event_type = 'LLM_RESPONSE'
        AND (is_final_llm_response OR NOT turn_has_final_llm_response)
      ) AS is_usage_row,
      usage_total,
      usage_prompt,
      usage_completion,
      total_ms,
      -- Orders the events that share a timestamp: the user message, a call, a
      -- transfer, a response, and then the rest.
      CASE
        WHEN event_type = 'USER_MESSAGE_RECEIVED' THEN 1
        WHEN
          event_type IN (
            'TOOL_STARTING',
            'HITL_CREDENTIAL_REQUEST',
            'HITL_CONFIRMATION_REQUEST',
            'HITL_INPUT_REQUEST'
          )
          THEN 2
        WHEN event_type = 'AGENT_TRANSFER' THEN 3
        WHEN
          event_type IN (
            'TOOL_COMPLETED',
            'TOOL_ERROR',
            'HITL_CREDENTIAL_REQUEST_COMPLETED',
            'HITL_CONFIRMATION_REQUEST_COMPLETED',
            'HITL_INPUT_REQUEST_COMPLETED'
          )
          THEN 4
        ELSE 5
      END AS event_seq,
      CASE event_type
        WHEN 'USER_MESSAGE_RECEIVED' THEN 'user'
        WHEN 'AGENT_TRANSFER'
          THEN COALESCE(JSON_VALUE(content, '$.from_agent'), author)
        ELSE author
      END AS event_author,
      CASE
        WHEN event_type = 'USER_MESSAGE_RECEIVED' AND message_text IS NOT NULL
          THEN JSON_OBJECT(
            'role', 'user',
            'parts', JSON_ARRAY(JSON_OBJECT('text', message_text))
          )
        WHEN
          is_tool_call
          OR event_type IN (
            'HITL_CREDENTIAL_REQUEST',
            'HITL_CONFIRMATION_REQUEST',
            'HITL_INPUT_REQUEST'
          )
          THEN JSON_OBJECT(
            'role', 'model',
            'parts', JSON_ARRAY(
              JSON_OBJECT(
                'function_call', JSON_OBJECT(
                  'name', COALESCE(tool_name, ''),
                  'args', IF(
                    JSON_TYPE(JSON_QUERY(content, '$.args')) = 'object',
                    JSON_QUERY(content, '$.args'),
                    JSON_OBJECT()
                  )
                )
              )
            )
          )
        WHEN event_type = 'AGENT_TRANSFER'
          THEN JSON_OBJECT(
            'role', 'model',
            'parts', JSON_ARRAY(
              JSON_OBJECT(
                'function_call', JSON_OBJECT(
                  'name', 'transfer_to_agent',
                  'args', JSON_OBJECT(
                    'agent_name',
                    COALESCE(JSON_VALUE(content, '$.to_agent'), '')
                  )
                )
              )
            )
          )
        WHEN
          (
            event_type = 'TOOL_COMPLETED'
            AND NULLIF(tool_name, '') IS NOT NULL
            AND NOT is_framework_tool
          )
          OR event_type IN (
            'HITL_CREDENTIAL_REQUEST_COMPLETED',
            'HITL_CONFIRMATION_REQUEST_COMPLETED',
            'HITL_INPUT_REQUEST_COMPLETED'
          )
          THEN JSON_OBJECT(
            'role', 'user',
            'parts', JSON_ARRAY(
              JSON_OBJECT(
                'function_response', JSON_OBJECT(
                  'name', COALESCE(tool_name, ''),
                  'response', IF(
                    JSON_TYPE(JSON_QUERY(content, '$.result')) = 'object',
                    JSON_QUERY(content, '$.result'),
                    JSON_OBJECT('result', JSON_QUERY(content, '$.result'))
                  )
                )
              )
            )
          )
        WHEN
          event_type = 'TOOL_ERROR'
          AND NULLIF(tool_name, '') IS NOT NULL
          AND NOT is_framework_tool
          THEN JSON_OBJECT(
            'role', 'user',
            'parts', JSON_ARRAY(
              JSON_OBJECT(
                'function_response', JSON_OBJECT(
                  'name', tool_name,
                  'response', JSON_OBJECT(
                    'error',
                    COALESCE(NULLIF(error_message, ''), 'Tool execution failed')
                  )
                )
              )
            )
          )
        WHEN event_type = 'AGENT_RESPONSE' AND response_text IS NOT NULL
          THEN JSON_OBJECT(
            'role', 'model',
            'parts', JSON_ARRAY(JSON_OBJECT('text', response_text))
          )
        ELSE NULL
      END AS event_content
    FROM
      AnnotatedEvents
  ),
  -- One row per turn, with its trace events in order and its metrics: tool
  -- calls and errors, token totals, latency, and whether it paused or
  -- finished.
  TurnRollup AS (
    SELECT
      session_id,
      invocation_id,
      ARRAY_AGG(
        user_id IGNORE NULLS ORDER BY timestamp LIMIT 1
      )[SAFE_OFFSET(0)] AS user_id,
      ARRAY_AGG(
        root_agent_name IGNORE NULLS ORDER BY timestamp LIMIT 1
      )[SAFE_OFFSET(0)] AS first_root_agent_name,
      ARRAY_AGG(
        agent IGNORE NULLS ORDER BY timestamp LIMIT 1
      )[SAFE_OFFSET(0)] AS first_agent,
      MIN(timestamp) AS turn_start_timestamp,
      MAX(timestamp) AS turn_end_timestamp,
      COUNTIF(is_tool_call) AS tool_call_count,
      COUNTIF(event_type = 'TOOL_ERROR' AND NOT is_framework_tool)
        AS tool_error_count,
      COUNTIF(event_type = 'LLM_ERROR') AS llm_error_count,
      COUNTIF(event_type = 'AGENT_ERROR') AS agent_error_count,
      COALESCE(
        LOGICAL_OR(
          status = 'ERROR'
          OR (COALESCE(status, '') != 'OK' AND error_message IS NOT NULL)
          OR (event_type = 'LLM_RESPONSE' AND error_message IS NOT NULL)
        ),
        FALSE
      ) AS has_error,
      -- Whether any row that puts an event in the trace, one with an
      -- event_content, has is_truncated set.  No other row counts: an
      -- LLM_REQUEST row repeats the conversation so far, so one oversized
      -- message would flag it in every later turn, though the events that
      -- the trace takes from that turn are whole.
      LOGICAL_OR(event_content IS NOT NULL AND COALESCE(is_truncated, FALSE))
        AS has_truncated_content,
      COALESCE(MAX(tool_call_run_length), 0) >= 3 AS has_tool_loop,
      COALESCE(MAX(tool_call_run_length), 0)
        AS max_consecutive_identical_tool_calls,
      COALESCE(
        ARRAY_AGG(
          DISTINCT invoked_tool_name IGNORE NULLS ORDER BY invoked_tool_name
        ),
        ARRAY<STRING>[]
      ) AS distinct_tools_used,
      COALESCE(SUM(IF(is_usage_row, usage_prompt, NULL)), 0)
        AS total_prompt_tokens,
      COALESCE(SUM(IF(is_usage_row, usage_completion, NULL)), 0)
        AS total_completion_tokens,
      COALESCE(SUM(IF(is_usage_row, usage_total, NULL)), 0) AS total_tokens,
      COALESCE(
        MAX(IF(event_type = 'INVOCATION_COMPLETED', total_ms, NULL)),
        MAX(IF(event_type = 'AGENT_COMPLETED', total_ms, NULL))
      ) AS turn_latency_ms,
      (
        COALESCE(LOGICAL_OR(event_type = 'INVOCATION_COMPLETED'), FALSE)
        AND NOT COALESCE(
          LOGICAL_OR(
            event_type = 'AGENT_RESPONSE' AND response_text IS NOT NULL
          ),
          FALSE
        )
        AND COALESCE(
          LOGICAL_OR(
            event_type IN (
              'TOOL_PAUSED',
              'HITL_CREDENTIAL_REQUEST',
              'HITL_CONFIRMATION_REQUEST',
              'HITL_INPUT_REQUEST'
            )
          ),
          FALSE
        )
      ) AS is_paused,
      (
        COALESCE(LOGICAL_OR(event_type = 'INVOCATION_COMPLETED'), FALSE)
        AND COALESCE(
          LOGICAL_OR(
            event_type = 'AGENT_RESPONSE' AND response_text IS NOT NULL
          ),
          FALSE
        )
      ) AS is_complete,
      COALESCE(
        LOGICAL_OR(
          event_type = 'USER_MESSAGE_RECEIVED' AND message_text IS NOT NULL
        ),
        FALSE
      ) AS has_user_message,
      COALESCE(LOGICAL_OR(is_function_response), FALSE)
        AS has_function_response,
      ARRAY_AGG(
        IF(
          event_content IS NULL,
          NULL,
          JSON_OBJECT(
            'author', event_author,
            'content', event_content,
            'event_time',
            FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%E6SZ', timestamp)
          )
        )
        IGNORE NULLS
        ORDER BY timestamp, event_seq, event_id
      ) AS events
    FROM
      TraceEvents
    GROUP BY
      session_id,
      invocation_id
  ),
  -- Each turn with its place in the session and whether it is ready for
  -- evaluation, on its own or as a part of its session.
  Turns AS (
    SELECT
      *,
      COALESCE(first_root_agent_name, first_agent) AS root_agent,
      ROW_NUMBER() OVER (
        PARTITION BY session_id
        ORDER BY turn_start_timestamp, invocation_id
      ) - 1 AS turn_index,
      IF(
        COALESCE(ARRAY_LENGTH(events), 0) > 0, TO_JSON(events), JSON_ARRAY()
      ) AS events_json,
      (
        has_user_message
        AND is_complete
        AND NOT has_truncated_content
        AND COALESCE(ARRAY_LENGTH(events), 0) > 0
      ) AS eval_ready,
      -- In a session, a paused turn and the function response turn that
      -- resumes it are ready together.
      (
        (
          has_user_message
          OR (
            has_function_response
            AND COALESCE(
              LAG(is_paused) OVER (
                PARTITION BY session_id
                ORDER BY turn_start_timestamp, invocation_id
              ),
              FALSE
            )
          )
        )
        AND (
          is_complete
          OR (
            is_paused
            AND COALESCE(
              LEAD(has_function_response) OVER (
                PARTITION BY session_id
                ORDER BY turn_start_timestamp, invocation_id
              ),
              FALSE
            )
          )
        )
        AND NOT has_truncated_content
        AND COALESCE(ARRAY_LENGTH(events), 0) > 0
      ) AS session_turn_eval_ready
    FROM
      TurnRollup
  ),
  -- One row per event of an agent, with the parts of its configuration that
  -- the event logs: the instruction, description and sub agents of
  -- AGENT_STARTING, and the tool declarations of LLM_REQUEST.  An instruction
  -- that is a callable, logged as its repr, is dropped.
  AgentRows AS (
    SELECT
      session_id,
      invocation_id,
      agent AS agent_id,
      timestamp,
      IF(
        event_type = 'AGENT_STARTING',
        COALESCE(
          JSON_VALUE(content, '$.text_summary'),
          IF(
            REGEXP_CONTAINS(
              JSON_VALUE(content, '$'),
              r'^(?:<(?:function|bound method) |<.+ at 0x[0-9a-fA-F]+>|functools\.partial\()'
            ),
            NULL,
            JSON_VALUE(content, '$')
          ),
          JSON_VALUE(attributes, '$.static_instruction')
        ),
        NULL
      ) AS instruction,
      IF(
        event_type = 'AGENT_STARTING',
        JSON_VALUE(attributes, '$.agent_description'),
        NULL
      ) AS description,
      IF(
        event_type = 'AGENT_STARTING'
        AND JSON_TYPE(JSON_QUERY(attributes, '$.sub_agents')) = 'array',
        JSON_QUERY(attributes, '$.sub_agents'),
        NULL
      ) AS sub_agents,
      IF(
        event_type = 'LLM_REQUEST'
        AND JSON_TYPE(JSON_QUERY(attributes, '$.tools')) = 'array',
        JSON_QUERY(attributes, '$.tools'),
        NULL
      ) AS tool_declarations
    FROM
      SessionEvents
    WHERE
      agent IS NOT NULL
  ),
  -- The latest logged value of each part of the configuration of each agent,
  -- in each session or turn.
  AgentLatest AS (
    SELECT
      {scope_columns},
      agent_id,
      ARRAY_AGG(
        instruction IGNORE NULLS ORDER BY timestamp DESC LIMIT 1
      )[SAFE_OFFSET(0)] AS instruction,
      ARRAY_AGG(
        description IGNORE NULLS ORDER BY timestamp DESC LIMIT 1
      )[SAFE_OFFSET(0)] AS description,
      ARRAY_AGG(
        sub_agents IGNORE NULLS ORDER BY timestamp DESC LIMIT 1
      )[SAFE_OFFSET(0)] AS sub_agents,
      ARRAY_AGG(
        tool_declarations IGNORE NULLS ORDER BY timestamp DESC LIMIT 1
      )[SAFE_OFFSET(0)] AS tool_declarations
    FROM
      AgentRows
    GROUP BY
      {scope_columns},
      agent_id
  ),
  -- The function declarations of the latest tools that each agent sent to the
  -- model, in the order that they were logged.
  AgentToolDeclarations AS (
    SELECT
      {scope_columns},
      agent_id,
      ARRAY_AGG(
        IF(
          JSON_TYPE(JSON_QUERY(declaration, '$.parameters')) = 'object',
          JSON_OBJECT(
            'name', JSON_VALUE(declaration, '$.name'),
            'description',
            COALESCE(JSON_VALUE(declaration, '$.description'), ''),
            'parameters_json_schema', JSON_QUERY(declaration, '$.parameters')
          ),
          JSON_OBJECT(
            'name', JSON_VALUE(declaration, '$.name'),
            'description',
            COALESCE(JSON_VALUE(declaration, '$.description'), '')
          )
        )
        ORDER BY declaration_offset
      ) AS function_declarations
    FROM
      AgentLatest
    INNER JOIN
      UNNEST(JSON_QUERY_ARRAY(tool_declarations)) AS declaration
      WITH OFFSET AS declaration_offset
    WHERE
      JSON_VALUE(declaration, '$.name') IS NOT NULL
    GROUP BY
      {scope_columns},
      agent_id
  ),
  -- The first non-blank system prompt that each agent sent to the model in
  -- the session.  An agent whose instruction is a callable logs no
  -- instruction text, so this is its fallback.
  SessionAgentPrompts AS (
    SELECT
      session_id,
      agent AS agent_id,
      ARRAY_AGG(
        IF(
          JSON_TYPE(JSON_QUERY(content, '$.system_prompt')) = 'string'
            AND TRIM(JSON_VALUE(content, '$.system_prompt')) != '',
          JSON_VALUE(content, '$.system_prompt'),
          NULL
        )
        IGNORE NULLS
        ORDER BY timestamp, event_id
        LIMIT 1
      )[SAFE_OFFSET(0)] AS logged_system_prompt
    FROM
      SessionEvents
    WHERE
      event_type = 'LLM_REQUEST'
      AND agent IS NOT NULL
    GROUP BY
      session_id,
      agent
  ),
  -- The AgentConfig of each agent.  Its instruction comes from the latest
  -- AGENT_STARTING event, else from the logged system prompt.
  AgentConfigs AS (
    SELECT
      {scope_columns},
      agent_id,
      JSON_OBJECT(
        'agent_id', agent_id,
        'instruction',
        COALESCE(NULLIF(instruction, ''), logged_system_prompt, ''),
        'description', COALESCE(description, ''),
        'sub_agents', COALESCE(sub_agents, JSON_ARRAY()),
        'tools', IF(
          function_declarations IS NULL,
          JSON_ARRAY(),
          JSON_ARRAY(
            JSON_OBJECT(
              'function_declarations', TO_JSON(function_declarations)
            )
          )
        )
      ) AS agent_config
    FROM
      AgentLatest
    LEFT JOIN
      SessionAgentPrompts
      USING (session_id, agent_id)
    LEFT JOIN
      AgentToolDeclarations
      USING ({scope_columns}, agent_id)
  ),
  -- The agents of each session or turn, as one JSON object keyed by agent id.
  AgentMaps AS (
    SELECT
      {scope_columns},
      JSON_OBJECT(
        ARRAY_AGG(agent_id ORDER BY agent_id),
        ARRAY_AGG(agent_config ORDER BY agent_id)
      ) AS agents
    FROM
      AgentConfigs
    GROUP BY
      {scope_columns}
  )
