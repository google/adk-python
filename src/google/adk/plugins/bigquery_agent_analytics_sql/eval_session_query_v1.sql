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
-- Eval session query of the BigQuery Agent Analytics plugin: the rows of the
-- eval session view, for a job that reads them without creating the view and
-- so can narrow both the events that it reads and the rows that it returns.
-- The query ends without a semicolon, so the job can append ORDER BY or LIMIT
-- to it, or read it as a subquery.
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
-- {ctes}
--     The text of eval_view_ctes_v1.sql after its header, with its
--     placeholders replaced and session_id as its scope columns: a WITH clause
--     that ends after its last CTE, with no trailing comma, which this
--     template continues.  That file lists its placeholders and the rules for
--     a condition that narrows the events table, whose query parameters the
--     running job supplies.
-- {where_clause}
--     Empty, or a trusted condition on the output columns that picks the rows
--     to return, such as the sessions that ended in a time window.  It filters
--     whole sessions after they are rebuilt, so it drops no event that a
--     session it keeps needs.  The template puts it right after the last
--     source, so it starts with a newline and has this shape, here indented,
--     with WHERE at the start of its line and the closing parenthesis on a
--     line of its own so that a trailing line comment in the condition cannot
--     swallow it:
--
--       WHERE
--         (
--           condition
--         )
--
--     The condition is SQL: bind its values as query parameters that the
--     running job supplies, and never splice in untrusted text.
--
-- View: the plugin's eval session view runs this query with no conditions.
-- To create the view, the plugin leaves both the raw_where_clause of {ctes}
-- and {where_clause} empty, and puts this line before the filled template,
-- with the view's name, as project.dataset.view_name, in the backticks:
--
--   CREATE OR REPLACE VIEW `project.dataset.view_name` AS
--
-- The name has no backtick, backslash, control or line-break character.
--
-- Truncated content: a row that puts an event in the trace and has
-- is_truncated set lacks part of its payload, so has_truncated_content is
-- TRUE for its session, and eval_ready is FALSE.  No other row counts, such
-- as an LLM_REQUEST row, which repeats the conversation so far.
-- has_truncated_content tells a caller that truncation keeps a session from
-- being ready, but not whether it is the only reason: no output column holds
-- the other conditions of eval_ready, so a truncated session may also have a
-- turn that is unfinished or lacks user text.
--
-- Versioning: the _v1 in the file name is the version of this contract.  A
-- backward-compatible fix, which keeps the placeholders and the output
-- columns, edits the file in place; a change to either needs a new _v2 file.

{ctes},
  -- One row per session, rolled up from its turns.
  SessionRollup AS (
    SELECT
      session_id,
      ARRAY_AGG(
        user_id IGNORE NULLS ORDER BY turn_start_timestamp LIMIT 1
      )[SAFE_OFFSET(0)] AS user_id,
      COALESCE(
        ARRAY_AGG(
          first_root_agent_name IGNORE NULLS
          ORDER BY turn_start_timestamp
          LIMIT 1
        )[SAFE_OFFSET(0)],
        ARRAY_AGG(
          first_agent IGNORE NULLS ORDER BY turn_start_timestamp LIMIT 1
        )[SAFE_OFFSET(0)]
      ) AS root_agent,
      MIN(turn_start_timestamp) AS session_start_timestamp,
      MAX(turn_end_timestamp) AS session_end_timestamp,
      COUNT(DISTINCT invocation_id) AS turn_count,
      SUM(tool_call_count) AS tool_call_count,
      SUM(tool_error_count) AS tool_error_count,
      SUM(llm_error_count) AS llm_error_count,
      SUM(agent_error_count) AS agent_error_count,
      LOGICAL_OR(has_error) AS has_error,
      LOGICAL_OR(has_truncated_content) AS has_truncated_content,
      LOGICAL_OR(has_tool_loop) AS has_tool_loop,
      MAX(max_consecutive_identical_tool_calls)
        AS max_consecutive_identical_tool_calls,
      ARRAY_CONCAT_AGG(distinct_tools_used) AS tools_used_by_turn,
      SUM(total_prompt_tokens) AS total_prompt_tokens,
      SUM(total_completion_tokens) AS total_completion_tokens,
      SUM(total_tokens) AS total_tokens,
      SUM(turn_latency_ms) AS total_latency_ms,
      LOGICAL_AND(session_turn_eval_ready) AS eval_ready,
      ARRAY_AGG(
        JSON_OBJECT(
          'turn_index', turn_index,
          'turn_id', invocation_id,
          'events', events_json
        )
        ORDER BY turn_index
      ) AS turn_traces
    FROM
      Turns
    GROUP BY
      session_id
  ),
  -- The rows of the eval session view, named by their output columns so that
  -- the where clause can filter them.
  EvalSessions AS (
    SELECT
      s.session_id,
      s.user_id,
      s.root_agent,
      s.session_start_timestamp,
      s.session_end_timestamp,
      s.turn_count,
      s.tool_call_count,
      s.tool_error_count,
      s.llm_error_count,
      s.agent_error_count,
      s.has_error,
      s.has_truncated_content,
      s.has_tool_loop,
      s.max_consecutive_identical_tool_calls,
      ARRAY(
        SELECT DISTINCT tool
        FROM UNNEST(s.tools_used_by_turn) AS tool
        ORDER BY tool
      ) AS distinct_tools_used,
      s.total_prompt_tokens,
      s.total_completion_tokens,
      s.total_tokens,
      s.total_latency_ms,
      s.eval_ready,
      JSON_OBJECT(
        'agents', COALESCE(m.agents, JSON_OBJECT()),
        'turns', TO_JSON(s.turn_traces)
      ) AS trace
    FROM
      SessionRollup AS s
    LEFT JOIN
      AgentMaps AS m
      ON m.session_id = s.session_id
  )
SELECT
  session_id,
  user_id,
  root_agent,
  session_start_timestamp,
  session_end_timestamp,
  turn_count,
  tool_call_count,
  tool_error_count,
  llm_error_count,
  agent_error_count,
  has_error,
  has_truncated_content,
  has_tool_loop,
  max_consecutive_identical_tool_calls,
  distinct_tools_used,
  total_prompt_tokens,
  total_completion_tokens,
  total_tokens,
  total_latency_ms,
  eval_ready,
  trace
FROM
  EvalSessions{where_clause}
