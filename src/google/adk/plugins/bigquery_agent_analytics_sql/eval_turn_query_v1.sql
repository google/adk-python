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
-- Eval turn query of the BigQuery Agent Analytics plugin: the rows of the eval
-- turn view, for a job that reads them without creating the view and so can
-- narrow both the events that it reads and the rows that it returns.  An
-- agent's instruction in a turn comes from its latest AGENT_STARTING event in
-- that turn, and falls back to the first system prompt that its session
-- logged for the agent.  The query ends without a semicolon, so the job can
-- append ORDER BY or LIMIT to it, or read it as a subquery.
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
--     placeholders replaced and session_id, invocation_id as its scope
--     columns: a WITH clause that ends after its last CTE, with no trailing
--     comma, which this template continues.  That file lists its placeholders
--     and the rules for a condition that narrows the events table, whose
--     query parameters the running job supplies.
-- {where_clause}
--     Empty, or a trusted condition on the output columns that picks the rows
--     to return, such as the turns that ended in a time window.  It filters
--     whole turns after they are rebuilt, so it drops no event that a turn it
--     keeps needs.  The template puts it right after the last source, so it
--     starts with a newline and has this shape, here indented, with WHERE at
--     the start of its line and the closing parenthesis on a line of its own
--     so that a trailing line comment in the condition cannot swallow it:
--
--       WHERE
--         (
--           condition
--         )
--
--     The condition is SQL: bind its values as query parameters that the
--     running job supplies, and never splice in untrusted text.
--
-- View: the plugin's eval turn view runs this query with no conditions.
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
-- TRUE for its turn, and eval_ready is FALSE.  No other row counts, such as
-- an LLM_REQUEST row, which repeats the conversation so far.
-- has_truncated_content tells a caller that truncation keeps a turn from
-- being ready, but not whether it is the only reason: no output column holds
-- the other conditions of eval_ready, so a truncated turn may also be
-- unfinished or lack user text.
--
-- Versioning: the _v1 in the file name is the version of this contract.  A
-- backward-compatible fix, which keeps the placeholders and the output
-- columns, edits the file in place; a change to either needs a new _v2 file.

{ctes},
  -- The rows of the eval turn view, named by their output columns so that the
  -- where clause can filter them.
  EvalTurns AS (
    SELECT
      t.session_id,
      t.invocation_id,
      t.turn_index,
      t.user_id,
      t.root_agent,
      t.turn_start_timestamp,
      t.turn_end_timestamp,
      t.tool_call_count,
      t.tool_error_count,
      t.llm_error_count,
      t.agent_error_count,
      t.has_error,
      t.has_truncated_content,
      t.has_tool_loop,
      t.max_consecutive_identical_tool_calls,
      t.distinct_tools_used,
      t.total_prompt_tokens,
      t.total_completion_tokens,
      t.total_tokens,
      t.turn_latency_ms,
      t.eval_ready,
      JSON_OBJECT(
        'agents', COALESCE(m.agents, JSON_OBJECT()),
        'turns', JSON_ARRAY(
          JSON_OBJECT(
            'turn_index', 0,
            'turn_id', t.invocation_id,
            'events', t.events_json
          )
        )
      ) AS trace
    FROM
      Turns AS t
    LEFT JOIN
      AgentMaps AS m
      ON m.session_id = t.session_id AND m.invocation_id = t.invocation_id
  )
SELECT
  session_id,
  invocation_id,
  turn_index,
  user_id,
  root_agent,
  turn_start_timestamp,
  turn_end_timestamp,
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
  turn_latency_ms,
  eval_ready,
  trace
FROM
  EvalTurns{where_clause}
