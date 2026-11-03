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
-- Eval turn view of the BigQuery Agent Analytics plugin: one row per
-- (session_id, invocation_id) turn, with the turn's AgentData trace and its
-- metrics.  An agent's instruction in a turn comes from its latest
-- AGENT_STARTING event in that turn, and falls back to the first system prompt
-- that its session logged for the agent.
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
-- {view_ref}
--     The view to create, as project.dataset.view_name.  The template wraps it
--     in backticks, so it has no backtick, backslash, control or line-break
--     character.
-- {ctes}
--     The text of eval_view_ctes_v1.sql after its header, with its
--     placeholders replaced and session_id, invocation_id as its scope
--     columns: a WITH clause that ends after its last CTE, with no trailing
--     comma, which this template continues.  That file lists its placeholders.
--
-- Versioning: the _v1 in the file name is the version of this contract.  A
-- backward-compatible fix, which keeps the placeholders and the output
-- columns, edits the file in place; a change to either needs a new _v2 file.

CREATE OR REPLACE VIEW `{view_ref}` AS
{ctes}
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
  ON m.session_id = t.session_id AND m.invocation_id = t.invocation_id;
