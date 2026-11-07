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

"""Builds the eval SQL of the BigQuery agent analytics plugin.

The builders fill the versioned templates of ``bigquery_agent_analytics_sql/``
with the standard library alone, so a caller that runs the eval queries needs
neither ``pyarrow`` nor ``google-cloud-bigquery-storage``, which the plugin's
writer imports and the ``bigquery-analytics`` extra installs.
``google.adk.plugins`` exports the two query builders, and
``bigquery_agent_analytics_plugin`` re-exports every name here, so code that
imports one from it keeps working.
"""

from __future__ import annotations

import collections.abc
import functools
import importlib.resources
import os
import pathlib
import re

# Payload columns eligible for physical projection.  Every other
# schema column is an identity / correlation / view-critical column and is
# *protected* — it cannot be projected out, because the BQAA execution tree
# and the per-event views depend on it.
_PROJECTABLE_PAYLOAD_COLUMNS = frozenset(
    {"content", "content_parts", "attributes", "latency_ms"}
)


def _validate_payload_column_denylist(
    denylist: list[str] | None,
) -> frozenset[str]:
  """Validates ``payload_column_denylist`` and returns the denied set.

  Only the projectable payload columns may be denied.  Anything else —
  an identity/correlation column or an unknown name — is a hard error,
  so a typo or an attempt to drop a join key fails loudly at construction
  rather than producing malformed rows or broken views.
  """
  denied = frozenset(denylist or ())
  invalid = denied - _PROJECTABLE_PAYLOAD_COLUMNS
  if invalid:
    raise ValueError(
        "payload_column_denylist may only contain projectable payload"
        f" columns {sorted(_PROJECTABLE_PAYLOAD_COLUMNS)}; got"
        f" {sorted(invalid)}. Identity/correlation columns (timestamp,"
        " event_type, session_id, invocation_id, trace_id, span_id,"
        " parent_span_id, is_truncated, ...) are protected and cannot be"
        " projected out."
    )
  return denied


# The eval SQL templates live in ``bigquery_agent_analytics_sql/``, the one
# copy of that SQL, which services that cannot import this module embed and
# run too.  Each file is a block of ``--`` comment lines (the license and the
# substitution contract), a blank line, and then the template, which has
# ``{name}`` placeholders and no doubled braces, so ``str.format`` and any
# other textual replacement give the same SQL.  The builders below load a
# template with ``_load_eval_sql_template``, which drops the header, and
# format it.  A view builder puts ``CREATE OR REPLACE VIEW`` before the SQL of
# its query builder, so that a view and the query that reads its rows share
# one template.
_EVAL_SQL_DIRNAME = "bigquery_agent_analytics_sql"


def _read_eval_sql_file(filename: str) -> str:
  """Reads a file of ``bigquery_agent_analytics_sql/`` beside this module.

  The directory is looked up beside this file's unresolved path first: a build
  may link this file and the templates into one tree, as Blaze does with
  runfiles that point into its content-addressed cache, and the target of a
  link has no neighbours there.  It is then looked up beside the resolved
  path, for a checkout that links this file alone, and last among the
  package's resources, for an install that keeps them in an archive.

  Args:
    filename: The bare file name.

  Returns:
    The text of the file.  It is read in text mode, which turns the CRLF line
    breaks of a Windows checkout into newlines.

  Raises:
    OSError: The file cannot be found or read.
  """
  module_path = pathlib.Path(os.path.abspath(__file__))
  for directory in (module_path.parent, module_path.resolve().parent):
    path = directory / _EVAL_SQL_DIRNAME / filename
    if path.is_file():
      return path.read_text(encoding="utf-8")
  resource = (
      importlib.resources.files(__package__ or "google.adk.plugins")
      .joinpath(_EVAL_SQL_DIRNAME)
      .joinpath(filename)
  )
  if resource.is_file():
    return resource.read_text(encoding="utf-8")
  raise FileNotFoundError(
      f"Cannot find {_EVAL_SQL_DIRNAME}/{filename} beside {__file__} or among"
      " the package's resources."
  )


@functools.lru_cache(maxsize=None)
def _load_eval_sql_template(filename: str) -> str:
  """Loads an eval SQL template of ``bigquery_agent_analytics_sql/``.

  A template file is a block of ``--`` comment lines (the license and the
  substitution contract), a blank line, and then the template.  The header is
  dropped: it names the placeholders, so it must not be formatted.  Each file
  is read once.

  Args:
    filename: The bare file name, such as ``eval_view_ctes_v1.sql``.

  Returns:
    The template text that follows the header.

  Raises:
    OSError: The file cannot be found or read.
    ValueError: ``filename`` is not a bare file name, or the file does not
      begin with a block of ``--`` comment lines and a blank line.
  """
  if (
      not filename
      or filename in {".", ".."}
      or pathlib.PurePath(filename).name != filename
  ):
    raise ValueError(f"filename must be a bare file name; got {filename!r}.")
  text = _read_eval_sql_file(filename)
  header, separator, template = text.partition("\n\n")
  header_lines = header.splitlines()
  if not (
      separator
      and header_lines
      and all(line.startswith("--") for line in header_lines)
  ):
    raise ValueError(
        f"{filename} must begin with a block of '--' comment lines (the"
        " license header) and a blank line."
    )
  return template


# A character that cannot be part of the table name that the eval SQL quotes in
# backticks: a backtick would end the name early, a backslash would start an
# escape sequence, and a control or line-break character would split it.  A
# space is legal: BigQuery allows one in a table name.
_EVAL_TABLE_ID_UNQUOTABLE_RE = re.compile(
    r"[`\\\x00-\x1f\x7f-\x9f\u2028\u2029]"
)


def _eval_view_dataset_ref(
    full_table_id: str,
    denied_columns: frozenset[str],
    view_prefix: str = "v",
) -> str:
  """Validates the eval view inputs and returns the table's dataset ref."""
  # A protected column such as is_truncated cannot be denied: read as FALSE,
  # it would mark truncated turns eval ready.
  _validate_payload_column_denylist(list(denied_columns))
  if "content" in denied_columns:
    raise ValueError(
        "Cannot build eval view SQL when 'content' is in"
        " payload_column_denylist."
    )
  dataset_ref, _, table = full_table_id.rpartition(".")
  if not dataset_ref or not table:
    raise ValueError(
        "full_table_id must be a dataset-qualified table name such as"
        f" 'project.dataset.table'; got {full_table_id!r}."
    )
  if _EVAL_TABLE_ID_UNQUOTABLE_RE.search(full_table_id):
    raise ValueError(
        "full_table_id must not contain a backtick, backslash, control or"
        " line-break character, because the SQL quotes it in backticks; got"
        f" {full_table_id!r}."
    )
  if not view_prefix or _EVAL_TABLE_ID_UNQUOTABLE_RE.search(view_prefix):
    raise ValueError(
        "view_prefix must be a non-empty string without a backtick, backslash,"
        f" control or line-break character; got {view_prefix!r}."
    )
  return dataset_ref


# The optional WHERE keyword at the start of a condition that a caller passes.
_EVAL_SQL_LEADING_WHERE_RE = re.compile(r"WHERE(?:\s+|\b)", re.IGNORECASE)
# The quote that opens a GoogleSQL string literal or quoted identifier, which
# runs to the same quote.  A triple quote is matched before a single one,
# which would read it as an empty string and a quote.
_EVAL_SQL_OPENING_QUOTE_RE = re.compile(r"'''|\"\"\"|['\"`]")


def _blank_eval_sql_comments(clause: str, param_name: str) -> str:
  """Returns a caller's condition with each GoogleSQL comment blanked out.

  ``--``, ``#`` and ``/* */`` comments become spaces, so the result is as long
  as ``clause`` and an offset in one is an offset in the other.  A string
  literal or quoted identifier is copied whole, so that a comment marker,
  semicolon or quote inside it is not read as one.  In it, a backslash and the
  character after it are skipped together: a quote after a backslash does not
  end a raw string either.

  Args:
    clause: A condition.
    param_name: The name of the argument that ``clause`` came from, for the
      error message.

  Returns:
    ``clause`` with spaces in place of its comments.

  Raises:
    ValueError: A block comment, string literal or quoted identifier in
      ``clause`` is not closed, so it would swallow the closing parenthesis
      that the templates put after the condition.
  """
  parts: list[str] = []
  start = 0
  while start < len(clause):
    if clause.startswith(("--", "#"), start):
      end = clause.find("\n", start)
      if end < 0:
        end = len(clause)
      parts.append(" " * (end - start))
    elif clause.startswith("/*", start):
      end = clause.find("*/", start + 2)
      if end < 0:
        raise ValueError(
            f"{param_name} has an unterminated block comment; got {clause!r}."
        )
      end += 2
      parts.append(" " * (end - start))
    else:
      end = start + 1
      opening = _EVAL_SQL_OPENING_QUOTE_RE.match(clause, start)
      if opening:
        quote = opening.group()
        end = opening.end()
        while not clause.startswith(quote, end):
          if end >= len(clause):
            token = "quoted identifier" if quote == "`" else "string literal"
            raise ValueError(
                f"{param_name} has an unterminated {token}; got {clause!r}."
            )
          end += 2 if clause[end] == "\\" else 1
        end += len(quote)
      parts.append(clause[start:end])
    start = end
  return "".join(parts)


def _normalize_eval_where_condition(clause: str, param_name: str) -> str:
  """Returns a caller's trusted condition without a leading ``WHERE``.

  Trailing semicolons are dropped too, even before a trailing comment: the
  templates put the condition in parentheses, where a semicolon is a syntax
  error.  The keyword and the semicolons are looked for outside comments,
  string literals and quoted identifiers, so one inside those stays.

  Args:
    clause: A condition, with or without a leading ``WHERE`` and trailing
      semicolons, or blank.
    param_name: The name of the argument that ``clause`` came from, for the
      error message.

  Returns:
    The condition, stripped, or ``""`` when ``clause`` is blank.

  Raises:
    ValueError: ``clause`` has no condition outside its comments besides a
      ``WHERE`` keyword and semicolons, or has a block comment, string literal
      or quoted identifier that is not closed.
  """
  if not clause.strip():
    return ""
  code = _blank_eval_sql_comments(clause, param_name)
  start = len(code) - len(code.lstrip())
  dropped: set[int] = set()
  match = _EVAL_SQL_LEADING_WHERE_RE.match(clause, start)
  if match:
    dropped.update(range(start, match.end()))
    start = match.end()
  end = len(code)
  while end > start and (code[end - 1].isspace() or code[end - 1] == ";"):
    end -= 1
    if code[end] == ";":
      dropped.add(end)
  if end == start:
    raise ValueError(f"{param_name} has no condition; got {clause!r}.")
  return "".join(
      char for index, char in enumerate(clause) if index not in dropped
  ).strip()


def _format_eval_raw_where_clause(raw_where_clause: str) -> str:
  """Formats ``raw_where_clause`` as the ``{raw_where_clause}`` of the CTEs.

  The closing parenthesis gets a line of its own, so that a trailing line
  comment in the condition cannot swallow it.
  """
  condition = _normalize_eval_where_condition(
      raw_where_clause, "raw_where_clause"
  )
  if not condition:
    return ""
  return f"\n      AND (\n        {condition}\n      )"


def _format_eval_outer_where_clause(where_clause: str) -> str:
  """Formats ``where_clause`` as the ``{where_clause}`` of a query template.

  The closing parenthesis gets a line of its own, so that a trailing line
  comment in the condition cannot swallow it.
  """
  condition = _normalize_eval_where_condition(where_clause, "where_clause")
  if not condition:
    return ""
  return f"\nWHERE\n  (\n    {condition}\n  )"


def _build_eval_view_ctes(
    full_table_id: str,
    denied_columns: frozenset[str],
    scope_columns: str,
    raw_where_clause: str = "",
) -> str:
  """Returns the CTE chain shared by the eval session and turn views.

  Args:
    full_table_id: The events table.
    denied_columns: Payload columns projected out of the table.  A denied
      ``content_parts``, ``attributes``, or ``latency_ms`` column reads as an
      empty array or NULL JSON, so the SQL never references it.
    scope_columns: The columns that scope ``trace.agents``: ``session_id`` for
      the session view, ``session_id, invocation_id`` for the turn view.
    raw_where_clause: A condition on the events table's columns, with or without
      a leading ``WHERE``, that narrows the rows every CTE reads, such as the
      lookback ``timestamp >= @lookback_start``, so that BigQuery can prune
      ``timestamp`` partitions.  It is trusted SQL: pass values as query
      parameters, never splice in untrusted text.  A session or turn with an
      event outside the condition comes out partial, and loses the agent
      configuration that its earlier events logged, so the lower bound must not
      be later than the first event of any session to read, and an upper bound,
      if there is one, must come after the last event of those sessions plus the
      delay before the plugin's writes are readable.  The end of the time window
      to read is too early a bound: it cuts off a session that is still running
      at that time, and the cut session passes for a finished one.  Never filter
      on a column that varies within a session, such as ``agent`` or
      ``event_type``.  To pick output rows, such as the sessions of a time
      window, use the ``where_clause`` of the query builders instead.

  Returns:
    The ``WITH`` clause, without a trailing newline.

  Raises:
    OSError: The template file cannot be found or read.
    ValueError: ``raw_where_clause`` has no condition outside its comments
      besides a ``WHERE`` keyword and semicolons, or has a block comment,
      string literal or quoted identifier that is not closed.
  """
  ctes = _load_eval_sql_template("eval_view_ctes_v1.sql").format(
      full_table_id=full_table_id,
      content_parts_column=(
          "ARRAY<STRUCT<part_index INT64, mime_type STRING, text STRING,"
          " part_attributes STRING, storage_mode STRING>>[] AS content_parts"
          if "content_parts" in denied_columns
          else "content_parts"
      ),
      attributes_column=(
          "CAST(NULL AS JSON) AS attributes"
          if "attributes" in denied_columns
          else "attributes"
      ),
      latency_ms_column=(
          "CAST(NULL AS JSON) AS latency_ms"
          if "latency_ms" in denied_columns
          else "latency_ms"
      ),
      scope_columns=scope_columns,
      raw_where_clause=_format_eval_raw_where_clause(raw_where_clause),
  )
  # The template ends in a newline, as its file does; the query templates
  # continue the chain after it.
  return ctes.rstrip("\n")


def _build_eval_session_view_sql(
    full_table_id: str,
    view_prefix: str = "v",
    denied_columns: frozenset[str] = frozenset(),
) -> str:
  """Builds the ``{view_prefix}_eval_session`` view DDL.

  The view has one row per session.  Its ``trace`` column is an ``AgentData``
  object holding every turn of the session, numbered from 0 in start order.
  It runs the query of ``build_eval_session_query_sql`` with no conditions.

  ``eval_ready`` is FALSE for a session that has ``has_truncated_content``,
  because a row that puts an event in its trace and is flagged
  ``is_truncated`` left the trace without part of a payload.  No other row
  counts, such as an ``LLM_REQUEST`` row, which repeats the conversation so
  far.  ``has_truncated_content`` tells a caller that truncation keeps a
  session from being ready, but not whether it is the only reason: no column
  of the view holds the other conditions of ``eval_ready``, so a truncated
  session may also have a turn that is unfinished or lacks user text.

  Args:
    full_table_id: The events table as ``project.dataset.table``.  The view is
      created in the same dataset.
    view_prefix: Prefix for the view name.
    denied_columns: Payload columns projected out of the table.

  Returns:
    A ``CREATE OR REPLACE VIEW`` statement.

  Raises:
    ValueError: ``denied_columns`` holds ``content`` or a column that is not a
      projectable payload column, such as the protected ``is_truncated``,
      ``full_table_id`` has no dataset or has a backtick, backslash, control
      or line-break character, or ``view_prefix`` is empty or has one of those
      characters.
  """
  dataset_ref = _eval_view_dataset_ref(
      full_table_id, denied_columns, view_prefix
  )
  view_ref = f"{dataset_ref}.{view_prefix}_eval_session"
  query_sql = build_eval_session_query_sql(
      full_table_id, denied_columns=denied_columns
  )
  return f"CREATE OR REPLACE VIEW `{view_ref}` AS\n{query_sql}"


def _build_eval_turn_view_sql(
    full_table_id: str,
    view_prefix: str = "v",
    denied_columns: frozenset[str] = frozenset(),
) -> str:
  """Builds the ``{view_prefix}_eval_turn`` view DDL.

  The view has one row per ``(session_id, invocation_id)`` turn.  Its
  ``trace`` column is an ``AgentData`` object whose ``turns`` array holds only
  that turn, with ``turn_index`` 0 as single-turn metrics require; the
  ``turn_index`` column keeps the turn's position in its session.  It runs
  the query of ``build_eval_turn_query_sql`` with no conditions.

  ``eval_ready`` is FALSE for a turn that has ``has_truncated_content``,
  because a row that puts an event in its trace and is flagged
  ``is_truncated`` left the trace without part of a payload.  No other row
  counts, such as an ``LLM_REQUEST`` row, which repeats the conversation so
  far.  ``has_truncated_content`` tells a caller that truncation keeps a turn
  from being ready, but not whether it is the only reason: no column of the
  view holds the other conditions of ``eval_ready``, so a truncated turn may
  also be unfinished or lack user text.

  Args:
    full_table_id: The events table as ``project.dataset.table``.  The view is
      created in the same dataset.
    view_prefix: Prefix for the view name.
    denied_columns: Payload columns projected out of the table.

  Returns:
    A ``CREATE OR REPLACE VIEW`` statement.

  Raises:
    ValueError: ``denied_columns`` holds ``content`` or a column that is not a
      projectable payload column, such as the protected ``is_truncated``,
      ``full_table_id`` has no dataset or has a backtick, backslash, control
      or line-break character, or ``view_prefix`` is empty or has one of those
      characters.
  """
  dataset_ref = _eval_view_dataset_ref(
      full_table_id, denied_columns, view_prefix
  )
  view_ref = f"{dataset_ref}.{view_prefix}_eval_turn"
  query_sql = build_eval_turn_query_sql(
      full_table_id, denied_columns=denied_columns
  )
  return f"CREATE OR REPLACE VIEW `{view_ref}` AS\n{query_sql}"


def build_eval_session_query_sql(
    full_table_id: str,
    *,
    denied_columns: collections.abc.Collection[str] | None = (),
    where_clause: str = "",
    raw_where_clause: str = "",
) -> str:
  """Builds a query that reads the rows of the ``eval_session`` view.

  A job that runs this query reads the view's rows without creating a view.
  ``raw_where_clause`` narrows the events that the query reads, so that
  BigQuery can prune ``timestamp`` partitions, and ``where_clause`` then picks
  the sessions to return.  The query declares no parameters of its own; the
  job supplies any ``@name`` parameter that the two conditions use.  It ends
  without a semicolon, so the job can append ``ORDER BY`` or ``LIMIT`` to it,
  or read it as a subquery.

  As in the view, ``eval_ready`` is FALSE for a session that has
  ``has_truncated_content``, because a row that puts an event in its trace and
  is flagged ``is_truncated`` left the trace without part of a payload.  No
  other row counts, such as an ``LLM_REQUEST`` row, which repeats the
  conversation so far.  ``has_truncated_content`` tells a caller that
  truncation keeps a session from being ready, but not whether it is the only
  reason: no output column holds the other conditions of ``eval_ready``, so a
  truncated session may also have a turn that is unfinished or lacks user
  text.

  Args:
    full_table_id: The events table as ``project.dataset.table``.
    denied_columns: Payload columns projected out of the table, in any
      collection, such as the ``payload_column_denylist`` of a
      ``BigQueryLoggerConfig``.  ``None``, the default of that list, denies
      none.
    where_clause: A trusted condition on the output columns, with or without a
      leading ``WHERE``, that picks the sessions to return, such as
      ``session_end_timestamp >= @window_start AND session_end_timestamp <
      @window_end AND eval_ready``.  It filters whole sessions after they are
      rebuilt, so it drops no event that a session needs.
    raw_where_clause: A trusted condition on the events table, with or without a
      leading ``WHERE``, that lets BigQuery prune ``timestamp`` partitions, such
      as the lookback ``timestamp >= @lookback_start``.  A session that loses an
      event to it comes out partial, so its lower bound must not be later than
      the first event of any session to read, and an upper bound, if it has one,
      must come after the last event of those sessions plus the delay before the
      plugin's writes are readable.  The end of the time window that
      ``where_clause`` picks is too early a bound: it cuts off a session that is
      still running at that time, and the cut session passes for a finished one.
      Never filter on a column that varies within a session, such as ``agent``
      or ``event_type``.

  Returns:
    A ``WITH ... SELECT`` query, without a trailing semicolon.

  Raises:
    ValueError: ``denied_columns`` holds ``content`` or a column that is not a
      projectable payload column, such as the protected ``is_truncated``;
      ``full_table_id`` has no dataset or has a backtick, backslash, control
      or line-break character; or ``where_clause`` or ``raw_where_clause`` has
      no condition outside its comments besides a ``WHERE`` keyword and
      semicolons, or has a block comment, string literal or quoted identifier
      that is not closed.
  """
  denied = frozenset(denied_columns or ())
  _eval_view_dataset_ref(full_table_id, denied)
  return _load_eval_sql_template("eval_session_query_v1.sql").format(
      ctes=_build_eval_view_ctes(
          full_table_id, denied, "session_id", raw_where_clause
      ),
      where_clause=_format_eval_outer_where_clause(where_clause),
  )


def build_eval_turn_query_sql(
    full_table_id: str,
    *,
    denied_columns: collections.abc.Collection[str] | None = (),
    where_clause: str = "",
    raw_where_clause: str = "",
) -> str:
  """Builds a query that reads the rows of the ``eval_turn`` view.

  A job that runs this query reads the view's rows without creating a view.
  ``raw_where_clause`` narrows the events that the query reads, so that
  BigQuery can prune ``timestamp`` partitions, and ``where_clause`` then picks
  the turns to return.  The query declares no parameters of its own; the job
  supplies any ``@name`` parameter that the two conditions use.  It ends
  without a semicolon, so the job can append ``ORDER BY`` or ``LIMIT`` to it,
  or read it as a subquery.

  As in the view, ``eval_ready`` is FALSE for a turn that has
  ``has_truncated_content``, because a row that puts an event in its trace and
  is flagged ``is_truncated`` left the trace without part of a payload.  No
  other row counts, such as an ``LLM_REQUEST`` row, which repeats the
  conversation so far.  ``has_truncated_content`` tells a caller that
  truncation keeps a turn from being ready, but not whether it is the only
  reason: no output column holds the other conditions of ``eval_ready``, so a
  truncated turn may also be unfinished or lack user text.

  Args:
    full_table_id: The events table as ``project.dataset.table``.
    denied_columns: Payload columns projected out of the table, in any
      collection, such as the ``payload_column_denylist`` of a
      ``BigQueryLoggerConfig``.  ``None``, the default of that list, denies
      none.
    where_clause: A trusted condition on the output columns, with or without a
      leading ``WHERE``, that picks the turns to return, such as
      ``turn_end_timestamp >= @window_start AND turn_end_timestamp < @window_end
      AND eval_ready``.  It filters whole turns after they are rebuilt, so it
      drops no event that a turn needs: a turn keeps its ``turn_index`` and the
      system prompt that its session logged first.
    raw_where_clause: A trusted condition on the events table, with or without a
      leading ``WHERE``, that lets BigQuery prune ``timestamp`` partitions, such
      as the lookback ``timestamp >= @lookback_start``.  A session that loses an
      event to it comes out partial, so its lower bound must not be later than
      the first event of any session to read, and an upper bound, if it has one,
      must come after the last event of those sessions plus the delay before the
      plugin's writes are readable.  The end of the time window that
      ``where_clause`` picks is too early a bound: it cuts off a session that is
      still running at that time, and the cut session passes for a finished one.
      Never filter on a column that varies within a session, such as ``agent``
      or ``event_type``.

  Returns:
    A ``WITH ... SELECT`` query, without a trailing semicolon.

  Raises:
    ValueError: ``denied_columns`` holds ``content`` or a column that is not a
      projectable payload column, such as the protected ``is_truncated``;
      ``full_table_id`` has no dataset or has a backtick, backslash, control
      or line-break character; or ``where_clause`` or ``raw_where_clause`` has
      no condition outside its comments besides a ``WHERE`` keyword and
      semicolons, or has a block comment, string literal or quoted identifier
      that is not closed.
  """
  denied = frozenset(denied_columns or ())
  _eval_view_dataset_ref(full_table_id, denied)
  return _load_eval_sql_template("eval_turn_query_v1.sql").format(
      ctes=_build_eval_view_ctes(
          full_table_id,
          denied,
          "session_id, invocation_id",
          raw_where_clause,
      ),
      where_clause=_format_eval_outer_where_clause(where_clause),
  )
