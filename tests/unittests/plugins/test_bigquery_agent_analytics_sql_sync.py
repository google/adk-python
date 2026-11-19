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

"""Checks the eval SQL templates of the BigQuery agent analytics plugin.

The plugin loads each eval SQL template from its ``.sql`` file, and services
that cannot import the plugin run the same files.  These tests read the files,
and the source of the plugin's module that loads them, as text, so they need
neither ADK nor its runtime dependencies.  They check the header of each file,
which holds the license and the substitution contract, the placeholders of
each template, the rules of the SQL lint and of the SQL style guide that a
template must keep, and a few choices in the SQL that a consumer relies on.
They live apart from the plugin's unit tests so that they can run without
importing the plugin.
"""

from __future__ import annotations

import os
import pathlib
import re
import string

import pytest

# The plugin's module that loads the templates, beside its SQL directory.
_LOADER_FILE = "_bigquery_agent_analytics_eval_sql.py"
_SQL_DIR = "bigquery_agent_analytics_sql"
_CTE_FILE = "eval_view_ctes_v1.sql"

# The placeholders that the plugin formats each template with.
_PLACEHOLDERS_BY_FILE = {
    _CTE_FILE: {
        "full_table_id",
        "content_parts_column",
        "attributes_column",
        "latency_ms_column",
        "scope_columns",
        "raw_where_clause",
    },
    "eval_session_query_v1.sql": {"ctes", "where_clause"},
    "eval_turn_query_v1.sql": {"ctes", "where_clause"},
}
_ALL_FILES = tuple(_PLACEHOLDERS_BY_FILE)
# Each query file, and the CTE of output rows that its final SELECT reads.
_SOURCE_BY_QUERY_FILE = {
    "eval_session_query_v1.sql": "EvalSessions",
    "eval_turn_query_v1.sql": "EvalTurns",
}

# What every header says about the substitution contract.
_CONTRACT_PHRASES = (
    "Substitution contract",
    "strip this header",
    "doubled braces",
    "declares no query parameters",
    "_v2",
)

# A token of SQL.  A comment has no group, so a scan skips it; a quoted name or
# string is one token, so the words inside it are not seen.
_SQL_TOKEN_RE = re.compile(
    r"--[^\n]*|/\*.*?\*/"
    r"|(?P<quoted>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`[^`]*`)"
    r"|(?P<word>[A-Za-z_]\w*)"
    r"|(?P<symbol>\S)",
    re.DOTALL,
)
# The words that may come before JOIN.
_JOIN_TYPES = frozenset({"INNER", "LEFT", "RIGHT", "FULL", "CROSS", "OUTER"})
# The words that end the sources of a FROM clause.
_FROM_CLAUSE_ENDS = frozenset({
    "WHERE",
    "GROUP",
    "HAVING",
    "QUALIFY",
    "WINDOW",
    "ORDER",
    "LIMIT",
    "SELECT",
    "UNION",
    "INTERSECT",
    "EXCEPT",
})
# The name of a CTE, at the start of the line that begins its definition.
_CTE_NAME_RE = re.compile(r"^ *(\w+) AS \($", re.MULTILINE)


def _search_roots() -> list[pathlib.Path]:
  """The directories that may hold the plugin's directory, nearest first.

  The plugin sits beside the directories that hold this file, so the walk starts
  from this file's own path.  It does not stop at a symbolic link: a test runner
  may link this file and the plugin into one tree, and the link's target then
  has no neighbours.  A checkout may instead link the test directory, so the
  target's directories follow, and the working directory comes last.
  """
  here = pathlib.Path(__file__)
  cwd = pathlib.Path.cwd()
  roots = [
      *pathlib.Path(os.path.abspath(here)).parents,
      *here.resolve().parents,
      cwd,
      *cwd.parents,
  ]
  return list(dict.fromkeys(roots))


@pytest.fixture(scope="module")
def plugins_dir() -> pathlib.Path:
  """The directory that holds the plugin's SQL module and its SQL directory."""
  for root in _search_roots():
    for candidate in (
        root / "plugins",
        root / "src" / "google" / "adk" / "plugins",
    ):
      if (candidate / _LOADER_FILE).is_file():
        return candidate
  raise FileNotFoundError(
      f"Cannot find {_LOADER_FILE} above {__file__} or the working directory"
      f" {pathlib.Path.cwd()}."
  )


@pytest.fixture(scope="module")
def file_texts(plugins_dir: pathlib.Path) -> dict[str, str]:
  """The full text of every template file, by file name."""
  return {
      name: (plugins_dir / _SQL_DIR / name).read_bytes().decode("utf-8")
      for name in _ALL_FILES
  }


@pytest.fixture(scope="module")
def bodies(file_texts: dict[str, str]) -> dict[str, str]:
  """The template of every file, after its header, by file name."""
  return {name: _split_header(text)[1] for name, text in file_texts.items()}


def _split_header(text: str) -> tuple[str, str]:
  """Splits a template file into its header and the template.

  The header is everything up to the first blank line.  The plugin loads a file
  by the same rule; it is not imported here, so the rule is repeated.
  """
  header, _, body = text.partition("\n\n")
  return header, body


def _header_prose(header: str) -> str:
  """The text of a header's comment lines, with the line breaks collapsed."""
  return re.sub(r"\s+", " ", re.sub(r"(?m)^--", "", header))


def _squash(sql: str) -> str:
  """Drops comments and collapses whitespace, so a check ignores line breaks."""
  sql = _SQL_TOKEN_RE.sub(
      lambda match: " " if match.lastgroup is None else match.group(), sql
  )
  sql = re.sub(r"\s+", " ", sql)
  return sql.replace("( ", "(").replace(" )", ")")


def _case_and_join_problems(sql: str) -> list[str]:
  """Lists each CASE without an ELSE and each join that does not name its type.

  The SQL lint and style guide of the repository require both: a CASE that
  falls through returns NULL without saying so, and a bare JOIN, or a comma
  between the sources of a FROM clause, does not say which join it is.  The
  scan skips comments and quoted text, a CASE inside a CASE needs an ELSE of
  its own, and a comma inside parentheses, as in an argument list, is not a
  join.

  Args:
    sql: The text of a SQL file.

  Returns:
    One line per problem, naming the line of the file.  Empty when there are
    none.
  """

  def line_of(match: re.Match[str]) -> int:
    return sql.count("\n", 0, match.start()) + 1

  problems: list[str] = []
  # Each open CASE, innermost last: whether it has an ELSE, and its line.
  open_cases: list[tuple[bool, int]] = []
  # For the top level and each open parenthesis, innermost last: whether the
  # scan is among the sources of a FROM clause at that level.
  in_from = [False]
  previous = ""
  for match in _SQL_TOKEN_RE.finditer(sql):
    if match.lastgroup is None:
      continue
    symbol = match.group() if match.lastgroup == "symbol" else ""
    word = match.group().upper() if match.lastgroup == "word" else ""
    if symbol == "(":
      in_from.append(False)
    elif symbol == ")" and len(in_from) > 1:
      in_from.pop()
    elif symbol == "," and in_from[-1]:
      problems.append(f"line {line_of(match)}: comma join without a JOIN type")
    elif word == "FROM":
      in_from[-1] = True
    elif word in _FROM_CLAUSE_ENDS:
      in_from[-1] = False
    elif word == "CASE":
      open_cases.append((False, line_of(match)))
    elif word == "ELSE" and open_cases:
      open_cases[-1] = (True, open_cases[-1][1])
    elif word == "END" and open_cases:
      has_else, case_line = open_cases.pop()
      if not has_else:
        problems.append(f"line {case_line}: CASE without ELSE")
    elif word == "JOIN" and previous not in _JOIN_TYPES:
      problems.append(
          f"line {line_of(match)}: JOIN without INNER, LEFT, RIGHT or CROSS"
      )
    previous = word
  problems.extend(f"line {line}: CASE without END" for _, line in open_cases)
  return problems


def _countif_tests(sql: str) -> list[str]:
  """Lists each COUNTIF that is compared with 0 or 1 to test for any row.

  LOGICAL_OR says that directly, as the style guide asks.  The scan reads the
  SQL without its comments.

  Args:
    sql: The text of a SQL file.

  Returns:
    Each COUNTIF and its comparison.  Empty when there are none.
  """
  sql = _squash(sql)
  found: list[str] = []
  for match in re.finditer(r"\bCOUNTIF\(", sql):
    depth, end = 1, match.end()
    while depth and end < len(sql):
      depth += {"(": 1, ")": -1}.get(sql[end], 0)
      end += 1
    comparison = re.match(r" ?(?:[<>]=?|!=|<>|=) ?[01]\b", sql[end:])
    if comparison:
      found.append(sql[match.start() : end + comparison.end()])
  return found


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_file_begins_with_license_header_and_blank_line(
    file_texts: dict[str, str], filename: str
) -> None:
  """Each file opens with a block of '--' lines that starts with the license."""
  header, body = _split_header(file_texts[filename])

  header_lines = header.split("\n")
  assert re.match(r"^-- Copyright \d{4} Google LLC$", header_lines[0])
  assert "-- limitations under the License." in header_lines
  assert all(line.startswith("--") for line in header_lines)
  assert body.strip() and not body.startswith("\n")


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_file_ends_with_exactly_one_newline(
    file_texts: dict[str, str], filename: str
) -> None:
  """Each file ends with one newline, as the repository's formatters require."""
  text = file_texts[filename]

  assert text.endswith("\n")
  assert not text.endswith("\n\n")


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_file_has_no_trailing_whitespace_or_carriage_returns(
    file_texts: dict[str, str], filename: str
) -> None:
  """No line of a file ends in whitespace or a carriage return."""
  lines = file_texts[filename].split("\n")

  offenders = [n for n, line in enumerate(lines, 1) if line != line.rstrip()]

  assert not offenders, f"Trailing whitespace on lines {offenders}."


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_file_has_no_doubled_braces(
    file_texts: dict[str, str], filename: str
) -> None:
  """A file has no '{{' or '}}', so a plain text replacement can fill it."""
  lines = file_texts[filename].split("\n")

  offenders = [
      n for n, line in enumerate(lines, 1) if "{{" in line or "}}" in line
  ]

  assert not offenders, f"Doubled braces on lines {offenders}."


@pytest.mark.parametrize("filename, expected", _PLACEHOLDERS_BY_FILE.items())
def test_file_names_exactly_the_placeholders_the_plugin_fills(
    bodies: dict[str, str], filename: str, expected: set[str]
) -> None:
  """A template has the placeholders the plugin formats it with, and no more."""
  fields = list(string.Formatter().parse(bodies[filename]))

  names = {name for _, name, _, _ in fields if name is not None}

  assert names == expected
  assert all(
      spec == "" and conversion is None
      for _, name, spec, conversion in fields
      if name is not None
  )


@pytest.mark.parametrize("filename, expected", _PLACEHOLDERS_BY_FILE.items())
def test_header_names_exactly_the_placeholders_of_its_template(
    file_texts: dict[str, str], filename: str, expected: set[str]
) -> None:
  """A header lists the placeholders its template has, each in single braces."""
  header, _ = _split_header(file_texts[filename])

  names = set(re.findall(r"\{(\w+)\}", header))

  assert names == expected
  assert len(re.findall(r"[{}]", header)) == 2 * len(
      re.findall(r"\{\w+\}", header)
  ), "The header has a brace that is not part of a {placeholder}."


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_header_states_the_substitution_contract(
    file_texts: dict[str, str], filename: str
) -> None:
  """A header tells a consumer to strip it, and what the template promises."""
  prose = _header_prose(_split_header(file_texts[filename])[0])

  for phrase in _CONTRACT_PHRASES:
    assert phrase in prose, f"The header of {filename} lacks {phrase!r}."


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_file_declares_no_query_parameters(
    bodies: dict[str, str], filename: str
) -> None:
  """Every query parameter comes from the caller, never from a template."""
  assert "@" not in bodies[filename]


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_file_gives_every_case_an_else_and_every_join_a_type(
    file_texts: dict[str, str], filename: str
) -> None:
  """Every CASE has an ELSE and every join names its type, as the lint needs."""
  assert _case_and_join_problems(file_texts[filename]) == []


@pytest.mark.parametrize(
    "sql, problem_count",
    [
        ("SELECT CASE WHEN a THEN CASE WHEN b THEN 1 ELSE 2 END ELSE 3 END", 0),
        ("SELECT * FROM t INNER JOIN UNNEST(arr) AS x WITH OFFSET AS i", 0),
        ("SELECT * FROM t LEFT OUTER JOIN u USING (k, j)", 0),
        (
            (
                "SELECT a, f(b, c) FROM t WHERE d IN (1, 2) GROUP BY a, b"
                " ORDER BY a, b"
            ),
            0,
        ),
        ("SELECT (SELECT x FROM u), y FROM t", 0),
        (
            "-- CASE JOIN\nSELECT 'CASE', \"JOIN\", `CASE`, turn_end_timestamp",
            0,
        ),
        ("SELECT /* CASE JOIN */ 1", 0),
        ("SELECT CASE WHEN a THEN 1 END", 1),
        ("SELECT CASE WHEN a THEN CASE WHEN b THEN 1 END ELSE 3 END", 1),
        ("SELECT CASE WHEN a THEN 1 ELSE CASE WHEN b THEN 2 END END", 1),
        ("SELECT * FROM t JOIN u USING (k)", 1),
        ("SELECT * FROM t, UNNEST(arr) AS x", 1),
        ("SELECT * FROM (SELECT * FROM t, u)", 1),
        ("SELECT CASE WHEN a THEN 1", 1),
    ],
    ids=[
        "nested-case-with-else",
        "inner-join-unnest",
        "left-outer-join",
        "commas-outside-from",
        "comma-after-subquery",
        "comments-quotes-and-longer-names",
        "block-comment",
        "case-without-else",
        "inner-case-without-else",
        "case-in-else-without-else",
        "bare-join",
        "comma-join",
        "comma-join-in-subquery",
        "case-without-end",
    ],
)
def test_case_and_join_scan_reports_what_the_lint_would(
    sql: str, problem_count: int
) -> None:
  """The scan reports each CASE and join problem, and nothing else."""
  assert len(_case_and_join_problems(sql)) == problem_count


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_file_names_its_ctes_in_pascal_case(
    bodies: dict[str, str], filename: str
) -> None:
  """Each CTE has a PascalCase name, as the SQL style guide asks."""
  names = _CTE_NAME_RE.findall(bodies[filename])

  assert [
      name for name in names if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", name)
  ] == []


def test_cte_name_scan_finds_each_cte() -> None:
  """The scan sees every CTE, so that the PascalCase check reads them all."""
  sql = (
      "WITH\n  raw_events AS (\n    SELECT 1\n  ),\n"
      "  Turns AS (\n    SELECT 2 AS n\n  )\nSELECT n FROM Turns;\n"
  )

  assert _CTE_NAME_RE.findall(sql) == ["raw_events", "Turns"]


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_file_tests_for_any_row_with_logical_or(
    bodies: dict[str, str], filename: str
) -> None:
  """No COUNTIF is compared with 0 or 1; LOGICAL_OR tests for any row."""
  assert _countif_tests(bodies[filename]) == []


@pytest.mark.parametrize(
    "sql, test_count",
    [
        ("SELECT COUNTIF(a) AS n", 0),
        ("SELECT COUNTIF(a) > 10", 0),
        ("SELECT COUNTIF(a) > 0", 1),
        ("SELECT COUNTIF(f(a, b) = 'x')\n  = 0", 1),
        ("SELECT COUNTIF(a) >= 1 AND COUNTIF(b) != 0", 2),
    ],
)
def test_countif_scan_finds_each_test_for_any_row(
    sql: str, test_count: int
) -> None:
  """The scan finds each COUNTIF that a comparison turns into a test."""
  assert len(_countif_tests(sql)) == test_count


@pytest.mark.parametrize("filename", _ALL_FILES)
def test_file_has_no_semicolon(bodies: dict[str, str], filename: str) -> None:
  """No template ends a statement, so a caller can extend a query or wrap it.

  The shared CTEs begin a query that a query template continues, and a query
  template ends without a ';', so that a caller can append ORDER BY or LIMIT
  to the query, or read it as a subquery.
  """
  assert ";" not in bodies[filename]


def test_plugin_loads_each_packaged_template(plugins_dir: pathlib.Path) -> None:
  """The SQL module loads each .sql file of its SQL directory and no other."""
  source = (plugins_dir / _LOADER_FILE).read_text(encoding="utf-8")

  loaded = set(
      re.findall(r'_load_eval_sql_template\(\s*"([^"]+)"\s*\)', source)
  )
  packaged = {path.name for path in (plugins_dir / _SQL_DIR).glob("*.sql")}

  assert loaded == packaged == set(_ALL_FILES)


def test_events_table_is_read_once_for_the_events_of_a_turn(
    bodies: dict[str, str],
) -> None:
  """One CTE reads the table, and only events with a session and a turn.

  The caller's condition is added to that filter.
  """
  sql = _squash(bodies[_CTE_FILE])

  assert sql.count("`{full_table_id}`") == 1
  assert (
      "FROM `{full_table_id}` WHERE session_id IS NOT NULL"
      " AND invocation_id IS NOT NULL{raw_where_clause}),"
  ) in sql


def test_instruction_falls_back_to_the_logged_system_prompt(
    bodies: dict[str, str],
) -> None:
  """An agent's instruction is its AGENT_STARTING text, else its prompt."""
  sql = _squash(bodies[_CTE_FILE])

  assert (
      "'instruction', COALESCE(NULLIF(instruction, ''), logged_system_prompt,"
      " '')"
  ) in sql


@pytest.mark.parametrize("filename, source", _SOURCE_BY_QUERY_FILE.items())
def test_query_where_clause_filters_the_output_rows_last(
    bodies: dict[str, str], filename: str, source: str
) -> None:
  """A query's where_clause follows the CTE of output rows that it reads."""
  assert bodies[filename].endswith(f"\nFROM\n  {source}{{where_clause}}\n")


def _ctes_by_name(sql: str) -> dict[str, str]:
  """Slices a CTE template body into each CTE's text, keyed by CTE name."""
  matches = list(_CTE_NAME_RE.finditer(sql))
  return {
      match.group(1): sql[
          match.start() : (
              matches[i + 1].start() if i + 1 < len(matches) else len(sql)
          )
      ]
      for i, match in enumerate(matches)
  }


def test_session_events_excludes_sub_agent_sessions(
    bodies: dict[str, str],
) -> None:
  """SessionEvents filters parent_session_id as the sole RawEvents reader."""
  ctes = _ctes_by_name(bodies[_CTE_FILE])
  squashed_body = _squash(bodies[_CTE_FILE])

  assert (
      "SessionEvents AS (SELECT * FROM RawEvents"
      " WHERE JSON_VALUE(attributes, '$.adk.parent_session_id') IS NULL)"
  ) in _squash(ctes["SessionEvents"])
  assert re.findall(r"\bFROM RawEvents\b", squashed_body) == ["FROM RawEvents"]


@pytest.mark.parametrize(
    "cte_name",
    ["TypedEvents", "AgentRows", "SessionAgentPrompts"],
)
def test_downstream_ctes_read_from_session_events(
    bodies: dict[str, str],
    cte_name: str,
) -> None:
  """Downstream CTEs read from SessionEvents instead of RawEvents."""
  ctes = _ctes_by_name(bodies[_CTE_FILE])
  cte_sql = _squash(ctes[cte_name])
  assert "FROM SessionEvents" in cte_sql
  assert "FROM RawEvents" not in cte_sql


def test_formatted_response_and_response_text_delimiters(
    bodies: dict[str, str],
) -> None:
  """TypedEvents reads string content and ResponseEvents anchors delimiters."""
  ctes = _ctes_by_name(bodies[_CTE_FILE])
  typed_sql = _squash(ctes["TypedEvents"])
  response_sql = ctes["ResponseEvents"]

  assert (
      "IF(TRIM(JSON_VALUE(content, '$.response')) != '',"
      " JSON_VALUE(content, '$.response'), NULL)"
  ) in typed_sql
  sentinel_sql = (
      "'[FORMATTER_FAILED]', '[FORMATTING FAILED]', '[REDACTED]',"
      " '[CONTENT_PARSE_FAILED]', '[UNPARSEABLE_JSON_BLOB]',"
      " '[UNSUPPORTED_OBJECT]', '[REDACTED_SENSITIVE_TEXT]'"
  )
  assert (
      "IF(JSON_TYPE(content) = 'string'"
      " AND TRIM(JSON_VALUE(content, '$')) != ''"
      f" AND JSON_VALUE(content, '$') NOT IN ({sentinel_sql}),"
      " JSON_VALUE(content, '$'), NULL)"
  ) in typed_sql

  squashed_response_sql = _squash(response_sql)
  assert "TRIM(REGEXP_REPLACE(" in squashed_response_sql
  assert ")) != ''" in squashed_response_sql
  assert f"CHR(10)) NOT IN ({sentinel_sql})" in squashed_response_sql
  assert f"formatted_response NOT IN ({sentinel_sql})" in squashed_response_sql

  assert (
      'r"\' [|] (?:call: |resp: |Executable code |Code execution result )"'
      in response_sql
  )
  assert 'r"\' [|] other(?: [|] |$)"' in response_sql
  assert "r' [|] " not in response_sql
