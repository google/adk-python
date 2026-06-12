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

"""Tests for SQL identifier validation in Spanner search tool.

Verifies that malicious SQL identifiers and filter patterns are rejected
before being interpolated into SQL queries (defense against SQL injection
via LLM-populated tool parameters).
"""

from google.adk.tools.spanner.search_tool import _generate_sql_for_ann
from google.adk.tools.spanner.search_tool import _generate_sql_for_knn
from google.adk.tools.spanner.search_tool import _validate_additional_filter
from google.adk.tools.spanner.search_tool import _validate_column_list
from google.adk.tools.spanner.search_tool import _validate_identifier
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect
import pytest


class TestValidateIdentifier:
  """Tests for _validate_identifier."""

  def test_simple_identifier(self):
    assert _validate_identifier("documents", "test") == "documents"

  def test_schema_qualified_identifier(self):
    assert (
        _validate_identifier("my_schema.my_table", "test")
        == "my_schema.my_table"
    )

  def test_identifier_with_underscores(self):
    assert _validate_identifier("embedding_col_1", "test") == "embedding_col_1"

  def test_backtick_quoted_identifier(self):
    assert _validate_identifier("`my table`", "test") == "`my table`"

  def test_double_quote_quoted_identifier(self):
    assert _validate_identifier('"my column"', "test") == '"my column"'

  def test_rejects_join_injection(self):
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
      _validate_identifier(
          "documents JOIN admin_credentials ac ON TRUE", "table_name"
      )

  def test_rejects_subquery_in_column(self):
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
      _validate_identifier(
          "(SELECT STRING_AGG(table_name, ',') FROM INFORMATION_SCHEMA.TABLES)"
          " AS schema_dump",
          "columns",
      )

  def test_rejects_semicolon(self):
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
      _validate_identifier("table; DROP TABLE users", "table_name")

  def test_rejects_empty(self):
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
      _validate_identifier("", "table_name")

  def test_rejects_sql_comment(self):
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
      _validate_identifier("table -- comment", "table_name")


class TestValidateColumnList:
  """Tests for _validate_column_list."""

  def test_valid_columns(self):
    result = _validate_column_list(["col1", "col2", "col3"], "columns")
    assert result == ["col1", "col2", "col3"]

  def test_rejects_subquery_column(self):
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
      _validate_column_list(
          [
              (
                  "(SELECT STRING_AGG(table_name, ',') FROM"
                  " INFORMATION_SCHEMA.TABLES) AS dump"
              ),
              "content",
          ],
          "columns",
      )


class TestValidateAdditionalFilter:
  """Tests for _validate_additional_filter."""

  def test_none_filter(self):
    assert _validate_additional_filter(None) is None

  def test_simple_filter(self):
    assert (
        _validate_additional_filter("price_in_cents < 100000")
        == "price_in_cents < 100000"
    )

  def test_rejects_union(self):
    with pytest.raises(ValueError, match="UNION"):
      _validate_additional_filter(
          "1=1 UNION ALL SELECT password, 0.0 FROM admin_credentials"
      )

  def test_rejects_semicolon(self):
    with pytest.raises(ValueError, match="disallowed pattern"):
      _validate_additional_filter("1=1; SELECT * FROM secrets")

  def test_rejects_line_comment(self):
    with pytest.raises(ValueError, match="disallowed pattern"):
      _validate_additional_filter("1=1 -- bypass")

  def test_rejects_block_comment(self):
    with pytest.raises(ValueError, match="disallowed pattern"):
      _validate_additional_filter("1=1 /* bypass */")


class TestGenerateSqlForKnn:
  """Tests for _generate_sql_for_knn with validation."""

  def test_valid_query_googlesql(self):
    sql = _generate_sql_for_knn(
        dialect=DatabaseDialect.GOOGLE_STANDARD_SQL,
        table_name="documents",
        embedding_column_to_search="embedding",
        columns=["content"],
        additional_filter=None,
        distance_type="COSINE",
        top_k=10,
    )
    assert "FROM documents" in sql
    assert "COSINE_DISTANCE" in sql

  def test_rejects_union_in_filter(self):
    with pytest.raises(ValueError, match="UNION"):
      _generate_sql_for_knn(
          dialect=DatabaseDialect.GOOGLE_STANDARD_SQL,
          table_name="documents",
          embedding_column_to_search="embedding",
          columns=["content"],
          additional_filter=(
              "1=1 UNION ALL SELECT password, 0.0 FROM admin_credentials"
          ),
          distance_type="COSINE",
          top_k=10,
      )

  def test_rejects_join_in_table_name(self):
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
      _generate_sql_for_knn(
          dialect=DatabaseDialect.GOOGLE_STANDARD_SQL,
          table_name="documents JOIN admin_credentials ac ON TRUE",
          embedding_column_to_search="embedding",
          columns=["content"],
          additional_filter=None,
          distance_type="COSINE",
          top_k=10,
      )

  def test_rejects_subquery_in_columns(self):
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
      _generate_sql_for_knn(
          dialect=DatabaseDialect.GOOGLE_STANDARD_SQL,
          table_name="documents",
          embedding_column_to_search="embedding",
          columns=[
              (
                  "(SELECT STRING_AGG(table_name, ',') FROM"
                  " INFORMATION_SCHEMA.TABLES) AS schema_dump"
              ),
          ],
          additional_filter=None,
          distance_type="COSINE",
          top_k=1,
      )

  def test_top_k_string_coerced_to_int(self):
    sql = _generate_sql_for_knn(
        dialect=DatabaseDialect.GOOGLE_STANDARD_SQL,
        table_name="documents",
        embedding_column_to_search="embedding",
        columns=["content"],
        additional_filter=None,
        distance_type="COSINE",
        top_k="10",  # String input
    )
    assert "LIMIT 10" in sql
