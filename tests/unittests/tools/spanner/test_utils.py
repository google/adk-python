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

from __future__ import annotations

from unittest import mock

from google.adk.tools.spanner import utils as spanner_utils
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.tools.spanner.settings import SpannerVectorStoreSettings
from google.adk.tools.spanner.settings import TableColumn
from google.adk.tools.spanner.settings import VectorSearchIndexSettings
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect
from google.cloud.spanner_v1 import batch as spanner_batch
from google.cloud.spanner_v1 import client as spanner_client_v1
from google.cloud.spanner_v1 import database as spanner_database
from google.cloud.spanner_v1 import instance as spanner_instance
import pytest


@pytest.fixture
def vector_store_settings():
  """Fixture for SpannerVectorStoreSettings."""
  return SpannerVectorStoreSettings(
      project_id="test-project",
      instance_id="test-instance",
      database_id="test-database",
      table_name="test_vector_store",
      content_column="content",
      embedding_column="embedding",
      vector_length=768,
      vertex_ai_embedding_model_name="textembedding",
  )


@pytest.fixture
def spanner_tool_settings(vector_store_settings):
  """Fixture for SpannerToolSettings."""
  return SpannerToolSettings(vector_store_settings=vector_store_settings)


@pytest.fixture
def mock_spanner_database():
  """Fixture for a mocked spanner database."""
  mock_database = mock.create_autospec(spanner_database.Database, instance=True)
  mock_database.exists.return_value = True
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  return mock_database


@pytest.fixture
def mock_spanner_instance(mock_spanner_database):
  """Fixture for a mocked spanner instance."""
  mock_instance = mock.create_autospec(spanner_instance.Instance, instance=True)
  mock_instance.exists.return_value = True
  mock_instance.database.return_value = mock_spanner_database
  return mock_instance


@pytest.fixture
def mock_spanner_client(mock_spanner_instance):
  """Fixture for a mocked spanner client."""
  mock_client = mock.create_autospec(spanner_client_v1.Client, instance=True)
  mock_client.instance.return_value = mock_spanner_instance
  mock_client._client_info = mock.Mock(user_agent="test-agent")
  return mock_client


@mock.patch.object(spanner_utils, "embed_contents", autospec=True)
def test_add_contents_successful(
    mock_embed_contents,
    spanner_tool_settings,
    mock_spanner_client,
    mock_spanner_database,
    mocker,
):
  """Test that add_contents successfully adds content."""
  mock_embed_contents.return_value = [[1.0, 2.0], [3.0, 4.0]]
  mock_batch = mocker.create_autospec(spanner_batch.Batch, instance=True)
  mock_batch.__enter__.return_value = mock_batch
  mock_spanner_database.batch.return_value = mock_batch

  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    vector_store._database = mock_spanner_database
    contents = ["content1", "content2"]
    vector_store.add_contents(contents=contents)

  mock_spanner_database.reload.assert_called_once()
  mock_spanner_database.batch.assert_called_once()
  mock_batch.insert_or_update.assert_called_once_with(
      table="test_vector_store",
      columns=["content", "embedding"],
      values=[
          ["content1", [1.0, 2.0]],
          ["content2", [3.0, 4.0]],
      ],
  )
  mock_embed_contents.assert_called_once_with(
      "textembedding", contents, 768, mock.ANY
  )


@mock.patch.object(spanner_utils, "embed_contents", autospec=True)
def test_add_contents_with_metadata(
    mock_embed_contents,
    spanner_tool_settings,
    mock_spanner_client,
    mock_spanner_database,
    mocker,
):
  """Test that add_contents successfully adds content with metadata."""
  mock_embed_contents.return_value = [[1.0, 2.0], [3.0, 4.0]]
  mock_batch = mocker.create_autospec(spanner_batch.Batch, instance=True)
  mock_batch.__enter__.return_value = mock_batch
  mock_spanner_database.batch.return_value = mock_batch
  spanner_tool_settings.vector_store_settings.additional_columns_to_setup = [
      TableColumn(name="metadata", type="JSON")
  ]

  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    vector_store._database = mock_spanner_database
    contents = ["content1", "content2"]
    additional_columns_values = [
        {"metadata": {"meta1": "val1"}},
        {"metadata": {"meta2": "val2"}},
    ]
    vector_store.add_contents(
        contents=contents,
        additional_columns_values=additional_columns_values,
    )

  mock_spanner_database.batch.assert_called_once()
  mock_batch.insert_or_update.assert_called_once_with(
      table="test_vector_store",
      columns=["content", "embedding", "metadata"],
      values=[
          ["content1", [1.0, 2.0], {"meta1": "val1"}],
          ["content2", [3.0, 4.0], {"meta2": "val2"}],
      ],
  )


def test_add_contents_empty_contents(
    spanner_tool_settings, mock_spanner_client, mock_spanner_database
):
  """Test that add_contents does nothing when contents is empty."""
  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    vector_store.add_contents(contents=[])
    mock_spanner_database.batch.assert_not_called()


@mock.patch.object(spanner_utils.client, "get_spanner_client", autospec=True)
def test_execute_sql_circular_row_fallback_to_string(mock_get_spanner_client):
  """Test execute_sql stringifies rows with circular references."""
  mock_spanner_client = mock.MagicMock()
  mock_instance = mock.MagicMock()
  mock_database = mock.MagicMock()
  mock_snapshot = mock.MagicMock()
  circular_row = []
  circular_row.append(circular_row)
  mock_snapshot.execute_sql.return_value = iter([circular_row])
  mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = spanner_utils.execute_sql(
      project_id="test-project",
      instance_id="test-instance",
      database_id="test-database",
      query="SELECT 1",
      credentials=mock.Mock(),
      settings=SpannerToolSettings(),
      tool_context=mock.Mock(),
  )

  assert result == {"status": "SUCCESS", "rows": [str(circular_row)]}


@mock.patch.object(spanner_utils, "embed_contents", autospec=True)
def test_add_contents_additional_columns_list_mismatch(
    mock_embed_contents, spanner_tool_settings, mock_spanner_client
):
  """Test that add_contents raises an error if additional_columns_values and contents lengths differ."""
  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    with pytest.raises(
        ValueError,
        match="additional_columns_values contains more items than contents.",
    ):
      vector_store.add_contents(
          contents=["content1"],
          additional_columns_values=[
              {"col1": "val1"},
              {"col1": "val2"},
          ],
      )


@mock.patch.object(spanner_utils, "embed_contents", autospec=True)
def test_add_contents_embedding_fails(
    mock_embed_contents, spanner_tool_settings, mock_spanner_client
):
  """Test that add_contents fails if embedding fails."""
  mock_embed_contents.side_effect = RuntimeError("Embedding failed")
  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    with pytest.raises(RuntimeError, match="Embedding failed"):
      vector_store.add_contents(contents=["content1", "content2"])


def test_init_raises_error_if_vector_store_settings_not_set():
  """Test that SpannerVectorStore raises an error if vector_store_settings is not set."""
  settings = SpannerToolSettings()
  with pytest.raises(
      ValueError, match="Spanner vector store settings are not set."
  ):
    spanner_utils.SpannerVectorStore(settings)


@pytest.mark.parametrize(
    "dialect, expected_ddl",
    [
        (
            DatabaseDialect.GOOGLE_STANDARD_SQL,
            (
                "CREATE TABLE IF NOT EXISTS test_vector_store (\n"
                "  id STRING(36) DEFAULT (GENERATE_UUID()),\n"
                "  content STRING(MAX),\n"
                "  embedding ARRAY<FLOAT32>(vector_length=>768)\n"
                ") PRIMARY KEY(id)"
            ),
        ),
        (
            DatabaseDialect.POSTGRESQL,
            (
                "CREATE TABLE IF NOT EXISTS test_vector_store (\n"
                "  id varchar(36) DEFAULT spanner.generate_uuid(),\n"
                "  content text,\n"
                "  embedding float4[] VECTOR LENGTH 768,\n"
                "  PRIMARY KEY(id)\n"
                ")"
            ),
        ),
    ],
)
def test_create_vector_store_table_ddl(
    spanner_tool_settings, mock_spanner_client, dialect, expected_ddl
):
  """Test DDL creation for different SQL dialects."""
  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    ddl = vector_store._create_vector_store_table_ddl(dialect)
    assert ddl == expected_ddl


def test_create_ann_vector_search_index_ddl_raises_error_for_postgresql(
    spanner_tool_settings, vector_store_settings, mock_spanner_client
):
  """Test that creating an ANN index raises an error for PostgreSQL."""
  vector_store_settings.vector_search_index_settings = mock.Mock()
  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    with pytest.raises(
        ValueError,
        match="ANN is only supported for the Google Standard SQL dialect.",
    ):
      vector_store._create_ann_vector_search_index_ddl(
          DatabaseDialect.POSTGRESQL
      )


def test_create_vector_store(
    spanner_tool_settings, mock_spanner_client, mock_spanner_database
):
  """Test the vector store creation process."""
  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    vector_store.create_vector_store()
    mock_spanner_database.update_ddl.assert_called_once()
    ddl_statement = mock_spanner_database.update_ddl.call_args[0][0]
    assert "CREATE TABLE IF NOT EXISTS test_vector_store" in ddl_statement[0]


def test_create_vector_search_index_no_settings(
    spanner_tool_settings, mock_spanner_client, mock_spanner_database
):
  """Test that create_vector_search_index does nothing if settings are not present."""
  spanner_tool_settings.vector_store_settings.vector_search_index_settings = (
      None
  )
  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    vector_store.create_vector_search_index()
    mock_spanner_database.update_ddl.assert_not_called()


def test_create_vector_search_index_successful_google_sql(
    spanner_tool_settings,
    vector_store_settings,
    mock_spanner_client,
    mock_spanner_database,
):
  """Test that create_vector_search_index successfully creates index for Google SQL."""
  mock_spanner_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  vector_store_settings.vector_search_index_settings = (
      VectorSearchIndexSettings(
          index_name="test_vector_index",
          tree_depth=3,
          num_branches=10,
          num_leaves=20,
      )
  )
  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    vector_store.create_vector_search_index()
    mock_spanner_database.update_ddl.assert_called_once()
    ddl_statement = mock_spanner_database.update_ddl.call_args[0][0]
    expected_ddl = (
        "CREATE VECTOR INDEX IF NOT EXISTS test_vector_index\n"
        "\tON test_vector_store(embedding)\n"
        "\tWHERE embedding IS NOT NULL\n"
        "\tOPTIONS(distance_type='COSINE', tree_depth=3, num_branches=10, "
        "num_leaves=20)"
    )
    assert ddl_statement[0] == expected_ddl


def test_create_vector_search_index_fails(
    spanner_tool_settings,
    vector_store_settings,
    mock_spanner_client,
    mock_spanner_database,
):
  """Test that create_vector_search_index raises an error if DDL execution fails."""
  mock_spanner_database.update_ddl.side_effect = RuntimeError("DDL failed")
  vector_store_settings.vector_search_index_settings = (
      VectorSearchIndexSettings(index_name="test_vector_index")
  )
  with mock.patch.object(
      spanner_utils.client,
      "get_spanner_client",
      autospec=True,
      return_value=mock_spanner_client,
  ):
    vector_store = spanner_utils.SpannerVectorStore(spanner_tool_settings)
    with pytest.raises(RuntimeError, match="DDL failed"):
      vector_store.create_vector_search_index()
