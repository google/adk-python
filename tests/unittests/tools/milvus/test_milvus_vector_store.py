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

from google.adk.tools.milvus.milvus_vector_store import MilvusVectorStore
from google.adk.tools.milvus.settings import MilvusToolSettings
from google.adk.tools.milvus.settings import MilvusVectorStoreSettings
import pytest

DIMENSION = 4


def _mock_embedding_fn(texts: list[str]) -> list[list[float]]:
  """A mock embedding function returning fixed-dimension vectors."""
  return [[0.1] * DIMENSION for _ in texts]


@pytest.fixture
def vector_store_settings():
  return MilvusVectorStoreSettings(
      collection_name="test_collection",
      dimension=DIMENSION,
      metric_type="COSINE",
  )


@pytest.fixture
def tool_settings(vector_store_settings):
  return MilvusToolSettings(vector_store_settings=vector_store_settings)


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_init_success(mock_milvus_client_cls, tool_settings):
  """Test successful initialization of MilvusVectorStore."""
  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )
  mock_milvus_client_cls.assert_called_once_with(
      uri="http://localhost:19530",
      token=None,
      db_name="default",
  )
  assert store._embedding_fn is _mock_embedding_fn


def test_init_missing_vector_store_settings():
  """Test that missing vector_store_settings raises ValueError."""
  settings = MilvusToolSettings()
  with pytest.raises(ValueError, match="not set"):
    MilvusVectorStore(
        settings=settings,
        embedding_fn=_mock_embedding_fn,
    )


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_setup_creates_collection(mock_milvus_client_cls, tool_settings):
  """Test that setup creates collection when it doesn't exist."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = False

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )

  store.setup()

  mock_client.has_collection.assert_called_once_with("test_collection")
  mock_client.create_collection.assert_called_once()


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_setup_skips_existing_collection(mock_milvus_client_cls, tool_settings):
  """Test that setup skips when collection already exists."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = True

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )
  store.setup()

  mock_client.has_collection.assert_called_once_with("test_collection")
  mock_client.create_collection.assert_not_called()


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_add_contents(mock_milvus_client_cls, tool_settings):
  """Test adding contents to the vector store."""
  mock_client = mock_milvus_client_cls.return_value

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )

  contents = ["hello world", "foo bar"]
  store.add_contents(contents)

  mock_client.insert.assert_called_once()
  call_args = mock_client.insert.call_args
  assert call_args.kwargs["collection_name"] == "test_collection"
  data = call_args.kwargs["data"]
  assert len(data) == 2
  assert data[0]["content"] == "hello world"
  assert data[1]["content"] == "foo bar"
  assert len(data[0]["embedding"]) == DIMENSION


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_add_contents_with_batching(mock_milvus_client_cls, tool_settings):
  """Test that add_contents batches correctly."""
  mock_client = mock_milvus_client_cls.return_value

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )

  contents = [f"doc_{i}" for i in range(5)]
  store.add_contents(contents, batch_size=2)

  # 5 items with batch_size=2 => 3 batches (2, 2, 1)
  assert mock_client.insert.call_count == 3


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_add_contents_with_additional_fields(
    mock_milvus_client_cls, tool_settings
):
  """Test adding contents with additional fields."""
  mock_client = mock_milvus_client_cls.return_value

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )

  contents = ["hello"]
  additional = [{"title": "greeting"}]
  store.add_contents(contents, additional_fields=additional)

  call_args = mock_client.insert.call_args
  data = call_args.kwargs["data"]
  assert data[0]["title"] == "greeting"


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_add_contents_mismatched_additional_fields(
    mock_milvus_client_cls, tool_settings
):
  """Test that mismatched additional_fields raises ValueError."""
  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )

  with pytest.raises(ValueError, match="must match"):
    store.add_contents(
        ["doc1", "doc2"],
        additional_fields=[{"title": "only_one"}],
    )


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_search(mock_milvus_client_cls, tool_settings):
  """Test similarity search."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.search.return_value = [[
      {"entity": {"content": "result1"}, "distance": 0.9},
      {"entity": {"content": "result2"}, "distance": 0.8},
  ]]

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )

  results = store.search("test query")

  mock_client.search.assert_called_once()
  call_args = mock_client.search.call_args
  assert call_args.kwargs["collection_name"] == "test_collection"
  assert call_args.kwargs["limit"] == 5
  assert call_args.kwargs["output_fields"] == ["content"]

  assert len(results) == 2
  assert results[0]["content"] == "result1"
  assert results[0]["distance"] == 0.9


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_search_with_custom_top_k(mock_milvus_client_cls, tool_settings):
  """Test search with custom top_k."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.search.return_value = [[]]

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )

  store.search("test", top_k=10)

  call_args = mock_client.search.call_args
  assert call_args.kwargs["limit"] == 10


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_search_with_filter(mock_milvus_client_cls, tool_settings):
  """Test search with filter expression."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.search.return_value = [[]]

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )

  store.search("test", filter_expr='category == "tech"')

  call_args = mock_client.search.call_args
  assert call_args.kwargs["filter"] == 'category == "tech"'


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_search_empty_results(mock_milvus_client_cls, tool_settings):
  """Test search returning empty results."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.search.return_value = [[]]

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )

  results = store.search("test")
  assert results == []


@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
def test_close(mock_milvus_client_cls, tool_settings):
  """Test closing the vector store."""
  mock_client = mock_milvus_client_cls.return_value

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )
  store.close()

  mock_client.close.assert_called_once()


@pytest.mark.asyncio
@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
async def test_add_contents_async(mock_milvus_client_cls, tool_settings):
  """Test async add_contents delegates to sync version."""
  mock_client = mock_milvus_client_cls.return_value

  store = MilvusVectorStore(
      settings=tool_settings,
      embedding_fn=_mock_embedding_fn,
  )

  await store.add_contents_async(["hello", "world"])

  mock_client.insert.assert_called_once()
