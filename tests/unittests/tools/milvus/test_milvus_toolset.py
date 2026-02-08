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

from google.adk.tools.milvus.milvus_tool import MilvusTool
from google.adk.tools.milvus.milvus_toolset import MilvusToolset
from google.adk.tools.milvus.settings import MilvusToolSettings
from google.adk.tools.milvus.settings import MilvusVectorStoreSettings
import pytest

DIMENSION = 4


def _mock_embedding_fn(texts: list[str]) -> list[list[float]]:
  return [[0.1] * DIMENSION for _ in texts]


@pytest.mark.asyncio
@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
async def test_toolset_returns_search_tool(mock_milvus_client_cls):
  """Test that MilvusToolset returns a similarity_search tool."""
  settings = MilvusToolSettings(
      vector_store_settings=MilvusVectorStoreSettings(
          collection_name="test",
          dimension=DIMENSION,
      ),
  )
  toolset = MilvusToolset(
      milvus_tool_settings=settings,
      embedding_fn=_mock_embedding_fn,
  )

  tools = await toolset.get_tools()
  assert len(tools) == 1
  assert isinstance(tools[0], MilvusTool)
  assert tools[0].name == "similarity_search"


@pytest.mark.asyncio
async def test_toolset_no_vector_store_settings():
  """Test that MilvusToolset returns no tools without vector store settings."""
  toolset = MilvusToolset(
      milvus_tool_settings=MilvusToolSettings(),
      embedding_fn=_mock_embedding_fn,
  )

  tools = await toolset.get_tools()
  assert len(tools) == 0


@pytest.mark.asyncio
@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
async def test_toolset_with_filter(mock_milvus_client_cls):
  """Test that MilvusToolset respects tool_filter."""
  settings = MilvusToolSettings(
      vector_store_settings=MilvusVectorStoreSettings(
          collection_name="test",
          dimension=DIMENSION,
      ),
  )
  toolset = MilvusToolset(
      milvus_tool_settings=settings,
      embedding_fn=_mock_embedding_fn,
      tool_filter=["nonexistent_tool"],
  )

  tools = await toolset.get_tools()
  assert len(tools) == 0


@pytest.mark.asyncio
@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
async def test_toolset_with_matching_filter(mock_milvus_client_cls):
  """Test that tool_filter includes matching tools."""
  settings = MilvusToolSettings(
      vector_store_settings=MilvusVectorStoreSettings(
          collection_name="test",
          dimension=DIMENSION,
      ),
  )
  toolset = MilvusToolset(
      milvus_tool_settings=settings,
      embedding_fn=_mock_embedding_fn,
      tool_filter=["similarity_search"],
  )

  tools = await toolset.get_tools()
  assert len(tools) == 1


@pytest.mark.asyncio
@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
async def test_toolset_close(mock_milvus_client_cls):
  """Test that close shuts down the vector store."""
  mock_client = mock_milvus_client_cls.return_value
  settings = MilvusToolSettings(
      vector_store_settings=MilvusVectorStoreSettings(
          collection_name="test",
          dimension=DIMENSION,
      ),
  )
  toolset = MilvusToolset(
      milvus_tool_settings=settings,
      embedding_fn=_mock_embedding_fn,
  )

  # Force vector store creation
  await toolset.get_tools()

  await toolset.close()
  mock_client.close.assert_called_once()


@pytest.mark.asyncio
@mock.patch("google.adk.tools.milvus.milvus_vector_store.MilvusClient")
async def test_toolset_close_without_init(mock_milvus_client_cls):
  """Test that close works even if vector store was never created."""
  toolset = MilvusToolset(
      milvus_tool_settings=MilvusToolSettings(),
      embedding_fn=_mock_embedding_fn,
  )
  # Should not raise
  await toolset.close()
