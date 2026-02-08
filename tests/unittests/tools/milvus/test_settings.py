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

from google.adk.tools.milvus.settings import MilvusToolSettings
from google.adk.tools.milvus.settings import MilvusVectorStoreSettings
import pytest


def test_default_settings():
  """Test MilvusVectorStoreSettings with default values."""
  settings = MilvusVectorStoreSettings(collection_name="test_collection")
  assert settings.uri == "http://localhost:19530"
  assert settings.token is None
  assert settings.db_name == "default"
  assert settings.collection_name == "test_collection"
  assert settings.dimension == 768
  assert settings.metric_type == "COSINE"
  assert settings.index_type == "AUTOINDEX"
  assert settings.content_field == "content"
  assert settings.embedding_field == "embedding"
  assert settings.primary_field == "id"
  assert settings.top_k == 5
  assert settings.output_fields is None


def test_custom_settings():
  """Test MilvusVectorStoreSettings with custom values."""
  settings = MilvusVectorStoreSettings(
      uri="http://milvus:19530",
      token="test_token",
      db_name="mydb",
      collection_name="docs",
      dimension=384,
      metric_type="L2",
      index_type="HNSW",
      content_field="text",
      embedding_field="vec",
      primary_field="pk",
      top_k=10,
      output_fields=["text", "title"],
  )
  assert settings.uri == "http://milvus:19530"
  assert settings.token == "test_token"
  assert settings.db_name == "mydb"
  assert settings.dimension == 384
  assert settings.metric_type == "L2"
  assert settings.index_type == "HNSW"
  assert settings.content_field == "text"
  assert settings.embedding_field == "vec"
  assert settings.top_k == 10
  assert settings.output_fields == ["text", "title"]


def test_invalid_dimension():
  """Test that invalid dimension raises ValueError."""
  with pytest.raises(ValueError, match="Invalid dimension"):
    MilvusVectorStoreSettings(
        collection_name="test",
        dimension=0,
    )


def test_invalid_dimension_negative():
  """Test that negative dimension raises ValueError."""
  with pytest.raises(ValueError, match="Invalid dimension"):
    MilvusVectorStoreSettings(
        collection_name="test",
        dimension=-1,
    )


def test_invalid_top_k():
  """Test that invalid top_k raises ValueError."""
  with pytest.raises(ValueError, match="Invalid top_k"):
    MilvusVectorStoreSettings(
        collection_name="test",
        top_k=0,
    )


def test_milvus_tool_settings_default():
  """Test MilvusToolSettings with default values."""
  settings = MilvusToolSettings()
  assert settings.vector_store_settings is None


def test_milvus_tool_settings_with_vector_store():
  """Test MilvusToolSettings with vector store settings."""
  vs_settings = MilvusVectorStoreSettings(collection_name="test")
  settings = MilvusToolSettings(vector_store_settings=vs_settings)
  assert settings.vector_store_settings is not None
  assert settings.vector_store_settings.collection_name == "test"
