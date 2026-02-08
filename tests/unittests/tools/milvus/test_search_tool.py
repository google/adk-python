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

import json
from unittest import mock

from google.adk.tools.milvus.milvus_vector_store import MilvusVectorStore
from google.adk.tools.milvus.search_tool import similarity_search


def test_similarity_search_with_results():
  """Test similarity_search returns JSON formatted results."""
  mock_store = mock.create_autospec(MilvusVectorStore, instance=True)
  mock_store.search.return_value = [
      {"content": "result1", "distance": 0.9},
      {"content": "result2", "distance": 0.8},
  ]

  result = similarity_search(
      query="test query",
      vector_store=mock_store,
  )

  mock_store.search.assert_called_once_with(
      query="test query", filter_expr=None
  )
  parsed = json.loads(result)
  assert len(parsed) == 2
  assert parsed[0]["content"] == "result1"


def test_similarity_search_no_results():
  """Test similarity_search with no matching results."""
  mock_store = mock.create_autospec(MilvusVectorStore, instance=True)
  mock_store.search.return_value = []

  result = similarity_search(
      query="unknown query",
      vector_store=mock_store,
  )

  assert result == "No matching results found."


def test_similarity_search_with_filter():
  """Test similarity_search with filter expression."""
  mock_store = mock.create_autospec(MilvusVectorStore, instance=True)
  mock_store.search.return_value = [
      {"content": "filtered result", "distance": 0.95},
  ]

  result = similarity_search(
      query="test",
      vector_store=mock_store,
      filter_expr='category == "tech"',
  )

  mock_store.search.assert_called_once_with(
      query="test", filter_expr='category == "tech"'
  )
  parsed = json.loads(result)
  assert len(parsed) == 1
  assert parsed[0]["content"] == "filtered result"
