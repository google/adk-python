# Copyright 2025 Google LLC
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

"""Tests for OllamaEmbeddingProvider."""

from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.memory.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider
import pytest
import requests


@pytest.fixture
def provider():
  return OllamaEmbeddingProvider(model="test-model", host="http://test-host")


@patch("requests.post")
def test_embed_success(mock_post, provider):
  """Test successful embedding generation."""
  mock_response = MagicMock()
  mock_response.json.return_value = {
      "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
  }
  mock_post.return_value = mock_response

  # Run the async method synchronously for testing logic
  # Since we mocked the synchronous _post_embed call via requests.post,
  # we can verify the result.
  # However, OllamaEmbeddingProvider.embed is async and uses asyncio.to_thread.
  # We need to run it in an async loop or trust pytest-asyncio.
  import asyncio

  embeddings = asyncio.run(provider.embed(["text1", "text2"]))

  assert len(embeddings) == 2
  assert embeddings[0] == [0.1, 0.2, 0.3]
  assert embeddings[1] == [0.4, 0.5, 0.6]
  assert provider.dimension == 3

  mock_post.assert_called_once()
  args, kwargs = mock_post.call_args
  assert args[0] == "http://test-host/api/embed"
  assert kwargs["json"] == {
      "model": "test-model",
      "input": ["text1", "text2"],
  }


@patch("requests.post")
def test_embed_http_error(mock_post, provider):
  """Test handling of HTTP errors."""
  mock_response = MagicMock()
  mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
      "404 Client Error"
  )
  mock_post.return_value = mock_response

  import asyncio

  with pytest.raises(RuntimeError, match="Failed to connect to Ollama"):
    asyncio.run(provider.embed(["text"]))


@patch("requests.post")
def test_embed_connection_error(mock_post, provider):
  """Test handling of connection errors."""
  mock_post.side_effect = requests.exceptions.ConnectionError(
      "Connection refused"
  )

  import asyncio

  with pytest.raises(RuntimeError, match="Failed to connect to Ollama"):
    asyncio.run(provider.embed(["text"]))


def test_dimension_property(provider):
  """Test dimension property raises error if not set."""
  with pytest.raises(ValueError, match="Dimension is not available"):
    _ = provider.dimension
