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

"""Tests for FilesRetrieval tool."""

import unittest.mock as mock

from google.adk.tools.retrieval.files_retrieval import _get_default_embedding_model
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval
from llama_index.core.base.embeddings.base import BaseEmbedding
import pytest


class MockEmbedding(BaseEmbedding):
  """Mock embedding model for testing."""

  def _get_query_embedding(self, query):
    return [0.1] * 384

  def _get_text_embedding(self, text):
    return [0.1] * 384

  async def _aget_query_embedding(self, query):
    return [0.1] * 384

  async def _aget_text_embedding(self, text):
    return [0.1] * 384


class TestFilesRetrieval:

  def test_files_retrieval_with_custom_embedding(self, tmp_path):
    """Test FilesRetrieval with custom embedding model."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document for retrieval testing.")

    custom_embedding = MockEmbedding()
    retrieval = FilesRetrieval(
        name="test_retrieval",
        description="Test retrieval tool",
        input_dir=str(tmp_path),
        embedding_model=custom_embedding,
    )

    assert retrieval.name == "test_retrieval"
    assert retrieval.input_dir == str(tmp_path)
    assert retrieval.retriever is not None

  @mock.patch(
      "google.adk.tools.retrieval.files_retrieval._get_default_embedding_model"
  )
  def test_files_retrieval_uses_default_embedding(
      self, mock_get_default_embedding, tmp_path
  ):
    """Test FilesRetrieval uses default embedding when none provided."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document for retrieval testing.")

    mock_embedding = MockEmbedding()
    mock_get_default_embedding.return_value = mock_embedding

    retrieval = FilesRetrieval(
        name="test_retrieval",
        description="Test retrieval tool",
        input_dir=str(tmp_path),
    )

    mock_get_default_embedding.assert_called_once()
    assert retrieval.name == "test_retrieval"
    assert retrieval.input_dir == str(tmp_path)

  def test_get_default_embedding_model_import_error(self):
    """Test _get_default_embedding_model handles ImportError correctly."""
    # Simulate the package not being installed by making import fail
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
      if name == "llama_index.embeddings.google_genai":
        raise ImportError(
            "No module named 'llama_index.embeddings.google_genai'"
        )
      return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mock_import):
      with pytest.raises(ImportError) as exc_info:
        _get_default_embedding_model()

      # The exception should be re-raised as our custom ImportError with helpful message
      assert "llama-index-embeddings-google-genai package not found" in str(
          exc_info.value
      )
      assert "pip install llama-index-embeddings-google-genai" in str(
          exc_info.value
      )

  def test_get_default_embedding_model_success(self):
    """Test _get_default_embedding_model returns Google embedding when available."""

    # Mock the module creation to avoid import issues
    mock_module = mock.MagicMock()
    mock_embedding_instance = MockEmbedding()
    mock_module.GoogleGenAIEmbedding.return_value = mock_embedding_instance

    with mock.patch.dict(
        "sys.modules", {"llama_index.embeddings.google_genai": mock_module}
    ):
      result = _get_default_embedding_model()

      mock_module.GoogleGenAIEmbedding.assert_called_once_with(
          model_name="text-embedding-004"
      )
      assert result == mock_embedding_instance

  def test_backward_compatibility(self, tmp_path):
    """Test that existing code without embedding_model parameter still works."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document for retrieval testing.")

    with mock.patch(
        "google.adk.tools.retrieval.files_retrieval._get_default_embedding_model"
    ) as mock_get_default:
      mock_get_default.return_value = MockEmbedding()

      # This should work exactly like before - no embedding_model parameter
      retrieval = FilesRetrieval(
          name="test_retrieval",
          description="Test retrieval tool",
          input_dir=str(tmp_path),
      )

      assert retrieval.name == "test_retrieval"
      assert retrieval.input_dir == str(tmp_path)
      mock_get_default.assert_called_once()
