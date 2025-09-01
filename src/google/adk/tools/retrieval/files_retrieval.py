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

"""Provides data for the agent."""

from __future__ import annotations

import logging
from typing import Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding

from .llama_index_retrieval import LlamaIndexRetrieval

logger = logging.getLogger("google_adk." + __name__)


def _get_default_embedding_model() -> BaseEmbedding:
  """Get the default Google Gemini embedding model.

  Returns:
    GoogleGenAIEmbedding instance configured with text-embedding-004 model.

  Raises:
    ImportError: If llama-index-embeddings-google-genai package is not installed.
  """
  try:
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    return GoogleGenAIEmbedding(model_name="text-embedding-004")
  except ImportError as e:
    raise ImportError(
        "llama-index-embeddings-google-genai package not found. "
        "Please run: pip install llama-index-embeddings-google-genai"
    ) from e


class FilesRetrieval(LlamaIndexRetrieval):

  def __init__(
      self,
      *,
      name: str,
      description: str,
      input_dir: str,
      embedding_model: Optional[BaseEmbedding] = None,
  ):
    """Initialize FilesRetrieval with optional embedding model.

    Args:
      name: Name of the tool.
      description: Description of the tool.
      input_dir: Directory path containing files to index.
      embedding_model: Optional custom embedding model. If None, defaults to
        Google's text-embedding-004 model.
    """
    self.input_dir = input_dir

    if embedding_model is None:
      embedding_model = _get_default_embedding_model()

    logger.info("Loading data from %s", input_dir)
    retriever = VectorStoreIndex.from_documents(
        SimpleDirectoryReader(input_dir).load_data(),
        embed_model=embedding_model,
    ).as_retriever()
    super().__init__(name=name, description=description, retriever=retriever)
