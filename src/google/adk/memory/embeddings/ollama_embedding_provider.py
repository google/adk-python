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

"""Ollama embedding provider for ChromaMemoryService."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
import requests

from .base_embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger("google_adk." + __name__)

_EMBED_ENDPOINT = "/api/embed"


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
  """Embedding provider using Ollama's embedding API.

  This provider uses Ollama's `/api/embed` endpoint to generate embeddings.
  It requires an Ollama server running with an embedding model available.

  Example:
      >>> provider = OllamaEmbeddingProvider(model="nomic-embed-text")
      >>> embeddings = await provider.embed(["Hello, world!"])
  """

  def __init__(
      self,
      model: str = "nomic-embed-text",
      host: Optional[str] = None,
      request_timeout: float = 60.0,
  ):
    """Initialize the Ollama embedding provider.

    Args:
        model: The name of the Ollama embedding model to use.
            Popular options: "nomic-embed-text", "mxbai-embed-large",
            "all-minilm".
        host: The base URL of the Ollama server. Defaults to
            http://localhost:11434 or OLLAMA_API_BASE env var.
        request_timeout: Timeout in seconds for embedding requests.
    """
    import os

    self._model = model
    self._host = host or os.environ.get(
        "OLLAMA_API_BASE", "http://localhost:11434"
    )
    self._request_timeout = request_timeout
    self._dimension: Optional[int] = None

  @property
  def dimension(self) -> int:
    """Return the dimension of the embedding vectors.

    The dimension is determined by the first embedding request.
    """
    if self._dimension is None:
      raise ValueError(
          "Dimension is not available until the first embedding is generated."
      )
    return self._dimension

  async def embed(self, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using Ollama.

    Args:
        texts: A list of strings to embed.

    Returns:
        A list of embeddings, where each embedding is a list of floats.

    Raises:
        RuntimeError: If the Ollama API call fails.
    """
    if not texts:
      return []

    try:
      response_json = await asyncio.to_thread(self._post_embed, texts)
    except RuntimeError as exc:
      logger.error("Failed to generate embeddings from Ollama: %s", exc)
      raise

    embeddings = response_json.get("embeddings", [])

    # Set dimension from first embedding if not already set
    if embeddings and self._dimension is None:
      self._dimension = len(embeddings[0])

    return embeddings

  def _post_embed(self, texts: list[str]) -> dict:
    """Perform a blocking POST /api/embed call to Ollama.

    Args:
        texts: A list of strings to embed.

    Returns:
        The JSON response from Ollama.

    Raises:
        RuntimeError: If the request fails.
    """
    url = self._host.rstrip("/") + _EMBED_ENDPOINT
    payload = {
        "model": self._model,
        "input": texts,
    }
    try:
      response = requests.post(
          url,
          json=payload,
          timeout=self._request_timeout,
      )
      response.raise_for_status()
      return response.json()
    except requests.exceptions.RequestException as exc:
      raise RuntimeError(f"Failed to connect to Ollama: {exc}") from exc
