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

"""Base class for embedding providers."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseEmbeddingProvider(ABC):
  """Abstract base class for embedding providers.

  Embedding providers are responsible for converting text into vector
  representations for use in semantic search.
  """

  @abstractmethod
  async def embed(self, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: A list of strings to embed.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """

  @property
  @abstractmethod
  def dimension(self) -> int:
    """Return the dimension of the embedding vectors."""
