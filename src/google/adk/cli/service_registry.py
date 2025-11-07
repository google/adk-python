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

from __future__ import annotations

import os
from typing import Any
from typing import Dict
from typing import Protocol
from urllib.parse import urlparse

from ..artifacts.base_artifact_service import BaseArtifactService
from ..memory.base_memory_service import BaseMemoryService
from ..sessions.base_session_service import BaseSessionService


def _load_gcp_config(
    agents_dir: str | None, service_name: str
) -> tuple[str, str]:
  """Loads GCP project and location from environment."""
  if not agents_dir:
    raise ValueError(f"agents_dir must be provided for {service_name}")

  from .utils import envs

  envs.load_dotenv_for_agent("", agents_dir)

  project = os.environ.get("GOOGLE_CLOUD_PROJECT")
  location = os.environ.get("GOOGLE_CLOUD_LOCATION")

  if not project or not location:
    raise ValueError("GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION not set.")

  return project, location


def _parse_agent_engine_kwargs(
    uri_part: str, agents_dir: str | None
) -> dict[str, Any]:
  """Helper to parse agent engine resource name."""
  if not uri_part:
    raise ValueError(
        "Agent engine resource name or resource id cannot be empty."
    )
  if "/" in uri_part:
    parts = uri_part.split("/")
    if not (
        len(parts) == 6
        and parts[0] == "projects"
        and parts[2] == "locations"
        and parts[4] == "reasoningEngines"
    ):
      raise ValueError(
          "Agent engine resource name is mal-formatted. It should be of"
          " format :"
          " projects/{project_id}/locations/{location}/reasoningEngines/{resource_id}"
      )
    project = parts[1]
    location = parts[3]
    agent_engine_id = parts[5]
  else:
    project, location = _load_gcp_config(
        agents_dir, "short-form agent engine IDs"
    )
    agent_engine_id = uri_part
  return {
      "project": project,
      "location": location,
      "agent_engine_id": agent_engine_id,
  }


class ServiceFactory(Protocol):
  """Protocol for service factory functions."""

  def __call__(
      self, uri: str, **kwargs
  ) -> BaseSessionService | BaseArtifactService | BaseMemoryService:
    ...


class ServiceRegistry:
  """Registry for custom service URI schemes."""

  def __init__(self):
    self._session_factories: Dict[str, ServiceFactory] = {}
    self._artifact_factories: Dict[str, ServiceFactory] = {}
    self._memory_factories: Dict[str, ServiceFactory] = {}

  def register_session_service(
      self, scheme: str, factory: ServiceFactory
  ) -> None:
    """Register a factory for a custom session service URI scheme.

    Args:
        scheme: URI scheme (e.g., 'custom')
        factory: Callable that takes (uri, **kwargs) and returns
          BaseSessionService
    """
    self._session_factories[scheme] = factory

  def register_artifact_service(
      self, scheme: str, factory: ServiceFactory
  ) -> None:
    """Register a factory for a custom artifact service URI scheme."""
    self._artifact_factories[scheme] = factory

  def register_memory_service(
      self, scheme: str, factory: ServiceFactory
  ) -> None:
    """Register a factory for a custom memory service URI scheme."""
    self._memory_factories[scheme] = factory

  def create_session_service(
      self, uri: str, **kwargs
  ) -> BaseSessionService | None:
    """Create session service from URI using registered factories."""
    scheme = urlparse(uri).scheme
    if scheme and scheme in self._session_factories:
      return self._session_factories[scheme](uri, **kwargs)
    return None

  def create_artifact_service(
      self, uri: str, **kwargs
  ) -> BaseArtifactService | None:
    """Create artifact service from URI using registered factories."""
    scheme = urlparse(uri).scheme
    if scheme and scheme in self._artifact_factories:
      return self._artifact_factories[scheme](uri, **kwargs)
    return None

  def create_memory_service(
      self, uri: str, **kwargs
  ) -> BaseMemoryService | None:
    """Create memory service from URI using registered factories."""
    scheme = urlparse(uri).scheme
    if scheme and scheme in self._memory_factories:
      return self._memory_factories[scheme](uri, **kwargs)
    return None


def _register_builtin_services(registry: ServiceRegistry) -> None:
  """Register built-in service implementations."""

  # -- Session Services --
  def agentengine_session_factory(uri: str, **kwargs):
    from ..sessions.vertex_ai_session_service import VertexAiSessionService

    parsed = urlparse(uri)
    params = _parse_agent_engine_kwargs(
        parsed.netloc + parsed.path, kwargs.get("agents_dir")
    )
    return VertexAiSessionService(**params)

  def database_session_factory(uri: str, **kwargs):
    from ..sessions.database_session_service import DatabaseSessionService

    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("agents_dir", None)
    return DatabaseSessionService(db_url=uri, **kwargs_copy)

  registry.register_session_service("agentengine", agentengine_session_factory)
  for scheme in ["sqlite", "postgresql", "mysql"]:
    registry.register_session_service(scheme, database_session_factory)

  # -- Artifact Services --
  def gcs_artifact_factory(uri: str, **kwargs):
    from ..artifacts.gcs_artifact_service import GcsArtifactService

    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("agents_dir", None)
    parsed_uri = urlparse(uri)
    bucket_name = parsed_uri.netloc
    return GcsArtifactService(bucket_name=bucket_name, **kwargs_copy)

  registry.register_artifact_service("gs", gcs_artifact_factory)

  # -- Memory Services --
  def rag_memory_factory(uri: str, **kwargs):
    from ..memory.vertex_ai_rag_memory_service import VertexAiRagMemoryService

    rag_corpus = urlparse(uri).netloc
    if not rag_corpus:
      raise ValueError("Rag corpus cannot be empty.")
    agents_dir = kwargs.get("agents_dir")
    project, location = _load_gcp_config(agents_dir, "RAG memory service")
    return VertexAiRagMemoryService(
        rag_corpus=(
            f"projects/{project}/locations/{location}/ragCorpora/{rag_corpus}"
        )
    )

  def agentengine_memory_factory(uri: str, **kwargs):
    from ..memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService

    parsed = urlparse(uri)
    params = _parse_agent_engine_kwargs(
        parsed.netloc + parsed.path, kwargs.get("agents_dir")
    )
    return VertexAiMemoryBankService(**params)

  registry.register_memory_service("rag", rag_memory_factory)
  registry.register_memory_service("agentengine", agentengine_memory_factory)


# Global registry instance
_global_registry = ServiceRegistry()
_register_builtin_services(_global_registry)


def get_service_registry() -> ServiceRegistry:
  """Get the global service registry instance."""
  return _global_registry
