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
"""
ADK Service Registry.

This module manages pluggable backend services for sessions, artifacts, and memory.
ADK includes built-in support for common backends like SQLite, PostgreSQL,
GCS, and Vertex AI Agent Engine. You can also extend ADK by registering
custom services.

There are two ways to register custom services:

1. YAML Configuration (Recommended for simple cases)
   If your custom service can be instantiated with `MyService(uri="...", **kwargs)`,
   you can register it without writing Python code by creating a `services.yaml`
   or `services.yml` file in your agent directory (e.g., `my_agent/services.yaml`).

   Example `services.yaml`:
   ```yaml
   services:
     - scheme: mysession
       type: session
       class: my_package.my_module.MyCustomSessionService
     - scheme: mymemory
       type: memory
       class: my_package.other_module.MyCustomMemoryService
   ```

2. Python Registration (`services.py`)
   For more complex initialization logic, create a `services.py` file in your
   agent directory (e.g., `my_agent/services.py`). In this file, get the
   registry instance and register your custom factory functions. This file can
   be used for registration in addition to, or instead of, `services.yaml`.

   Example `services.py`:
   ```python
   from google.adk.cli.service_registry import get_service_registry
   from my_package.my_module import MyCustomSessionService

   def my_session_factory(uri: str, **kwargs):
       # custom logic
       return MyCustomSessionService(...)

   get_service_registry().register_session_service("mysession", my_session_factory)
   ```

Note: If both `services.yaml` (or `.yml`) and `services.py` are present in the
same directory, services from **both** files will be loaded. YAML files are
processed first, then `services.py`. If the same service scheme is defined in
both, the definition in `services.py` will overwrite the one from YAML.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
import sys
from typing import Any
from typing import Optional
from typing import Protocol
from urllib.parse import unquote
from urllib.parse import urlparse

from ..artifacts.base_artifact_service import BaseArtifactService
from ..memory.base_memory_service import BaseMemoryService
from ..sessions.base_session_service import BaseSessionService
from ..utils import yaml_utils

logger = logging.getLogger("google_adk." + __name__)


class ServiceFactory(Protocol):
  """Protocol for service factory functions."""

  def __call__(
      self, uri: str, **kwargs
  ) -> BaseSessionService | BaseArtifactService | BaseMemoryService:
    ...


class ServiceRegistry:
  """Registry for custom service URI schemes."""

  def __init__(self):
    self._session_factories: dict[str, ServiceFactory] = {}
    self._artifact_factories: dict[str, ServiceFactory] = {}
    self._memory_factories: dict[str, ServiceFactory] = {}

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


def get_service_registry() -> ServiceRegistry:
  """Gets the singleton ServiceRegistry instance, initializing it if needed."""
  global _service_registry_instance
  if _service_registry_instance is None:
    _service_registry_instance = ServiceRegistry()
    _register_builtin_services(_service_registry_instance)
  return _service_registry_instance


def load_services_module(agents_dir: str) -> None:
  """Load services.py or services.yaml from agents_dir for custom service registration.

  If services.yaml or services.yml is found, it will be loaded first,
  followed by services.py if it exists.

  Skip if neither services.yaml/yml nor services.py is not found.
  """
  if not os.path.isdir(agents_dir):
    logger.debug(
        "agents_dir %s is not a valid directory, skipping service loading.",
        agents_dir,
    )
    return
  if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)

  # Try loading services.yaml or services.yml first
  for yaml_file in ["services.yaml", "services.yml"]:
    yaml_path = os.path.join(agents_dir, yaml_file)
    if os.path.exists(yaml_path):
      try:
        config = yaml_utils.load_yaml_file(yaml_path)
        _register_services_from_yaml_config(config, get_service_registry())
        logger.debug(
            "Loaded custom services from %s in %s.", yaml_file, agents_dir
        )
      except Exception as e:
        logger.warning(
            "Failed to load %s from %s: %s",
            yaml_file,
            agents_dir,
            e,
        )
        return  # If yaml exists but fails to load, stop.

  try:
    importlib.import_module("services")
    logger.debug(
        "Loaded services.py from %s for custom service registration.",
        agents_dir,
    )
  except ModuleNotFoundError:
    logger.debug("services.py not found in %s, skipping.", agents_dir)
  except Exception as e:
    logger.warning(
        "Failed to load services.py from %s: %s",
        agents_dir,
        e,
    )


_service_registry_instance: ServiceRegistry | None = None


def _register_builtin_services(registry: ServiceRegistry) -> None:
  """Register built-in service implementations."""

  # -- Session Services --
  def memory_session_factory(uri: str, **kwargs):
    from ..sessions.in_memory_session_service import InMemorySessionService

    return InMemorySessionService()

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

  def sqlite_session_factory(uri: str, **kwargs):
    from ..sessions.sqlite_session_service import SqliteSessionService

    parsed = urlparse(uri)
    db_path = parsed.path
    if not db_path:
      # Treat sqlite:// without a path as an in-memory session service.
      return memory_session_factory("memory://", **kwargs)
    elif db_path.startswith("/"):
      db_path = db_path[1:]
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("agents_dir", None)
    return SqliteSessionService(db_path=db_path, **kwargs_copy)

  registry.register_session_service("memory", memory_session_factory)
  registry.register_session_service("agentengine", agentengine_session_factory)
  registry.register_session_service("sqlite", sqlite_session_factory)
  for scheme in ["postgresql", "mysql"]:
    registry.register_session_service(scheme, database_session_factory)

  # -- Artifact Services --
  def memory_artifact_factory(uri: str, **kwargs):
    from ..artifacts.in_memory_artifact_service import InMemoryArtifactService

    return InMemoryArtifactService()

  def gcs_artifact_factory(uri: str, **kwargs):
    from ..artifacts.gcs_artifact_service import GcsArtifactService

    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("agents_dir", None)
    kwargs_copy.pop("per_agent", None)
    parsed_uri = urlparse(uri)
    bucket_name = parsed_uri.netloc
    return GcsArtifactService(bucket_name=bucket_name, **kwargs_copy)

  def file_artifact_factory(uri: str, **_):
    from ..artifacts.file_artifact_service import FileArtifactService

    parsed_uri = urlparse(uri)
    if parsed_uri.netloc not in ("", "localhost"):
      raise ValueError(
          "file:// artifact URIs must reference the local filesystem."
      )
    if not parsed_uri.path:
      raise ValueError("file:// artifact URIs must include a path component.")
    artifact_path = Path(unquote(parsed_uri.path))
    return FileArtifactService(root_dir=artifact_path)

  registry.register_artifact_service("memory", memory_artifact_factory)
  registry.register_artifact_service("gs", gcs_artifact_factory)
  registry.register_artifact_service("file", file_artifact_factory)

  # -- Memory Services --
  def rag_memory_factory(uri: str, **kwargs):
    from ..memory.vertex_ai_rag_memory_service import VertexAiRagMemoryService

    rag_corpus = urlparse(uri).netloc
    if not rag_corpus:
      raise ValueError("Rag corpus can not be empty.")
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


def _load_gcp_config(
    agents_dir: Optional[str], service_name: str
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
    uri_part: str, agents_dir: Optional[str]
) -> dict[str, Any]:
  """Helper to parse agent engine resource name."""
  if not uri_part:
    raise ValueError(
        "Agent engine resource name or resource id cannot be empty."
    )

  # If uri_part is just an ID, load project/location from env
  if "/" not in uri_part:
    project, location = _load_gcp_config(
        agents_dir, "short-form agent engine IDs"
    )
    return {
        "project": project,
        "location": location,
        "agent_engine_id": uri_part,
    }

  # If uri_part is a full resource name, parse it
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
  return {
      "project": parts[1],
      "location": parts[3],
      "agent_engine_id": parts[5],
  }


def _get_class_from_string(class_path: str) -> Any:
  """Dynamically import a class from a string path."""
  try:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
  except Exception as e:
    raise ImportError(f"Could not import class {class_path}: {e}") from e


def _create_generic_factory(class_path: str) -> ServiceFactory:
  """Create a generic factory for a service class."""
  cls = _get_class_from_string(class_path)

  def factory(uri: str, **kwargs):
    return cls(uri=uri, **kwargs)

  return factory


def _register_services_from_yaml_config(
    config: dict[str, Any], registry: ServiceRegistry
) -> None:
  """Register services defined in a YAML configuration."""
  if not config or "services" not in config:
    return

  for service_config in config["services"]:
    scheme = service_config.get("scheme")
    service_type = service_config.get("type")
    class_path = service_config.get("class")

    if not all([scheme, service_type, class_path]):
      logger.warning("Invalid service config in YAML: %s", service_config)
      continue

    factory = _create_generic_factory(class_path)
    if service_type == "session":
      registry.register_session_service(scheme, factory)
    elif service_type == "artifact":
      registry.register_artifact_service(scheme, factory)
    elif service_type == "memory":
      registry.register_memory_service(scheme, factory)
    else:
      logger.warning("Unknown service type in YAML: %s", service_type)
