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

"""Resolves the agent engine and templates Agent Engine sandboxes come from.

Agent Engine runs sandboxes created from a template as gVisor pods on GKE.
Creating a sandbox from an inline spec instead makes the client SDK provision
a new template on every call, so this module finds or creates one template per
container configuration and reuses it for every sandbox of that configuration.
"""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Mapping
from collections.abc import Sequence
import hashlib
import json
import logging
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import agentplatform

logger = logging.getLogger("google_adk." + __name__)

_AGENT_ENGINE_DISPLAY_NAME = "adk-sandboxes"
_SHELL_TEMPLATE_PREFIX = "adk-shell"
_CUSTOM_CONTAINER_TEMPLATE_PREFIX = "adk-custom-container"
_SHELL_CONTAINER_CATEGORY = "DEFAULT_CONTAINER_CATEGORY_SHELL_SANDBOX"
_TEMPLATE_STATE_ACTIVE = "ACTIVE"

# Templates take a while to provision, so poll less often than the SDK default.
_TEMPLATE_POLL_INTERVAL_SECONDS = 1.0


class SandboxManager:
  """Finds or creates the Agent Engine resources sandboxes are created from.

  A manager resolves one agent engine and caches the template it resolves for
  each container configuration. A template's display name encodes a hash of
  its container configuration, so a template is reused, by this manager or by
  any other manager on the same agent engine, whenever an ACTIVE template with
  the expected display name exists. Templates are created only when none does.
  A cached template is checked to still exist and be ACTIVE before each reuse,
  and is resolved again if it is not.

  Without an explicit agent engine, the manager creates one on first use.
  Templates are scoped to their agent engine, so pass the same
  ``agent_engine_name`` across processes to reuse templates between them.
  """

  def __init__(
      self,
      *,
      project_id: str | None = None,
      location: str = "us-central1",
      agent_engine_name: str | None = None,
      client: agentplatform.Client | None = None,
  ):
    """Initializes the manager.

    Args:
      project_id: Google Cloud project ID. If None, the client uses the
        Application Default Credentials project.
      location: Agent Engine location.
      agent_engine_name: Agent engine to create templates and sandboxes under.
        Format: projects/{project}/locations/{location}/reasoningEngines/{id}.
        If None, one is created on first use.
      client: Agent Platform client. If None, one is created on first use from
        ``project_id`` and ``location``.
    """
    self._project_id = project_id
    self._location = location
    self._agent_engine_name = agent_engine_name
    self._client = client
    self._template_names: dict[str, str] = {}
    self._agent_engine_lock = asyncio.Lock()
    # One lock per template display name, so creating the template of one
    # container configuration does not hold up the others.
    self._template_locks: collections.defaultdict[str, asyncio.Lock] = (
        collections.defaultdict(asyncio.Lock)
    )

  async def get_agent_engine_name(self) -> str:
    """Returns the agent engine name, creating the engine if none was given."""
    async with self._agent_engine_lock:
      if self._agent_engine_name is None:
        logger.info("Creating an agent engine for sandboxes")
        agent_engine = await asyncio.to_thread(
            self._get_client().runtimes.create,
            config={"display_name": _AGENT_ENGINE_DISPLAY_NAME},
        )
        self._agent_engine_name = agent_engine.api_resource.name
        logger.info("Created agent engine: %s", self._agent_engine_name)
      return self._agent_engine_name

  async def get_shell_template_name(
      self, *, resources: Mapping[str, Mapping[str, str]] | None = None
  ) -> str:
    """Returns a template for the default shell sandbox container.

    Args:
      resources: Container resource ``requests`` and ``limits``, e.g.
        ``{"limits": {"cpu": "2", "memory": "4Gi"}}``.
    """
    environment: dict[str, Any] = {
        "default_container_category": _SHELL_CONTAINER_CATEGORY
    }
    if resources:
      environment["resources"] = _to_dict(resources)
    return await self._get_template_name(
        prefix=_SHELL_TEMPLATE_PREFIX,
        config={"default_container_environment": environment},
    )

  async def get_custom_container_template_name(
      self,
      *,
      image_uri: str,
      ports: Sequence[int] = (),
      resources: Mapping[str, Mapping[str, str]] | None = None,
  ) -> str:
    """Returns a template for a custom container image.

    Args:
      image_uri: Artifact Registry URI of the container image.
      ports: Container ports to expose.
      resources: Container resource ``requests`` and ``limits``, e.g.
        ``{"limits": {"cpu": "2", "memory": "4Gi"}}``.
    """
    if not image_uri:
      raise ValueError("image_uri must not be empty.")
    environment: dict[str, Any] = {
        "custom_container_spec": {"image_uri": image_uri}
    }
    if ports:
      environment["ports"] = [{"port": port} for port in ports]
    if resources:
      environment["resources"] = _to_dict(resources)
    return await self._get_template_name(
        prefix=_CUSTOM_CONTAINER_TEMPLATE_PREFIX,
        config={"custom_container_environment": environment},
    )

  def _get_client(self) -> agentplatform.Client:
    if self._client is None:
      import agentplatform

      self._client = agentplatform.Client(
          project=self._project_id, location=self._location
      )
    return self._client

  async def _get_template_name(
      self, *, prefix: str, config: dict[str, Any]
  ) -> str:
    display_name = _template_display_name(prefix, config)
    template_name = await self._get_cached_template_name(display_name)
    if template_name is not None:
      return template_name
    async with self._template_locks[display_name]:
      # Another caller may have resolved the template while this one waited.
      template_name = self._template_names.get(display_name)
      if template_name is not None:
        return template_name
      agent_engine_name = await self.get_agent_engine_name()
      client = self._get_client()
      template_name = await asyncio.to_thread(
          _find_active_template, client, agent_engine_name, display_name
      )
      if template_name is None:
        template_name = await asyncio.to_thread(
            _create_template, client, agent_engine_name, display_name, config
        )
      self._template_names[display_name] = template_name
      return template_name

  async def _get_cached_template_name(self, display_name: str) -> str | None:
    """Returns the cached template for a display name if it is still ACTIVE."""
    template_name = self._template_names.get(display_name)
    if template_name is None:
      return None
    if await asyncio.to_thread(
        _is_template_active, self._get_client(), template_name
    ):
      return template_name
    logger.info("Sandbox template %s is no longer active", template_name)
    if self._template_names.get(display_name) == template_name:
      del self._template_names[display_name]
    return None


def _template_display_name(prefix: str, config: dict[str, Any]) -> str:
  digest = hashlib.sha256(
      json.dumps(config, sort_keys=True).encode("utf-8")
  ).hexdigest()
  return f"{prefix}-{digest[:12]}"


def _to_dict(
    resources: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
  return {key: dict(value) for key, value in resources.items()}


def _find_active_template(
    client: agentplatform.Client, agent_engine_name: str, display_name: str
) -> str | None:
  for template in client.sandboxes.templates.list(name=agent_engine_name):
    if (
        template.display_name == display_name
        and template.state == _TEMPLATE_STATE_ACTIVE
    ):
      logger.info("Reusing sandbox template: %s", template.name)
      return template.name
  return None


def _is_template_active(
    client: agentplatform.Client, template_name: str
) -> bool:
  from google.genai.errors import ClientError

  try:
    template = client.sandboxes.templates.get(name=template_name)
  except ClientError as e:
    if e.code == 404:
      return False
    raise
  return template.state == _TEMPLATE_STATE_ACTIVE


def _create_template(
    client: agentplatform.Client,
    agent_engine_name: str,
    display_name: str,
    config: dict[str, Any],
) -> str:
  logger.info(
      "Creating sandbox template %s under %s", display_name, agent_engine_name
  )
  operation = client.sandboxes.templates.create(
      name=agent_engine_name,
      display_name=display_name,
      config=config,
      poll_interval_seconds=_TEMPLATE_POLL_INTERVAL_SECONDS,
  )
  template = operation.response
  if template is None or template.state != _TEMPLATE_STATE_ACTIVE:
    raise RuntimeError(
        f"Failed to create sandbox template {display_name!r}: {operation.error}"
    )
  logger.info("Created sandbox template: %s", template.name)
  return template.name
