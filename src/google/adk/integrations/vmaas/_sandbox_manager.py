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

"""Creates and manages Agent Engine sandboxes from reusable templates.

Agent Engine runs sandboxes created from a template as gVisor pods on GKE.
Creating a sandbox from an inline spec instead makes the client SDK provision
a new template on every call, so this module finds or creates one template per
container configuration, reuses it for every sandbox of that configuration,
and creates sandboxes only from a template or a snapshot.
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
_SANDBOX_DISPLAY_NAME = "adk-sandbox"
_SANDBOX_STATE_PAUSED = "STATE_PAUSED"
_SANDBOX_STATE_PAUSING = "STATE_PAUSING"
_SANDBOX_STATE_TERMINATED = "STATE_TERMINATED"

# Templates and sandboxes take a while to provision, so poll less often than
# the SDK default of every 0.1 seconds.
_TEMPLATE_POLL_INTERVAL_SECONDS = 1.0
_SANDBOX_POLL_INTERVAL_SECONDS = 0.5


class SandboxManager:
  """Creates Agent Engine sandboxes and the resources they are created from.

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

  Sandboxes are created under the agent engine that owns their template or
  snapshot. Pausing and resuming a sandbox through one manager is serialized
  per sandbox, so concurrent callers resume a paused sandbox only once.
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
    self._sandbox_locks: collections.defaultdict[str, asyncio.Lock] = (
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

  async def create_sandbox(
      self,
      *,
      template_name: str | None = None,
      snapshot_name: str | None = None,
      display_name: str = _SANDBOX_DISPLAY_NAME,
      ttl_seconds: int | None = None,
  ) -> str:
    """Creates a sandbox from a template or a snapshot and returns its name.

    Exactly one of ``template_name`` and ``snapshot_name`` must be set. The
    sandbox is created under the agent engine that owns it.

    Args:
      template_name: Template to create the sandbox from.
      snapshot_name: Snapshot to restore the sandbox from.
      display_name: Display name of the sandbox.
      ttl_seconds: Time to live of the sandbox. If None, the service default
        applies.
    """
    if template_name and not snapshot_name:
      source_field, source_name = "sandbox_environment_template", template_name
    elif snapshot_name and not template_name:
      source_field, source_name = "sandbox_environment_snapshot", snapshot_name
    else:
      raise ValueError(
          "Exactly one of template_name and snapshot_name must be set."
      )
    config: dict[str, Any] = {
        "display_name": display_name,
        source_field: source_name,
    }
    if ttl_seconds is not None:
      config["ttl"] = f"{ttl_seconds}s"

    logger.info("Creating sandbox from %s", source_name)
    operation = await asyncio.to_thread(
        self._get_client().sandboxes.create,
        name=_parent_agent_engine_name(source_name),
        config=config,
        poll_interval_seconds=_SANDBOX_POLL_INTERVAL_SECONDS,
    )
    sandbox = operation.response
    if sandbox is None or _state(sandbox) == _SANDBOX_STATE_TERMINATED:
      raise RuntimeError(
          f"Failed to create a sandbox from {source_name!r}: {operation.error}"
      )
    sandbox_name: str = sandbox.name
    logger.info("Created sandbox: %s", sandbox_name)
    return sandbox_name

  async def get_sandbox_state(self, sandbox_name: str) -> str:
    """Returns the state of a sandbox, e.g. ``STATE_RUNNING``."""
    sandbox = await asyncio.to_thread(
        self._get_client().sandboxes.get, name=sandbox_name
    )
    return _state(sandbox)

  async def pause_sandbox(self, sandbox_name: str) -> bool:
    """Pauses a sandbox, keeping its filesystem for a later resume.

    Returns:
      Whether the sandbox was paused by this call. A sandbox that is already
      paused or pausing is left as is.
    """
    async with self._sandbox_locks[sandbox_name]:
      state = await self.get_sandbox_state(sandbox_name)
      if state in (_SANDBOX_STATE_PAUSED, _SANDBOX_STATE_PAUSING):
        return False
      logger.info("Pausing sandbox: %s", sandbox_name)
      operation = await asyncio.to_thread(
          self._get_client().sandboxes.pause,
          name=sandbox_name,
          poll_interval_seconds=_SANDBOX_POLL_INTERVAL_SECONDS,
      )
      if operation.error:
        raise RuntimeError(
            f"Failed to pause sandbox {sandbox_name}: {operation.error}"
        )
      return True

  async def resume_sandbox_if_paused(self, sandbox_name: str) -> bool:
    """Resumes a sandbox if it is paused.

    Returns:
      Whether the sandbox was resumed by this call.
    """
    async with self._sandbox_locks[sandbox_name]:
      if await self.get_sandbox_state(sandbox_name) != _SANDBOX_STATE_PAUSED:
        return False
      logger.info("Resuming sandbox: %s", sandbox_name)
      operation = await asyncio.to_thread(
          self._get_client().sandboxes.resume,
          name=sandbox_name,
          poll_interval_seconds=_SANDBOX_POLL_INTERVAL_SECONDS,
      )
      if operation.error:
        raise RuntimeError(
            f"Failed to resume sandbox {sandbox_name}: {operation.error}"
        )
      return True

  async def delete_sandbox(self, sandbox_name: str) -> None:
    """Deletes a sandbox, after any pause or resume of it has finished.

    The service deletes a sandbox in any state but PROVISIONING, including one
    that is still resuming, so a deletion waits for this manager's own pause or
    resume of the sandbox to finish first.
    """
    async with self._sandbox_locks[sandbox_name]:
      logger.info("Deleting sandbox: %s", sandbox_name)
      await asyncio.to_thread(
          self._get_client().sandboxes.delete, name=sandbox_name
      )
    self._sandbox_locks.pop(sandbox_name, None)

  def _get_client(self) -> agentplatform.Client:
    if self._client is None:
      from ...dependencies._agentplatform import agentplatform

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


def _parent_agent_engine_name(resource_name: str) -> str:
  agent_engine_name, separator, _ = resource_name.partition(
      "/sandboxEnvironment"
  )
  if not separator or "/reasoningEngines/" not in agent_engine_name:
    raise ValueError(
        f"{resource_name!r} is not an Agent Engine sandbox template or"
        " snapshot name."
    )
  return agent_engine_name


def _state(sandbox: Any) -> str:
  # The SDK reports states as string-valued enums.
  state = sandbox.state
  if state is None:
    return "STATE_UNSPECIFIED"
  value: str = getattr(state, "value", state)
  return value


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
