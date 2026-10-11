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

"""Tests for SandboxManager.

Verifies that the manager resolves one agent engine and reuses one sandbox
template per container configuration.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from google.adk.integrations.vmaas._sandbox_manager import SandboxManager
from google.genai.errors import ClientError
import pytest

_ENGINE = "projects/p/locations/us-central1/reasoningEngines/1"
_IMAGE = "us-central1-docker.pkg.dev/p/repo/image:1"


class _FakeTemplates:
  """In-memory stand-in for the sandbox templates API."""

  def __init__(self):
    self.templates = []
    self.create_calls = []
    self.list_calls = 0

  def list(self, *, name):
    self.list_calls += 1
    return [t for t in self.templates if t.name.startswith(name + "/")]

  def get(self, *, name):
    for template in self.templates:
      if template.name == name:
        return template
    raise ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND"}})

  def create(self, *, name, display_name, config, poll_interval_seconds):
    self.create_calls.append(
        {"name": name, "display_name": display_name, "config": config}
    )
    template_id = len(self.create_calls)
    template = SimpleNamespace(
        name=f"{name}/sandboxEnvironmentTemplates/{template_id}",
        display_name=display_name,
        state="ACTIVE",
    )
    self.templates.append(template)
    return SimpleNamespace(response=template, error=None)


def _make_client():
  client = MagicMock()
  client.runtimes.create.return_value = SimpleNamespace(
      api_resource=SimpleNamespace(name=_ENGINE)
  )
  client.sandboxes.templates = _FakeTemplates()
  return client


async def test_shell_template_uses_shell_container_category():
  """A shell template is created from the default shell sandbox container."""
  client = _make_client()

  template_name = await SandboxManager(client=client).get_shell_template_name()

  [call] = client.sandboxes.templates.create_calls
  assert call["name"] == _ENGINE
  assert call["config"] == {
      "default_container_environment": {
          "default_container_category": (
              "DEFAULT_CONTAINER_CATEGORY_SHELL_SANDBOX"
          )
      }
  }
  assert template_name == f"{_ENGINE}/sandboxEnvironmentTemplates/1"


async def test_custom_container_template_includes_image_ports_and_resources():
  """A custom container template carries the image, ports, and resources."""
  client = _make_client()
  manager = SandboxManager(client=client)

  await manager.get_custom_container_template_name(
      image_uri=_IMAGE,
      ports=[8080],
      resources={"limits": {"cpu": "2"}},
  )

  [call] = client.sandboxes.templates.create_calls
  assert call["config"] == {
      "custom_container_environment": {
          "custom_container_spec": {"image_uri": _IMAGE},
          "ports": [{"port": 8080}],
          "resources": {"limits": {"cpu": "2"}},
      }
  }


async def test_another_manager_reuses_active_template_on_same_engine():
  """A manager reuses the template another manager created on its engine."""
  client = _make_client()
  first = await SandboxManager(
      client=client, agent_engine_name=_ENGINE
  ).get_shell_template_name()

  second = await SandboxManager(
      client=client, agent_engine_name=_ENGINE
  ).get_shell_template_name()

  assert second == first
  assert len(client.sandboxes.templates.create_calls) == 1


async def test_template_that_is_not_active_is_not_reused():
  """A template that is not ACTIVE is replaced by a newly created one."""
  client = _make_client()
  stale = await SandboxManager(
      client=client, agent_engine_name=_ENGINE
  ).get_shell_template_name()
  client.sandboxes.templates.templates[0].state = "FAILED"

  fresh = await SandboxManager(
      client=client, agent_engine_name=_ENGINE
  ).get_shell_template_name()

  assert fresh != stale
  assert len(client.sandboxes.templates.create_calls) == 2


async def test_resolved_template_is_cached_by_the_manager():
  """Resolving the same template again reuses it without listing templates."""
  client = _make_client()
  manager = SandboxManager(client=client)
  first = await manager.get_shell_template_name()

  second = await manager.get_shell_template_name()

  assert second == first
  assert client.sandboxes.templates.list_calls == 1


@pytest.mark.parametrize(
    "make_stale",
    [
        lambda templates: templates.clear(),
        lambda templates: setattr(templates[0], "state", "DELETED"),
    ],
    ids=["deleted", "not_active"],
)
async def test_cached_template_that_is_gone_or_not_active_is_replaced(
    make_stale,
):
  """A cached template that was deleted or is no longer ACTIVE is replaced."""
  client = _make_client()
  manager = SandboxManager(client=client)
  stale = await manager.get_shell_template_name()
  make_stale(client.sandboxes.templates.templates)

  fresh = await manager.get_shell_template_name()

  assert fresh != stale
  assert len(client.sandboxes.templates.create_calls) == 2


async def test_error_checking_cached_template_is_raised():
  """An error other than not found while checking a cached template raises."""
  client = _make_client()
  manager = SandboxManager(client=client)
  await manager.get_shell_template_name()
  client.sandboxes.templates.get = MagicMock(
      side_effect=ClientError(403, {"error": {"code": 403}})
  )

  with pytest.raises(ClientError):
    await manager.get_shell_template_name()


async def test_template_being_created_does_not_block_other_templates():
  """Creating one template does not hold up resolving other templates.

  Setup: the shell template is cached, and creating the template for _IMAGE
    blocks until released.
  Act: while that creation blocks, resolve the shell template and the template
    of another image.
  Assert: both resolve while the blocked creation is still running.
  """
  client = _make_client()
  manager = SandboxManager(client=client)
  shell = await manager.get_shell_template_name()
  templates = client.sandboxes.templates
  create = templates.create
  started = threading.Event()
  release = threading.Event()

  def blocking_create(**kwargs):
    environment = kwargs["config"].get("custom_container_environment", {})
    if environment.get("custom_container_spec") == {"image_uri": _IMAGE}:
      started.set()
      release.wait(timeout=10)
    return create(**kwargs)

  templates.create = blocking_create
  blocked = asyncio.create_task(
      manager.get_custom_container_template_name(image_uri=_IMAGE)
  )
  try:
    await asyncio.to_thread(started.wait, 5)
    cached, other = await asyncio.wait_for(
        asyncio.gather(
            manager.get_shell_template_name(),
            manager.get_custom_container_template_name(
                image_uri="us-central1-docker.pkg.dev/p/repo/image:2"
            ),
        ),
        timeout=5,
    )
    blocked_was_running = not blocked.done()
  finally:
    release.set()
    await blocked

  assert cached == shell
  assert other not in (shell, blocked.result())
  assert blocked_was_running


async def test_concurrent_requests_create_one_template():
  """Concurrent requests for the same template create it only once."""
  client = _make_client()
  manager = SandboxManager(client=client)

  names = await asyncio.gather(
      *(manager.get_shell_template_name() for _ in range(5))
  )

  assert len(set(names)) == 1
  assert len(client.sandboxes.templates.create_calls) == 1


async def test_different_configurations_get_different_templates():
  """Templates differing in container configuration are not shared."""
  client = _make_client()
  manager = SandboxManager(client=client)

  first = await manager.get_custom_container_template_name(image_uri=_IMAGE)
  second = await manager.get_custom_container_template_name(
      image_uri=_IMAGE, ports=[8080]
  )

  assert first != second
  [first_call, second_call] = client.sandboxes.templates.create_calls
  assert first_call["display_name"] != second_call["display_name"]


async def test_agent_engine_is_created_once_when_not_given():
  """Without an agent engine, one is created and shared by all templates."""
  client = _make_client()
  manager = SandboxManager(client=client)

  await asyncio.gather(
      manager.get_shell_template_name(),
      manager.get_custom_container_template_name(image_uri=_IMAGE),
  )

  client.runtimes.create.assert_called_once()
  assert await manager.get_agent_engine_name() == _ENGINE


async def test_given_agent_engine_is_used_without_creating_one():
  """A given agent engine is used for templates and no engine is created."""
  client = _make_client()
  engine = "projects/p/locations/us-central1/reasoningEngines/2"

  await SandboxManager(
      client=client, agent_engine_name=engine
  ).get_shell_template_name()

  client.runtimes.create.assert_not_called()
  [call] = client.sandboxes.templates.create_calls
  assert call["name"] == engine


async def test_failed_template_creation_raises():
  """A template that fails to provision raises instead of being used."""
  client = _make_client()
  client.sandboxes.templates = MagicMock()
  client.sandboxes.templates.list.return_value = []
  client.sandboxes.templates.create.return_value = SimpleNamespace(
      response=SimpleNamespace(name="t", state="FAILED"), error=None
  )
  manager = SandboxManager(client=client)

  with pytest.raises(RuntimeError, match="Failed to create sandbox template"):
    await manager.get_shell_template_name()


async def test_custom_container_without_image_raises():
  """A custom container template requires an image URI."""
  manager = SandboxManager(client=_make_client())

  with pytest.raises(ValueError, match="image_uri"):
    await manager.get_custom_container_template_name(image_uri="")
