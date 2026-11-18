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

Verifies that the manager resolves one agent engine, reuses one sandbox
template per container configuration, and creates, pauses, resumes, and
deletes sandboxes created only from templates or snapshots.
"""

import asyncio
import enum
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from google.adk.integrations.vmaas import _sandbox_manager
from google.adk.integrations.vmaas._sandbox_manager import SandboxManager
from google.genai.errors import ClientError
import pytest

_ENGINE = "projects/p/locations/us-central1/reasoningEngines/1"
_IMAGE = "us-central1-docker.pkg.dev/p/repo/image:1"
_OTHER_ENGINE = "projects/p/locations/us-central1/reasoningEngines/9"
_TEMPLATE = f"{_OTHER_ENGINE}/sandboxEnvironmentTemplates/7"
_SNAPSHOT = f"{_OTHER_ENGINE}/sandboxEnvironmentSnapshots/8"


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


class _FakeSandboxes:
  """In-memory stand-in for the sandboxes API."""

  def __init__(self):
    self.templates = _FakeTemplates()
    self.sandboxes = {}
    self.create_calls = []
    self.pause_calls = []
    self.resume_calls = []
    self.delete_calls = []

  def create(self, **kwargs):
    self.create_calls.append(kwargs)
    name = f"{kwargs['name']}/sandboxEnvironments/{len(self.sandboxes) + 1}"
    self.sandboxes[name] = SimpleNamespace(name=name, state="STATE_RUNNING")
    return SimpleNamespace(response=self.sandboxes[name], error=None)

  def get(self, *, name):
    return self.sandboxes[name]

  def pause(self, *, name, poll_interval_seconds):
    self.pause_calls.append(name)
    self.sandboxes[name].state = "STATE_PAUSED"
    return SimpleNamespace(error=None)

  def resume(self, *, name, poll_interval_seconds):
    self.resume_calls.append(name)
    self.sandboxes[name].state = "STATE_RUNNING"
    return SimpleNamespace(error=None)

  def delete(self, *, name):
    self.delete_calls.append(name)
    del self.sandboxes[name]


def _make_client():
  client = MagicMock()
  client.runtimes.create.return_value = SimpleNamespace(
      api_resource=SimpleNamespace(name=_ENGINE)
  )
  client.sandboxes = _FakeSandboxes()
  return client


async def _make_sandbox(client, state="STATE_RUNNING"):
  """Returns the name of a sandbox in the given state."""
  name = await SandboxManager(client=client).create_sandbox(
      template_name=_TEMPLATE
  )
  client.sandboxes.sandboxes[name].state = state
  return name


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


async def test_sandbox_is_created_from_template_under_its_agent_engine():
  """A sandbox is created from the template, under the template's engine."""
  client = _make_client()

  name = await SandboxManager(client=client).create_sandbox(
      template_name=_TEMPLATE, ttl_seconds=600
  )

  assert client.sandboxes.create_calls == [{
      "name": _OTHER_ENGINE,
      "config": {
          "display_name": "adk-sandbox",
          "sandbox_environment_template": _TEMPLATE,
          "ttl": "600s",
      },
      "poll_interval_seconds": 0.5,
  }]
  assert name == f"{_OTHER_ENGINE}/sandboxEnvironments/1"
  client.runtimes.create.assert_not_called()


async def test_sandbox_is_restored_from_snapshot():
  """A sandbox created from a snapshot references the snapshot."""
  client = _make_client()

  await SandboxManager(client=client).create_sandbox(snapshot_name=_SNAPSHOT)

  [call] = client.sandboxes.create_calls
  assert call["name"] == _OTHER_ENGINE
  assert call["config"]["sandbox_environment_snapshot"] == _SNAPSHOT
  assert "spec" not in call


@pytest.mark.parametrize(
    "sources",
    [
        {},
        {"template_name": ""},
        {"template_name": _TEMPLATE, "snapshot_name": _SNAPSHOT},
    ],
)
async def test_sandbox_requires_exactly_one_template_or_snapshot(sources):
  """A sandbox is never created from an inline spec or two sources."""
  client = _make_client()

  with pytest.raises(ValueError, match="Exactly one"):
    await SandboxManager(client=client).create_sandbox(**sources)

  assert not client.sandboxes.create_calls


async def test_sandbox_from_name_outside_an_agent_engine_raises():
  """A template name that names no agent engine is rejected."""
  manager = SandboxManager(client=_make_client())

  with pytest.raises(ValueError, match="not an Agent Engine"):
    await manager.create_sandbox(template_name="templates/7")


async def test_terminated_sandbox_after_creation_raises():
  """A sandbox that terminates while being created raises."""
  client = _make_client()
  client.sandboxes.create = MagicMock(
      return_value=SimpleNamespace(
          response=SimpleNamespace(name="s", state="STATE_TERMINATED"),
          error={"message": "boom"},
      )
  )

  with pytest.raises(RuntimeError, match="boom"):
    await SandboxManager(client=client).create_sandbox(template_name=_TEMPLATE)


async def test_sandbox_state_is_reported_as_its_value():
  """The state of a sandbox is reported as the state's string value."""

  class _State(str, enum.Enum):
    STATE_PAUSED = "STATE_PAUSED"

  client = _make_client()
  name = await _make_sandbox(client, state=_State.STATE_PAUSED)

  state = await SandboxManager(client=client).get_sandbox_state(name)

  assert state == "STATE_PAUSED"
  assert type(state) is str


async def test_running_sandbox_is_paused():
  """Pausing a running sandbox pauses it."""
  client = _make_client()
  name = await _make_sandbox(client)

  paused = await SandboxManager(client=client).pause_sandbox(name)

  assert paused
  assert client.sandboxes.pause_calls == [name]


async def test_paused_sandbox_is_not_paused_again():
  """Pausing a paused sandbox leaves it as is."""
  client = _make_client()
  name = await _make_sandbox(client, state="STATE_PAUSED")

  paused = await SandboxManager(client=client).pause_sandbox(name)

  assert not paused
  assert not client.sandboxes.pause_calls


@pytest.mark.parametrize(
    "method", ["pause_sandbox", "resume_sandbox_if_paused"]
)
async def test_failed_pause_or_resume_raises(method):
  """A pause or resume operation that fails raises."""
  client = _make_client()
  state = "STATE_RUNNING" if method == "pause_sandbox" else "STATE_PAUSED"
  name = await _make_sandbox(client, state=state)
  failed = MagicMock(return_value=SimpleNamespace(error={"message": "boom"}))
  client.sandboxes.pause = client.sandboxes.resume = failed

  with pytest.raises(RuntimeError, match="boom"):
    await getattr(SandboxManager(client=client), method)(name)


async def test_paused_sandbox_is_resumed():
  """A paused sandbox is resumed."""
  client = _make_client()
  name = await _make_sandbox(client, state="STATE_PAUSED")

  resumed = await SandboxManager(client=client).resume_sandbox_if_paused(name)

  assert resumed
  assert await SandboxManager(client=client).get_sandbox_state(name) == (
      "STATE_RUNNING"
  )


async def test_running_sandbox_is_not_resumed():
  """Resuming a running sandbox does nothing."""
  client = _make_client()
  name = await _make_sandbox(client)

  resumed = await SandboxManager(client=client).resume_sandbox_if_paused(name)

  assert not resumed
  assert not client.sandboxes.resume_calls


async def test_concurrent_resumes_resume_sandbox_once():
  """Concurrent resumes of a paused sandbox resume it only once."""
  client = _make_client()
  name = await _make_sandbox(client, state="STATE_PAUSED")
  manager = SandboxManager(client=client)

  results = await asyncio.gather(
      *(manager.resume_sandbox_if_paused(name) for _ in range(5))
  )

  assert sorted(results) == [False, False, False, False, True]
  assert client.sandboxes.resume_calls == [name]


async def test_deletion_waits_for_resume_in_progress():
  """A sandbox is deleted only after a resume of it in progress finishes.

  Setup: a paused sandbox whose resume blocks until released.
  Act: start the resume, then delete the sandbox while the resume blocks.
  Assert: the deletion reaches the API only after the resume.
  """
  client = _make_client()
  name = await _make_sandbox(client, state="STATE_PAUSED")
  resume, delete = client.sandboxes.resume, client.sandboxes.delete
  started, release = threading.Event(), threading.Event()
  calls = []

  def blocking_resume(**kwargs):
    started.set()
    release.wait(timeout=10)
    calls.append("resume")
    return resume(**kwargs)

  def recording_delete(**kwargs):
    calls.append("delete")
    return delete(**kwargs)

  client.sandboxes.resume = blocking_resume
  client.sandboxes.delete = recording_delete
  manager = SandboxManager(client=client)
  resuming = asyncio.create_task(manager.resume_sandbox_if_paused(name))
  try:
    await asyncio.to_thread(started.wait, 5)
    deleting = asyncio.create_task(manager.delete_sandbox(name))
    # Give the deletion time to reach the API if nothing holds it back.
    await asyncio.sleep(0.2)
  finally:
    release.set()
  await asyncio.gather(resuming, deleting)

  assert calls == ["resume", "delete"]


async def test_sandbox_is_deleted():
  """Deleting a sandbox deletes it through the API."""
  client = _make_client()
  name = await _make_sandbox(client)

  await SandboxManager(client=client).delete_sandbox(name)

  assert client.sandboxes.delete_calls == [name]


async def test_bash_command_runs_in_sandbox():
  """A bash command is sent to the sandbox and its result is returned."""
  client = _make_client()
  client.sandboxes.execute_bash = MagicMock(
      return_value={"stdout": "hi\n", "returncode": 0}
  )

  result = await SandboxManager(client=client).execute_bash(
      "sandbox", "echo hi", timeout=5
  )

  assert result == {"stdout": "hi\n", "returncode": 0}
  client.sandboxes.execute_bash.assert_called_once_with(
      name="sandbox", command="echo hi", timeout=5
  )


@pytest.mark.parametrize(
    "state, exists",
    [
        ("STATE_RUNNING", True),
        ("STATE_PAUSED", True),
        ("STATE_TERMINATED", False),
    ],
)
async def test_sandbox_exists_until_it_terminates(state, exists):
  """A sandbox exists until it terminates."""
  client = _make_client()
  name = await _make_sandbox(client, state=state)

  assert await SandboxManager(client=client).sandbox_exists(name) is exists


async def test_deleted_sandbox_does_not_exist():
  """A sandbox the service no longer knows does not exist."""
  client = _make_client()
  client.sandboxes.get = MagicMock(
      side_effect=ClientError(404, {"error": {"code": 404}})
  )

  assert not await SandboxManager(client=client).sandbox_exists("sandbox")


async def test_waiting_for_paused_sandbox_resumes_it():
  """Waiting for a paused sandbox to run resumes it."""
  client = _make_client()
  name = await _make_sandbox(client, state="STATE_PAUSED")

  await SandboxManager(client=client).wait_until_running(name)

  assert client.sandboxes.resume_calls == [name]


async def test_waiting_for_pausing_sandbox_resumes_it_once_paused(monkeypatch):
  """A sandbox still pausing is resumed once its pause finishes."""
  monkeypatch.setattr(_sandbox_manager, "_RESUME_POLL_INTERVAL_SECONDS", 0)
  client = _make_client()
  name = await _make_sandbox(client, state="STATE_PAUSING")
  get = client.sandboxes.get

  def finish_pause_after_first_get(*, name):
    sandbox = get(name=name)
    state = sandbox.state
    if state == "STATE_PAUSING":
      sandbox.state = "STATE_PAUSED"
    return SimpleNamespace(name=name, state=state)

  client.sandboxes.get = finish_pause_after_first_get

  await SandboxManager(client=client).wait_until_running(name)

  assert client.sandboxes.resume_calls == [name]


async def test_waiting_for_terminated_sandbox_raises():
  """Waiting for a terminated sandbox raises instead of waiting."""
  client = _make_client()
  name = await _make_sandbox(client, state="STATE_TERMINATED")

  with pytest.raises(RuntimeError, match="STATE_TERMINATED"):
    await SandboxManager(client=client).wait_until_running(name)


async def test_waiting_for_sandbox_that_stays_paused_times_out(monkeypatch):
  """Waiting for a sandbox that stays paused after resuming times out."""
  monkeypatch.setattr(_sandbox_manager, "_RESUME_POLL_INTERVAL_SECONDS", 0)
  client = _make_client()
  name = await _make_sandbox(client, state="STATE_PAUSED")
  client.sandboxes.resume = MagicMock(return_value=SimpleNamespace(error=None))

  with pytest.raises(TimeoutError, match="STATE_PAUSED"):
    await SandboxManager(client=client).wait_until_running(
        name, timeout_seconds=0.1
    )

  client.sandboxes.resume.assert_called()


async def test_waiting_with_no_time_left_does_not_resume():
  """Waiting with no time left times out without resuming a paused sandbox."""
  client = _make_client()
  name = await _make_sandbox(client, state="STATE_PAUSED")

  with pytest.raises(TimeoutError):
    await SandboxManager(client=client).wait_until_running(
        name, timeout_seconds=0
    )

  assert not client.sandboxes.resume_calls


async def test_waiting_for_deleted_sandbox_raises():
  """Waiting for a sandbox the service no longer knows raises RuntimeError."""
  client = _make_client()
  client.sandboxes.get = MagicMock(
      side_effect=ClientError(404, {"error": {"code": 404}})
  )

  with pytest.raises(RuntimeError, match="has been deleted"):
    await SandboxManager(client=client).wait_until_running("sandbox")


async def test_waiting_for_sandbox_times_out():
  """Waiting for a sandbox that stays resuming times out."""
  client = _make_client()
  name = await _make_sandbox(client, state="STATE_RESUMING")

  with pytest.raises(TimeoutError):
    await SandboxManager(client=client).wait_until_running(
        name, timeout_seconds=0
    )
