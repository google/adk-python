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

"""Integration tests for AgentEngineEnvironment against live Agent Engine.

Each test creates a sandbox and deletes it when done. The tests share one
agent engine, so the shell sandbox template is created at most once per run.

Required environment variables:
  ADK_AGENT_ENGINE_TEST_PROJECT  — Google Cloud project with Agent Engine
                                   sandboxes enabled

Optional:
  ADK_AGENT_ENGINE_TEST_LOCATION      — location (default: us-central1)
  ADK_AGENT_ENGINE_TEST_AGENT_ENGINE  — existing agent engine to use, so its
      shell template is reused across runs. If unset, the tests create an
      agent engine and delete it, with its template, at the end.

Run with Application Default Credentials, for example:
  ADK_AGENT_ENGINE_TEST_PROJECT=my-project \\
      pytest -s tests/integration/integrations/vmaas
"""

import contextlib
import hashlib
import os
import time

from google.adk.integrations.vmaas import AgentEngineEnvironment
from google.adk.integrations.vmaas._sandbox_manager import SandboxManager
import pytest

_PROJECT = os.environ.get("ADK_AGENT_ENGINE_TEST_PROJECT", "")
_LOCATION = os.environ.get("ADK_AGENT_ENGINE_TEST_LOCATION", "us-central1")
_AGENT_ENGINE = os.environ.get("ADK_AGENT_ENGINE_TEST_AGENT_ENGINE")

# Sandboxes left behind by a failed teardown expire on their own.
_SANDBOX_TTL_SECONDS = 1800

pytestmark = pytest.mark.skipif(
    not _PROJECT,
    reason=(
        "Agent Engine integration tests require ADK_AGENT_ENGINE_TEST_PROJECT"
        " to be set."
    ),
)


# The integration conftest parametrizes its autouse ``llm_backend`` fixture
# across GOOGLE_AI and VERTEX. These tests use no LLM, so the override below
# stops the fixture's environment changes, and ``_SINGLE_RUN``, applied to every
# test, pins it to one value so each test (and each sandbox) runs once.
@pytest.fixture(autouse=True)
def llm_backend():
  yield


_SINGLE_RUN = pytest.mark.parametrize(
    "llm_backend", [None], indirect=True, ids=[""]
)


@pytest.fixture(scope="module")
def agent_engine_name():
  """Returns the agent engine the tests share, creating one if none is given."""
  if _AGENT_ENGINE:
    yield _AGENT_ENGINE
    return
  from google.adk.dependencies._agentplatform import agentplatform

  client = agentplatform.Client(project=_PROJECT, location=_LOCATION)
  agent_engine = client.runtimes.create(
      config={"display_name": "adk-integration-test"}
  )
  name = agent_engine.api_resource.name
  print(f"\nCreated agent engine {name}")
  yield name
  client.runtimes.delete(name=name, force=True)


@contextlib.contextmanager
def _timed(label):
  """Prints how long the block took, as a latency baseline."""
  start = time.monotonic()
  yield
  print(f"\n[latency] {label}: {time.monotonic() - start:.1f}s")


@contextlib.asynccontextmanager
async def _environment(agent_engine_name):
  """Yields an initialized environment and deletes its sandbox afterwards."""
  environment = AgentEngineEnvironment(
      project_id=_PROJECT,
      location=_LOCATION,
      agent_engine_name=agent_engine_name,
      sandbox_ttl_seconds=_SANDBOX_TTL_SECONDS,
  )
  try:
    with _timed("initialize (template + sandbox create)"):
      await environment.initialize()
    yield environment
  finally:
    if environment.sandbox_name:
      await SandboxManager(
          project_id=_PROJECT, location=_LOCATION
      ).delete_sandbox(environment.sandbox_name)


@_SINGLE_RUN
async def test_commands_and_files_survive_close_and_initialize(
    agent_engine_name,
):
  """Commands run and files persist across a pause and resume.

  Setup: a new sandbox from the agent engine's shell template.
  Act: run a command, write and read a file, close (pause), and initialize
    again (resume).
  Assert: output and exit codes are reported, the file round-trips, the
    sandbox is paused after close, and the file is still there after resume.
  """
  async with _environment(agent_engine_name) as environment:
    manager = SandboxManager(project_id=_PROJECT, location=_LOCATION)
    content = bytes(range(256)) * 64

    with _timed("first execute"):
      result = await environment.execute("echo out; echo err >&2; exit 3")
    with _timed("second execute"):
      await environment.execute("true")
    await environment.write_file("data/blob.bin", content)
    read_back = await environment.read_file("data/blob.bin")
    with pytest.raises(FileNotFoundError):
      await environment.read_file("data/missing.txt")
    with _timed("close (pause)"):
      await environment.close()
    state_after_close = await manager.get_sandbox_state(
        environment.sandbox_name
    )
    with _timed("initialize (resume)"):
      await environment.initialize()
    read_after_resume = await environment.read_file("data/blob.bin")

    assert (result.stdout, result.stderr, result.exit_code) == (
        "out\n",
        "err\n",
        3,
    )
    assert read_back == content
    assert state_after_close == "STATE_PAUSED"
    assert read_after_resume == content


@_SINGLE_RUN
async def test_command_to_paused_sandbox_resumes_it(agent_engine_name):
  """A command sent to a sandbox paused out of band resumes it and runs."""
  async with _environment(agent_engine_name) as environment:
    manager = SandboxManager(project_id=_PROJECT, location=_LOCATION)
    await manager.pause_sandbox(environment.sandbox_name)

    with _timed("execute on paused sandbox (auto-resume)"):
      result = await environment.execute("echo resumed")

    assert result.stdout == "resumed\n"
    assert await manager.get_sandbox_state(environment.sandbox_name) == (
        "STATE_RUNNING"
    )


@_SINGLE_RUN
async def test_large_files_round_trip(agent_engine_name):
  """Files larger than one command are written and read in chunks.

  Act: write a 1 MiB file, which takes several commands, and read it back;
    create a 3 MiB file in the sandbox and read it, which takes several
    slices.
  Assert: both files read back unchanged, and no temporary file is left.
  """
  async with _environment(agent_engine_name) as environment:
    content = os.urandom(1024 * 1024)

    with _timed("write 1 MiB"):
      await environment.write_file("large/one.bin", content)
    with _timed("read 1 MiB"):
      read_back = await environment.read_file("large/one.bin")
    await environment.execute(
        "head -c 3145728 /dev/urandom > large/three.bin"
        " && sha256sum large/three.bin > large/three.sha256"
    )
    with _timed("read 3 MiB"):
      three = await environment.read_file("large/three.bin")
    expected_digest = (
        await environment.read_file("large/three.sha256")
    ).split()[0]
    listing = await environment.execute("ls -A large")

    assert read_back == content
    assert len(three) == 3 * 1024 * 1024
    assert hashlib.sha256(three).hexdigest().encode() == expected_digest
    assert listing.stdout.split() == ["one.bin", "three.bin", "three.sha256"]


@_SINGLE_RUN
async def test_command_past_its_timeout_is_reported_timed_out(
    agent_engine_name,
):
  """A command that runs past its timeout is reported as timed out."""
  async with _environment(agent_engine_name) as environment:
    result = await environment.execute("sleep 30", timeout=2)

    assert result.timed_out
