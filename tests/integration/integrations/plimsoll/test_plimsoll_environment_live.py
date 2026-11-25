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

"""Integration tests for PlimsollEnvironment against a live Plimsoll daemon.

Each test opens a session and closes it when done. No model is called.

Required environment variables:
  PLIMSOLL_URL    the daemon's address, for example http://127.0.0.1:8746
  PLIMSOLL_TOKEN  the caller token the daemon was started with

Optional:
  PLIMSOLL_MIN_ISOLATION  the floor to open at (default: kernel). A Docker
                          daemon without gVisor states the container tier,
                          which needs PLIMSOLL_MIN_ISOLATION=container.

A daemon with sessions and python3 in its project image, from a Plimsoll
checkout after `make docker-images`:
  SANDBOX_PROVIDER=docker SANDBOX_DOCKER_PROJECT_IMAGE=plimsoll/sandbox-python \\
      SANDBOX_MAX_SESSIONS=4 PLIMSOLL_ADDR=127.0.0.1:8746 go run ./cmd/plimsolld

Run:
  PLIMSOLL_URL=http://127.0.0.1:8746 PLIMSOLL_TOKEN=... \\
      pytest -s tests/integration/integrations/plimsoll
"""

import os
from pathlib import Path

from google.adk.integrations.plimsoll import PlimsollEnvironment
import pytest

_URL = os.environ.get("PLIMSOLL_URL", "")
_FLOOR = os.environ.get("PLIMSOLL_MIN_ISOLATION", "kernel")

pytestmark = pytest.mark.skipif(
    not _URL,
    reason="Plimsoll integration tests require PLIMSOLL_URL to be set.",
)


# The integration conftest parametrizes its autouse ``llm_backend`` fixture
# across GOOGLE_AI and VERTEX. These tests use no LLM: the override stops the
# fixture's environment changes, and ``_SINGLE_RUN`` runs each test once.
@pytest.fixture(autouse=True)
def llm_backend():
  yield


_SINGLE_RUN = pytest.mark.parametrize(
    "llm_backend", [None], indirect=True, ids=[""]
)


@_SINGLE_RUN
@pytest.mark.asyncio
async def test_files_persist_between_commands():
  env = PlimsollEnvironment(minimum_isolation=_FLOOR)
  await env.initialize()
  try:
    assert env.working_dir is not None
    assert env.isolation is not None

    await env.write_file(Path("sub/answer.py"), "print(6 * 7)\n")
    result = await env.execute("python3 sub/answer.py")
    assert result.exit_code == 0
    assert result.stdout == "42\n"

    await env.write_file(Path("blob.bin"), bytes(range(256)))
    assert await env.read_file(Path("blob.bin")) == bytes(range(256))

    failed = await env.execute("echo oops >&2; exit 3")
    assert failed.exit_code == 3
    assert "oops" in failed.stderr

    with pytest.raises(FileNotFoundError):
      await env.read_file(Path("missing.txt"))
  finally:
    await env.close()
  assert env.is_initialized is False


@_SINGLE_RUN
@pytest.mark.asyncio
async def test_command_timeout_is_reported():
  env = PlimsollEnvironment(minimum_isolation=_FLOOR)
  await env.initialize()
  try:
    result = await env.execute("sleep 30", timeout=2)
    assert result.timed_out is True
  finally:
    await env.close()
