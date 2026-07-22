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

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.adk.cli.fast_api import get_fast_api_app
import pytest


@pytest.fixture
def test_client(tmp_path):
  """Client with a temporary agents directory."""
  app = get_fast_api_app(
      agents_dir=str(tmp_path),
      web=True,
      session_service_uri="",
      artifact_service_uri="",
      memory_service_uri="",
      allow_origins=["*"],
      a2a=False,
      host="127.0.0.1",
      port=8000,
  )
  return TestClient(app)


def test_list_tests_empty(test_client):
  response = test_client.get("/dev/apps/test_app/tests")
  assert response.status_code == 200
  assert response.json() == []


def test_create_test(test_client, tmp_path):
  # Create agent dir so it exists
  agent_dir = tmp_path / "test_app"
  agent_dir.mkdir()

  payload = {"session_data": {"events": []}}

  response = test_client.put(
      "/dev/apps/test_app/tests/my_test.json", json=payload
  )
  assert response.status_code == 200
  assert response.json() == {"status": "success", "file": "my_test.json"}

  # Verify file exists
  assert (agent_dir / "tests" / "my_test.json").exists()


def test_list_tests_not_empty(test_client, tmp_path):
  agent_dir = tmp_path / "test_app"
  tests_dir = agent_dir / "tests"
  tests_dir.mkdir(parents=True)
  (tests_dir / "test1.json").write_text("{}")
  (tests_dir / "test2.json").write_text("{}")

  response = test_client.get("/dev/apps/test_app/tests")
  assert response.status_code == 200
  assert response.json() == ["test1.json", "test2.json"]


def test_delete_test(test_client, tmp_path):
  agent_dir = tmp_path / "test_app"
  tests_dir = agent_dir / "tests"
  tests_dir.mkdir(parents=True)
  test_file = tests_dir / "test1.json"
  test_file.write_text("{}")

  response = test_client.delete("/dev/apps/test_app/tests/test1.json")
  assert response.status_code == 200
  assert response.json() == {"status": "success"}
  assert not test_file.exists()


def test_get_test_content(test_client, tmp_path):
  agent_dir = tmp_path / "test_app"
  tests_dir = agent_dir / "tests"
  tests_dir.mkdir(parents=True)
  test_file = tests_dir / "test_get.json"
  test_file.write_text('{"foo": "bar"}')

  response = test_client.get("/dev/apps/test_app/tests/test_get.json")
  assert response.status_code == 200
  assert response.json() == {"foo": "bar"}


def test_get_test_content_not_found(test_client):
  response = test_client.get("/dev/apps/test_app/tests/non_existent.json")
  assert response.status_code == 404


def test_rebuild_tests(test_client):
  with patch("google.adk.cli.dev_server.asyncio.to_thread") as mock_to_thread:
    mock_to_thread.return_value = None
    response = test_client.post("/dev/apps/test_app/tests/rebuild", json={})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    mock_to_thread.assert_called_once()


def test_rebuild_single_test(test_client):
  with patch("google.adk.cli.dev_server.asyncio.to_thread") as mock_to_thread:
    mock_to_thread.return_value = None
    response = test_client.post(
        "/dev/apps/test_app/tests/rebuild?test_name=my_test.json", json={}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    mock_to_thread.assert_called_once()
    args, kwargs = mock_to_thread.call_args
    assert args[1].endswith("tests/my_test.json")


def test_run_tests(test_client):
  from unittest.mock import AsyncMock
  from unittest.mock import MagicMock
  from unittest.mock import patch

  mock_process = MagicMock()
  mock_process.stdout.readline = AsyncMock(
      side_effect=[b"line1\n", b"line2\n", b""]
  )
  mock_process.wait = AsyncMock(return_value=0)

  with patch(
      "google.adk.cli.dev_server.asyncio.create_subprocess_exec",
      new_callable=AsyncMock,
  ) as mock_create_subprocess:
    mock_create_subprocess.return_value = mock_process

    response = test_client.post("/dev/apps/test_app/tests/run", json={})
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    # Read stream
    content = response.content
    assert b"line1\n" in content
    assert b"line2\n" in content
