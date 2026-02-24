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

import asyncio
import types
from typing import Any

from fastapi.testclient import TestClient
from google.adk.agents.base_agent import BaseAgent
from google.adk.cli.adk_web_server import AdkWebServer
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
import pytest


class _DummyAgent(BaseAgent):

  def __init__(self) -> None:
    super().__init__(name="dummy_agent")
    self.sub_agents = []


class _DummyAgentLoader:

  def load_agent(self, app_name: str) -> BaseAgent:
    return _DummyAgent()

  def list_agents(self) -> list[str]:
    return ["test_app"]

  def list_agents_detailed(self) -> list[dict[str, Any]]:
    return []


class _CapturingRunner:

  def __init__(self) -> None:
    self.captured_run_config = None

  async def run_live(
      self,
      *,
      session,
      live_request_queue,
      run_config=None,
      **unused_kwargs,
  ):
    self.captured_run_config = run_config
    yield Event(author="runner")


def test_run_live_applies_run_config_query_options():
  session_service = InMemorySessionService()
  asyncio.run(
      session_service.create_session(
          app_name="test_app",
          user_id="user",
          session_id="session",
          state={},
      )
  )

  runner = _CapturingRunner()
  adk_web_server = AdkWebServer(
      agent_loader=_DummyAgentLoader(),
      session_service=session_service,
      memory_service=types.SimpleNamespace(),
      artifact_service=types.SimpleNamespace(),
      credential_service=types.SimpleNamespace(),
      eval_sets_manager=types.SimpleNamespace(),
      eval_set_results_manager=types.SimpleNamespace(),
      agents_dir=".",
  )

  async def _get_runner_async(_self, _app_name: str):
    return runner

  adk_web_server.get_runner_async = _get_runner_async.__get__(adk_web_server)  # pytype: disable=attribute-error

  fast_api_app = adk_web_server.get_fast_api_app(
      setup_observer=lambda _observer, _server: None,
      tear_down_observer=lambda _observer, _server: None,
  )

  client = TestClient(fast_api_app)
  url = (
      "/run_live"
      "?app_name=test_app"
      "&user_id=user"
      "&session_id=session"
      "&modalities=TEXT"
      "&modalities=AUDIO"
      "&proactive_audio=true"
      "&enable_affective_dialog=true"
      "&enable_session_resumption=true"
  )

  with client.websocket_connect(url) as ws:
    _ = ws.receive_text()

  run_config = runner.captured_run_config
  assert run_config is not None
  assert run_config.response_modalities == ["TEXT", "AUDIO"]
  assert run_config.enable_affective_dialog is True
  assert run_config.proactivity is not None
  assert run_config.proactivity.proactive_audio is True
  assert run_config.session_resumption is not None
  assert run_config.session_resumption.transparent is True


@pytest.mark.parametrize(
    (
        "query,expected_enable_affective,expected_proactive_audio,"
        "expected_session_resumption_transparent"
    ),
    [
        ("", None, None, None),
        ("&proactive_audio=true", None, True, None),
        ("&proactive_audio=false", None, False, None),
        ("&enable_affective_dialog=true", True, None, None),
        ("&enable_affective_dialog=false", False, None, None),
        ("&enable_session_resumption=true", None, None, True),
        ("&enable_session_resumption=false", None, None, False),
    ],
)
def test_run_live_defaults_and_individual_options(
    query: str,
    expected_enable_affective: bool | None,
    expected_proactive_audio: bool | None,
    expected_session_resumption_transparent: bool | None,
):
  session_service = InMemorySessionService()
  asyncio.run(
      session_service.create_session(
          app_name="test_app",
          user_id="user",
          session_id="session",
          state={},
      )
  )

  runner = _CapturingRunner()
  adk_web_server = AdkWebServer(
      agent_loader=_DummyAgentLoader(),
      session_service=session_service,
      memory_service=types.SimpleNamespace(),
      artifact_service=types.SimpleNamespace(),
      credential_service=types.SimpleNamespace(),
      eval_sets_manager=types.SimpleNamespace(),
      eval_set_results_manager=types.SimpleNamespace(),
      agents_dir=".",
  )

  async def _get_runner_async(_self, _app_name: str):
    return runner

  adk_web_server.get_runner_async = _get_runner_async.__get__(adk_web_server)  # pytype: disable=attribute-error

  fast_api_app = adk_web_server.get_fast_api_app(
      setup_observer=lambda _observer, _server: None,
      tear_down_observer=lambda _observer, _server: None,
  )

  client = TestClient(fast_api_app)
  url = (
      "/run_live"
      "?app_name=test_app"
      "&user_id=user"
      "&session_id=session"
      "&modalities=AUDIO"
      f"{query}"
  )

  with client.websocket_connect(url) as ws:
    _ = ws.receive_text()

  run_config = runner.captured_run_config
  assert run_config is not None
  assert run_config.enable_affective_dialog == expected_enable_affective

  if expected_proactive_audio is None:
    assert run_config.proactivity is None
  else:
    assert run_config.proactivity is not None
    assert run_config.proactivity.proactive_audio is expected_proactive_audio

  if expected_session_resumption_transparent is None:
    assert run_config.session_resumption is None
  else:
    assert run_config.session_resumption is not None
    assert (
        run_config.session_resumption.transparent
        is expected_session_resumption_transparent
    )


def _make_server(**kwargs) -> AdkWebServer:
  defaults = dict(
      agent_loader=_DummyAgentLoader(),
      session_service=InMemorySessionService(),
      memory_service=types.SimpleNamespace(),
      artifact_service=types.SimpleNamespace(),
      credential_service=types.SimpleNamespace(),
      eval_sets_manager=types.SimpleNamespace(),
      eval_set_results_manager=types.SimpleNamespace(),
      agents_dir=".",
  )
  defaults.update(kwargs)
  return AdkWebServer(**defaults)


class TestBuildRuntimeConfig:

  def test_default_no_prefix(self):
    server = _make_server()
    config = server._build_runtime_config()
    assert config == {"backendUrl": ""}

  def test_with_url_prefix(self):
    server = _make_server(url_prefix="/my-prefix")
    config = server._build_runtime_config()
    assert config == {"backendUrl": "/my-prefix"}

  def test_with_logo(self):
    server = _make_server(
        logo_text="My App",
        logo_image_url="https://example.com/logo.png",
    )
    config = server._build_runtime_config()
    assert config == {
        "backendUrl": "",
        "logo": {
            "text": "My App",
            "imageUrl": "https://example.com/logo.png",
        },
    }

  def test_logo_text_only_raises(self):
    server = _make_server(logo_text="My App")
    with pytest.raises(ValueError, match="Both --logo-text and --logo-image-url"):
      server._build_runtime_config()

  def test_logo_image_url_only_raises(self):
    server = _make_server(logo_image_url="https://example.com/logo.png")
    with pytest.raises(ValueError, match="Both --logo-text and --logo-image-url"):
      server._build_runtime_config()


class TestRuntimeConfigEndpoint:

  def test_get_runtime_config_json(self, tmp_path):
    server = _make_server(url_prefix="/test-prefix")
    app = server.get_fast_api_app(
        setup_observer=lambda _observer, _server: None,
        tear_down_observer=lambda _observer, _server: None,
        web_assets_dir=str(tmp_path),
    )
    client = TestClient(app)
    resp = client.get("/dev-ui/assets/config/runtime-config.json")
    assert resp.status_code == 200
    assert resp.json() == {"backendUrl": "/test-prefix"}
