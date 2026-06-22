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

from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.base_agent import BaseAgent
from google.adk.cli.fast_api import get_fast_api_app


def test_a2a_infers_agent_card_without_agent_json(tmp_path, monkeypatch):
  """A2A setup builds an agent card from agent.py when agent.json is absent."""

  class _TestAgent(BaseAgent):
    pass

  agent_dir = tmp_path / "test_a2a_agent"
  agent_dir.mkdir()
  (agent_dir / "agent.py").write_text("root_agent = None\n")
  agent = _TestAgent(
      name="test_a2a_agent",
      description="Generated card from ADK agent",
  )
  agent_loader = MagicMock()
  agent_loader.load_agent.return_value = agent

  with (
      patch(
          "google.adk.cli.fast_api.create_session_service_from_options",
          return_value=MagicMock(),
      ),
      patch(
          "google.adk.cli.fast_api.create_artifact_service_from_options",
          return_value=MagicMock(),
      ),
      patch(
          "google.adk.cli.fast_api.create_memory_service_from_options",
          return_value=MagicMock(),
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetsManager",
          return_value=MagicMock(),
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetResultsManager",
          return_value=MagicMock(),
      ),
      patch(
          "google.adk.cli.fast_api._create_task_store_from_options",
          return_value=MagicMock(),
      ),
      patch(
          "google.adk.a2a.executor.a2a_agent_executor.A2aAgentExecutor",
          return_value=MagicMock(),
      ),
      patch(
          "a2a.server.request_handlers.DefaultRequestHandler",
          return_value=MagicMock(),
      ),
      patch("a2a.server.apps.A2AStarletteApplication") as mock_a2a_app,
  ):
    mock_a2a_app.return_value.routes.return_value = []
    monkeypatch.chdir(tmp_path)

    get_fast_api_app(
        agents_dir=".",
        agent_loader=agent_loader,
        web=False,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        a2a=True,
        host="127.0.0.1",
        port=8000,
    )

  agent_loader.load_agent.assert_called_once_with("test_a2a_agent")
  mock_a2a_app.assert_called_once()
  agent_card = mock_a2a_app.call_args.kwargs["agent_card"]
  assert agent_card.name == "test_a2a_agent"
  assert agent_card.description == "Generated card from ADK agent"
  assert agent_card.url == "http://127.0.0.1:8000/a2a/test_a2a_agent"
