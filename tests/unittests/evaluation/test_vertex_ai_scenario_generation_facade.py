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

"""Tests for the Vertex AI Scenario Generation Facade."""

from __future__ import annotations

import os

from google.adk.agents.base_agent import BaseAgent
from google.adk.dependencies.vertexai import vertexai
from google.adk.evaluation._vertex_ai_scenario_generation_facade import ScenarioGenerator
from google.adk.evaluation.conversation_scenarios import ConversationGenerationConfig
import pytest

vertexai_types = vertexai.types


class TestScenarioGenerator:
  """Unit tests for ScenarioGenerator."""

  def test_constructor_with_api_key(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    ScenarioGenerator()

    mock_client_cls.assert_called_once_with(api_key="test_api_key")

  def test_constructor_with_project_and_location(self, mocker):
    """Test constructor with project and location in env."""
    mocker.patch.dict(
        os.environ,
        {
            "GOOGLE_CLOUD_PROJECT": "test_project",
            "GOOGLE_CLOUD_LOCATION": "test_location",
        },
        clear=True,
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    ScenarioGenerator()

    mock_client_cls.assert_called_once_with(
        project="test_project", location="test_location"
    )

  def test_constructor_with_project_only_raises_error(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_CLOUD_PROJECT": "test_project"}, clear=True
    )
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")

    with pytest.raises(ValueError, match="Missing location."):
      ScenarioGenerator()

  def test_constructor_with_location_only_raises_error(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_CLOUD_LOCATION": "test_location"}, clear=True
    )
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")

    with pytest.raises(ValueError, match="Missing project id."):
      ScenarioGenerator()

  def test_constructor_with_no_env_vars_raises_error(self, mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")

    with pytest.raises(
        ValueError,
        match=(
            "Either API Key or Google cloud Project id and location should be"
            " specified."
        ),
    ):
      ScenarioGenerator()

  def test_generate_scenarios(self, mocker):
    """Test scenario generation with mocked components."""
    mocker.patch.dict(
        os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    mock_client = mock_client_cls.return_value

    # I need to mock AgentInfo.load_from_agent(agent=agent)
    mock_agent_info_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.types.evals.AgentInfo"
    )
    mock_agent_info_cls.load_from_agent.return_value = "mock_agent_info"

    mock_generate = mocker.patch.object(
        mock_client.evals, "generate_conversation_scenarios"
    )

    mock_eval_cases = [
        mocker.Mock(
            user_scenario=mocker.Mock(
                starting_prompt="Hello", conversation_plan="Say hello"
            )
        ),
        mocker.Mock(user_scenario=None),  # testing handling of None
        mocker.Mock(
            user_scenario=mocker.Mock(
                starting_prompt="Bye", conversation_plan="Say bye"
            )
        ),
    ]
    mock_generate.return_value = mocker.Mock(eval_cases=mock_eval_cases)

    generator = ScenarioGenerator()

    mock_agent = mocker.Mock(spec=BaseAgent)
    config = ConversationGenerationConfig(
        count=2,
        generation_instruction="Test generation",
        model_name="gemini-2.5-flash",
    )

    scenarios = generator.generate_scenarios(mock_agent, config)

    assert len(scenarios) == 2
    assert scenarios[0].starting_prompt == "Hello"
    assert scenarios[0].conversation_plan == "Say hello"
    assert scenarios[1].starting_prompt == "Bye"
    assert scenarios[1].conversation_plan == "Say bye"

    mock_agent_info_cls.load_from_agent.assert_called_once_with(
        agent=mock_agent
    )

    mock_generate.assert_called_once()
    _, kwargs = mock_generate.call_args
    assert kwargs["agent_info"] == "mock_agent_info"
    passed_config = kwargs["config"]
    assert passed_config.count == 2
    assert passed_config.generation_instruction == "Test generation"
