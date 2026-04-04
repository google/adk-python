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

"""Tests for the Response Evaluator."""
import math
import os
import random

from google.adk.dependencies.vertexai import vertexai
from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.vertex_ai_eval_facade import _MultiTurnVertexiAiEvalFacade
from google.adk.evaluation.vertex_ai_eval_facade import _SingleTurnVertexAiEvalFacade
from google.adk.evaluation.vertex_ai_eval_facade import _VertexAiEvalFacade
from google.genai import types as genai_types
import pandas as pd
import pytest

vertexai_types = vertexai.types


class TestSingleTurnVertexAiEvalFacade:
  """A class to help organize "patch" that are applicable to all tests."""

  def test_evaluate_invocations_metric_passed(self, mocker):
    """Test evaluate_invocations function for a metric."""
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")
    mock_perform_eval = mocker.patch(
        "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
    )
    actual_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[
                    genai_types.Part(text="This is a test candidate response.")
                ]
            ),
        )
    ]
    expected_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="This is a test reference.")]
            ),
        )
    ]
    evaluator = _SingleTurnVertexAiEvalFacade(
        threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.return_value = vertexai_types.EvaluationResult(
        summary_metrics=[vertexai_types.AggregatedMetricResult(mean_score=0.9)],
        eval_case_results=[],
    )

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations, expected_invocations
    )

    assert evaluation_result.overall_score == 0.9
    assert evaluation_result.overall_eval_status == EvalStatus.PASSED
    mock_perform_eval.assert_called_once()
    _, mock_kwargs = mock_perform_eval.call_args
    # Compare the names of the metrics.
    assert [m.name for m in mock_kwargs["metrics"]] == [
        vertexai_types.PrebuiltMetric.COHERENCE.name
    ]

  def test_evaluate_invocations_metric_failed(self, mocker):
    """Test evaluate_invocations function for a metric."""
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")
    mock_perform_eval = mocker.patch(
        "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
    )
    actual_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[
                    genai_types.Part(text="This is a test candidate response.")
                ]
            ),
        )
    ]
    expected_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="This is a test reference.")]
            ),
        )
    ]
    evaluator = _SingleTurnVertexAiEvalFacade(
        threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.return_value = vertexai_types.EvaluationResult(
        summary_metrics=[vertexai_types.AggregatedMetricResult(mean_score=0.7)],
        eval_case_results=[],
    )

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations, expected_invocations
    )

    assert evaluation_result.overall_score == 0.7
    assert evaluation_result.overall_eval_status == EvalStatus.FAILED
    mock_perform_eval.assert_called_once()
    _, mock_kwargs = mock_perform_eval.call_args
    # Compare the names of the metrics.
    assert [m.name for m in mock_kwargs["metrics"]] == [
        vertexai_types.PrebuiltMetric.COHERENCE.name
    ]

  @pytest.mark.parametrize(
      "summary_metric_with_no_score",
      [
          ([]),
          ([vertexai_types.AggregatedMetricResult(mean_score=float("nan"))]),
          ([vertexai_types.AggregatedMetricResult(mean_score=None)]),
          ([vertexai_types.AggregatedMetricResult(mean_score=math.nan)]),
      ],
  )
  def test_evaluate_invocations_metric_no_score(
      self, mocker, summary_metric_with_no_score
  ):
    """Test evaluate_invocations function for a metric."""
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")
    mock_perform_eval = mocker.patch(
        "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
    )
    actual_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[
                    genai_types.Part(text="This is a test candidate response.")
                ]
            ),
        )
    ]
    expected_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="This is a test reference.")]
            ),
        )
    ]
    evaluator = _SingleTurnVertexAiEvalFacade(
        threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.return_value = vertexai_types.EvaluationResult(
        summary_metrics=summary_metric_with_no_score,
        eval_case_results=[],
    )

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations, expected_invocations
    )

    assert evaluation_result.overall_score is None
    assert evaluation_result.overall_eval_status == EvalStatus.NOT_EVALUATED
    mock_perform_eval.assert_called_once()
    _, mock_kwargs = mock_perform_eval.call_args
    # Compare the names of the metrics.
    assert [m.name for m in mock_kwargs["metrics"]] == [
        vertexai_types.PrebuiltMetric.COHERENCE.name
    ]

  def test_evaluate_invocations_metric_multiple_invocations(self, mocker):
    """Test evaluate_invocations function for a metric with multiple invocations."""
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")
    mock_perform_eval = mocker.patch(
        "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
    )
    num_invocations = 6
    actual_invocations = []
    expected_invocations = []
    mock_eval_results = []
    random.seed(61553)
    scores = [random.random() for _ in range(num_invocations)]

    for i in range(num_invocations):
      actual_invocations.append(
          Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text=f"Query {i+1}")]
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text=f"Response {i+1}")]
              ),
          )
      )
      expected_invocations.append(
          Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text=f"Query {i+1}")]
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text=f"Reference {i+1}")]
              ),
          )
      )
      mock_eval_results.append(
          vertexai_types.EvaluationResult(
              summary_metrics=[
                  vertexai_types.AggregatedMetricResult(mean_score=scores[i])
              ],
              eval_case_results=[],
          )
      )

    evaluator = _SingleTurnVertexAiEvalFacade(
        threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.side_effect = mock_eval_results

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations, expected_invocations
    )

    assert evaluation_result.overall_score == pytest.approx(
        sum(scores) / num_invocations
    )
    assert evaluation_result.overall_eval_status == EvalStatus.FAILED
    assert mock_perform_eval.call_count == num_invocations


class TestVertexAiEvalFacade:
  """A class to help organize "patch" that are applicable to all tests."""

  def test_constructor_with_api_key(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    _SingleTurnVertexAiEvalFacade(
        threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
    )

    mock_client_cls.assert_called_once_with(api_key="test_api_key")

  def test_constructor_with_project_and_location(self, mocker):

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
    _SingleTurnVertexAiEvalFacade(
        threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
    )

    mock_client_cls.assert_called_once_with(
        project="test_project", location="test_location"
    )

  def test_constructor_with_project_only_raises_error(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_CLOUD_PROJECT": "test_project"}, clear=True
    )
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")

    with pytest.raises(ValueError, match="Missing location."):
      _SingleTurnVertexAiEvalFacade(
          threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
      )

  def test_constructor_with_location_only_raises_error(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_CLOUD_LOCATION": "test_location"}, clear=True
    )
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")

    with pytest.raises(ValueError, match="Missing project id."):
      _SingleTurnVertexAiEvalFacade(
          threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
      )

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
      _SingleTurnVertexAiEvalFacade(
          threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
      )


class TestMultiTurnVertexAiEvalFacade:
  """Tests for _MultiTurnVertexiAiEvalFacade."""

  def test_map_agent_details_to_agent_config(self):
    tool_declarations = [
        genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name="tool_1",
                    description="this is tool 1",
                )
            ]
        )
    ]
    agent_details = AgentDetails(
        name="test_agent",
        instructions="test_instructions",
        tool_declarations=tool_declarations,
    )
    agent_config = (
        _MultiTurnVertexiAiEvalFacade._map_agent_details_to_agent_config(
            agent_details
        )
    )
    assert agent_config.agent_id == "test_agent"
    assert agent_config.instruction == "test_instructions"
    assert agent_config.tools == tool_declarations

  def test_get_agent_details(self):
    invocations = [
        Invocation(
            user_content=genai_types.Content(),
            app_details=AppDetails(
                agent_details={
                    "agent1": AgentDetails(
                        name="agent1", instructions="instructions1"
                    ),
                    "agent2": AgentDetails(
                        name="agent2", instructions="instructions2"
                    ),
                }
            ),
        ),
        Invocation(
            user_content=genai_types.Content(),
            app_details=AppDetails(
                agent_details={
                    "agent1": AgentDetails(
                        name="agent1", instructions="instructions1"
                    ),
                    "agent3": AgentDetails(
                        name="agent3", instructions="instructions3"
                    ),
                }
            ),
        ),
    ]
    agent_configs = _MultiTurnVertexiAiEvalFacade._get_agent_details(
        invocations
    )
    assert len(agent_configs) == 3
    assert "agent1" in agent_configs
    assert "agent2" in agent_configs
    assert "agent3" in agent_configs
    assert agent_configs["agent1"].instruction == "instructions1"
    assert agent_configs["agent2"].instruction == "instructions2"
    assert agent_configs["agent3"].instruction == "instructions3"

  def test_map_invocation_event_to_agent_event(self):
    invocation_event = InvocationEvent(
        author="test_author",
        content=genai_types.Content(
            parts=[genai_types.Part(text="test_content")]
        ),
    )
    agent_event = (
        _MultiTurnVertexiAiEvalFacade._map_inovcation_event_to_agent_event(
            invocation_event
        )
    )
    assert agent_event.author == "test_author"
    assert agent_event.content.parts[0].text == "test_content"

  def test_map_invocation_turn(self):
    invocation = Invocation(
        invocation_id="inv1",
        user_content=genai_types.Content(
            parts=[genai_types.Part(text="user query")]
        ),
        intermediate_data=InvocationEvents(
            invocation_events=[
                InvocationEvent(
                    author="agent1",
                    content=genai_types.Content(
                        parts=[genai_types.Part(text="intermediate content")]
                    ),
                )
            ]
        ),
        final_response=genai_types.Content(
            parts=[genai_types.Part(text="final response")]
        ),
    )
    conversation_turn = _MultiTurnVertexiAiEvalFacade._map_invocation_turn(
        0, invocation
    )
    assert conversation_turn.turn_index == 0
    assert conversation_turn.turn_id == "inv1"
    assert len(conversation_turn.events) == 3
    assert conversation_turn.events[0].author == "user"
    assert conversation_turn.events[0].content.parts[0].text == "user query"
    assert conversation_turn.events[1].author == "agent1"
    assert (
        conversation_turn.events[1].content.parts[0].text
        == "intermediate content"
    )
    assert conversation_turn.events[2].author == "agent"
    assert conversation_turn.events[2].content.parts[0].text == "final response"

  def test_get_turns(self):
    invocations = [
        Invocation(
            invocation_id="inv1",
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="q1")]
            ),
            intermediate_data=InvocationEvents(invocation_events=[]),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="r1")]
            ),
        ),
        Invocation(
            invocation_id="inv2",
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="q2")]
            ),
            intermediate_data=InvocationEvents(invocation_events=[]),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="r2")]
            ),
        ),
    ]
    turns = _MultiTurnVertexiAiEvalFacade._get_turns(invocations)
    assert len(turns) == 2
    assert turns[0].turn_id == "inv1"
    assert turns[1].turn_id == "inv2"

  def test_get_agent_data(self):
    invocations = [
        Invocation(
            invocation_id="inv1",
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="q1")]
            ),
            intermediate_data=InvocationEvents(invocation_events=[]),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="r1")]
            ),
            app_details=AppDetails(
                agent_details={
                    "agent1": AgentDetails(
                        name="agent1", instructions="instructions1"
                    )
                }
            ),
        )
    ]
    agent_data = _MultiTurnVertexiAiEvalFacade._get_agent_data(invocations)
    assert "agent1" in agent_data.agents
    assert len(agent_data.turns) == 1

  def test_evaluate_invocations_multi_turn_metric_passed(self, mocker):
    """Test evaluate_invocations function for a multi-turn metric."""
    mock_perform_eval = mocker.patch(
        "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
    )
    actual_invocations = [
        Invocation(
            invocation_id="inv1",
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="q1")]
            ),
            intermediate_data=InvocationEvents(invocation_events=[]),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="r1")]
            ),
            app_details=AppDetails(
                agent_details={
                    "agent1": AgentDetails(
                        name="agent1", instructions="instructions1"
                    )
                }
            ),
        ),
        Invocation(
            invocation_id="inv2",
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="q2")]
            ),
            intermediate_data=InvocationEvents(
                invocation_events=[
                    InvocationEvent(
                        author="agent1",
                        content=genai_types.Content(
                            parts=[genai_types.Part(text="intermediate")]
                        ),
                    )
                ]
            ),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="r2")]
            ),
            app_details=AppDetails(
                agent_details={
                    "agent1": AgentDetails(
                        name="agent1", instructions="instructions1"
                    )
                }
            ),
        ),
    ]
    evaluator = _MultiTurnVertexiAiEvalFacade(
        threshold=0.8,
        metric_name=vertexai_types.PrebuiltMetric.CONVERSATIONAL_COHERENCE,
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.return_value = vertexai_types.EvaluationResult(
        summary_metrics=[vertexai_types.AggregatedMetricResult(mean_score=0.9)],
        eval_case_results=[],
    )

    evaluation_result = evaluator.evaluate_invocations(actual_invocations)

    assert evaluation_result.overall_score == 0.9
    assert evaluation_result.overall_eval_status == EvalStatus.PASSED
    assert len(evaluation_result.per_invocation_results) == 2
    assert (
        evaluation_result.per_invocation_results[0].eval_status
        == EvalStatus.NOT_EVALUATED
    )
    assert (
        evaluation_result.per_invocation_results[1].eval_status
        == EvalStatus.PASSED
    )
    mock_perform_eval.assert_called_once()
    _, mock_kwargs = mock_perform_eval.call_args
    assert [m.name for m in mock_kwargs["metrics"]] == [
        vertexai_types.PrebuiltMetric.CONVERSATIONAL_COHERENCE.name
    ]
    dataset = mock_kwargs["dataset"]
    assert len(dataset.eval_cases) == 1
    agent_data = dataset.eval_cases[0].agent_data
    assert "agent1" in agent_data.agents
    assert len(agent_data.turns) == 2
    assert agent_data.turns[0].turn_id == "inv1"
    assert agent_data.turns[1].turn_id == "inv2"
    assert len(agent_data.turns[1].events) == 3  # user, intermediate, agent
