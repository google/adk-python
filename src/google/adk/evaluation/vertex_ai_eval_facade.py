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

import abc
import logging
import math
import os
from typing import Optional
from typing import Union

from google.genai import types as genai_types
import pandas as pd
from typing_extensions import override

from ..dependencies.vertexai import vertexai
from .app_details import AgentDetails
from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_case import InvocationEvent
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult

logger = logging.getLogger("google_adk." + __name__)

_ERROR_MESSAGE_SUFFIX = """
You should specify both project id and location. This metric uses Vertex Gen AI
Eval SDK, and it requires google cloud credentials.

If using an .env file add the values there, or explicitly set in the code using
the template below:

os.environ['GOOGLE_CLOUD_LOCATION'] = <LOCATION>
os.environ['GOOGLE_CLOUD_PROJECT'] = <PROJECT ID>
"""


class _VertexAiEvalFacade(Evaluator):
  """Simple facade for Vertex Gen AI Eval SDK.

  Vertex Gen AI Eval SDK exposes quite a few metrics that are valuable for
  agentic evals. This class helps us to access those metrics.

  Using this class requires a GCP project. Please set GOOGLE_CLOUD_PROJECT and
  GOOGLE_CLOUD_LOCATION in your .env file.
  """

  def __init__(
      self,
      threshold: float,
      metric_name: Union[
          vertexai.types.PrebuiltMetric, vertexai.types.RubricMetric
      ],
      expected_invocations_required=False,
  ):
    self._threshold = threshold
    self._metric_name = metric_name
    self._expected_invocations_required = expected_invocations_required

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", None)
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", None)
    api_key = os.environ.get("GOOGLE_API_KEY", None)

    if api_key:
      self._client = vertexai.Client(api_key=api_key)
    elif project_id or location:
      if not project_id:
        raise ValueError("Missing project id." + _ERROR_MESSAGE_SUFFIX)
      if not location:
        raise ValueError("Missing location." + _ERROR_MESSAGE_SUFFIX)
      self._client = vertexai.Client(project=project_id, location=location)
    else:
      raise ValueError(
          "Either API Key or Google cloud Project id and location should be"
          " specified."
      )

  @abc.abstractmethod
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    """Returns EvaluationResult after performing evaluations using actual and expected invocations.

    Args:
      actual_invocations: These are the invocations that are obtained from the
        agent under test.
      expected_invocations: An optional list of invocations, if specified,
        usually act as a benchmark/golden response. If these are specified
        usually the expectation is that the length of this list and actual
        invocation is the same.
      conversation_scenario: An optional conversation scenario for multi-turn
        conversations.
    """

  def _get_text(self, content: Optional[genai_types.Content]) -> str:
    if content and content.parts:
      return "\n".join([p.text for p in content.parts if p.text])

    return ""

  def _get_score(self, eval_result) -> Optional[float]:
    if (
        eval_result
        and eval_result.summary_metrics
        and isinstance(eval_result.summary_metrics[0].mean_score, float)
        and not math.isnan(eval_result.summary_metrics[0].mean_score)
    ):
      return eval_result.summary_metrics[0].mean_score

    return None

  def _get_eval_status(self, score: Optional[float]):
    if score is not None:
      return (
          EvalStatus.PASSED if score >= self._threshold else EvalStatus.FAILED
      )

    return EvalStatus.NOT_EVALUATED

  def _perform_eval(self, dataset, metrics):
    """This method hides away the call to external service.

    Primarily helps with unit testing.
    """
    return self._client.evals.evaluate(
        dataset=dataset,
        metrics=metrics,
    )


class _SingleTurnVertexAiEvalFacade(_VertexAiEvalFacade):
  """A facade for single turn metrics exposed in Vertex Gen AI Eval SDK."""

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    if self._expected_invocations_required and expected_invocations is None:
      raise ValueError("expected_invocations is needed by this metric.")
    del conversation_scenario  # not supported for per-invocation evaluation.

    # If expected_invocation are not required by the metric and if they are not
    # supplied, we provide a list of None.
    expected_invocations = (
        [None] * len(actual_invocations)
        if expected_invocations is None
        else expected_invocations
    )

    total_score = 0.0
    num_invocations = 0
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations):
      prompt = self._get_text(actual.user_content)
      reference = self._get_text(expected.final_response) if expected else None
      response = self._get_text(actual.final_response)
      eval_case = {
          "prompt": prompt,
          "reference": reference,
          "response": response,
      }

      dataset = vertexai.types.EvaluationDataset(
          eval_dataset_df=pd.DataFrame([eval_case])
      )
      eval_case_result = self._perform_eval(
          dataset=dataset, metrics=[self._metric_name]
      )
      score = self._get_score(eval_case_result)
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=score,
              eval_status=self._get_eval_status(score),
          )
      )

      if score is not None:
        total_score += score
        num_invocations += 1

    if per_invocation_results:
      overall_score = (
          total_score / num_invocations if num_invocations > 0 else None
      )
      return EvaluationResult(
          overall_score=overall_score,
          overall_eval_status=self._get_eval_status(overall_score),
          per_invocation_results=per_invocation_results,
      )

    return EvaluationResult()


class _MultiTurnVertexiAiEvalFacade(_VertexAiEvalFacade):
  """A facade for multi turn metrics exposed in Vertex Gen AI Eval SDK."""

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    del conversation_scenario

    per_invocation_results = []
    # If expected_invocation are not required by the metric and if they are not
    # supplied, we provide a list of None.
    expected_invocations = (
        [None] * len(actual_invocations)
        if expected_invocations is None
        else expected_invocations
    )

    # We mark all the n-1 turns as NOT-EVALUATED for these metrics.
    for actual, expected in zip(
        actual_invocations[:-1], expected_invocations[:-1]
    ):
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=None,
              eval_status=self._get_eval_status(None),
          )
      )

    # Only evaluate the last turn and take into account all the previous turns.
    eval_case = vertexai.types.EvalCase(
        agent_data=_MultiTurnVertexiAiEvalFacade._get_agent_data(
            actual_invocations
        )
    )
    dataset = vertexai.types.EvaluationDataset(eval_cases=[eval_case])

    eval_case_result = self._perform_eval(
        dataset=dataset, metrics=[self._metric_name]
    )

    score = self._get_score(eval_case_result)
    per_invocation_results.append(
        PerInvocationResult(
            actual_invocation=actual_invocations[-1],
            expected_invocation=expected_invocations[-1],
            score=score,
            eval_status=self._get_eval_status(score),
        )
    )

    if score is not None:
      return EvaluationResult(
          overall_score=score,
          overall_eval_status=self._get_eval_status(score),
          per_invocation_results=per_invocation_results,
      )

    return EvaluationResult()

  @staticmethod
  def _get_agent_data(
      actual_invocations: list[Invocation],
  ) -> vertexai.types.evals.AgentData:
    return vertexai.types.evals.AgentData(
        agents=_MultiTurnVertexiAiEvalFacade._get_agent_details(
            actual_invocations
        ),
        turns=_MultiTurnVertexiAiEvalFacade._get_turns(actual_invocations),
    )

  @staticmethod
  def _get_turns(
      actual_invocations: list[Invocation],
  ) -> list[vertexai.types.evals.ConversationTurn]:
    return [
        _MultiTurnVertexiAiEvalFacade._map_invocation_turn(index, invocation)
        for index, invocation in enumerate(actual_invocations)
    ]

  @staticmethod
  def _map_invocation_turn(
      turn_index: int,
      invocation: Invocation,
  ) -> vertexai.types.evals.ConversationTurn:
    agent_events = []
    agent_events.append(
        vertexai.types.evals.AgentEvent(
            author="user", content=invocation.user_content
        )
    )

    for invocation_event in invocation.intermediate_data.invocation_events:
      agent_events.append(
          _MultiTurnVertexiAiEvalFacade._map_inovcation_event_to_agent_event(
              invocation_event
          )
      )

    agent_events.append(
        vertexai.types.evals.AgentEvent(
            author="agent", content=invocation.final_response
        )
    )

    return vertexai.types.evals.ConversationTurn(
        turn_index=turn_index,
        events=agent_events,
        turn_id=invocation.invocation_id,
    )

  @staticmethod
  def _map_inovcation_event_to_agent_event(
      invocation_event: InvocationEvent,
  ) -> vertexai.types.evals.AgentEvent:
    return vertexai.types.evals.AgentEvent(
        author=invocation_event.author, content=invocation_event.content
    )

  @staticmethod
  def _get_agent_details(
      actual_invocations: list[Invocation],
  ) -> dict[str, vertexai.types.evals.AgentConfig]:
    agent_configs = {}
    for invocation in actual_invocations:
      if invocation.app_details and invocation.app_details.agent_details:
        for (
            agent_name,
            agent_details,
        ) in invocation.app_details.agent_details.items():
          if agent_name not in agent_configs:
            agent_configs[agent_name] = (
                _MultiTurnVertexiAiEvalFacade._map_agent_details_to_agent_config(
                    agent_details
                )
            )

    return agent_configs

  @staticmethod
  def _map_agent_details_to_agent_config(
      agent_details: AgentDetails,
  ) -> vertexai.types.evals.AgentConfig:
    return vertexai.types.evals.AgentConfig(
        agent_id=agent_details.name,
        instruction=agent_details.instructions,
        tools=agent_details.tool_declarations,
    )
