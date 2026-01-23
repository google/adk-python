# Copyright 2025 Google LLC
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

import importlib
import inspect
from typing import Callable
from typing import Optional

from typing_extensions import override

from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .eval_metrics import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator


def _get_metric_function(
    custom_function_path: str,
) -> Callable[..., EvaluationResult]:
  """Returns the custom metric function from the given path."""
  try:
    module_name, function_name = custom_function_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    metric_function = getattr(module, function_name)
    return metric_function
  except (ImportError, AttributeError, ValueError) as e:
    raise ImportError(
        f"Could not import custom metric function from {custom_function_path}"
    ) from e


def _get_eval_status(score: Optional[float], threshold: float) -> EvalStatus:
  if score is None:
    return EvalStatus.NOT_EVALUATED
  return EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED


class _CustomMetricEvaluator(Evaluator):
  """Evaluator for custom metrics."""

  def __init__(self, eval_metric: EvalMetric, custom_function_path: str):
    self._eval_metric = eval_metric
    self._metric_function = _get_metric_function(custom_function_path)

  @override
  async def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]],
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    if inspect.iscoroutinefunction(self._metric_function):
      eval_result = await self._metric_function(
          actual_invocations, expected_invocations, conversation_scenario
      )
    else:
      eval_result = self._metric_function(
          actual_invocations, expected_invocations, conversation_scenario
      )

    eval_result.overall_eval_status = _get_eval_status(
        eval_result.overall_score, self._eval_metric.threshold
    )
    return eval_result
