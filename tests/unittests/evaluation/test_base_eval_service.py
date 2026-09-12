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

from google.adk.evaluation.base_eval_service import EvaluateConfig
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.eval_metrics import EvalMetric
from pydantic import ValidationError
import pytest


def test_evaluate_config_rejects_zero_parallelism():
  """parallelism=0 must be rejected.

  It is forwarded unclamped to asyncio.Semaphore(value=parallelism), where a
  zero value hangs every acquire() indefinitely instead of raising -- a
  degenerate value that fails silently rather than loudly.
  """
  with pytest.raises(ValidationError):
    EvaluateConfig(
        eval_metrics=[EvalMetric(metric_name="response_match_score")],
        parallelism=0,
    )


def test_inference_config_rejects_zero_parallelism():
  """Same defect as EvaluateConfig.parallelism, on the inference side."""
  with pytest.raises(ValidationError):
    InferenceConfig(parallelism=0)
