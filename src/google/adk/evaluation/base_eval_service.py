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

from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import AsyncGenerator
from typing import Optional

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .eval_result import EvalCaseResult


class EvaluateConfig(BaseModel):
  """Contains configurations needed to run evaluations."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  eval_metrics: list[EvalMetric] = Field(
      description="""The list of metrics to be used in Eval.""",
  )

  parallelism: int = Field(
      default=4,
      description="""Number of parallel evaluations to run during an Eval. Few
factors to consider while changing this value:

1) Your available quota with the model, especially for those metrics that use
a model as a judge. Models tend to enforce per-minute or per-second SLAs. Using
a larger value could result in the eval quickly consuming the quota.
""",
  )


class InferenceConfig(BaseModel):
  """Contains configurations need to run inferences."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  labels: Optional[dict[str, str]] = Field(
      default=None,
      description="""Labels with user-defined metadata to break down billed
charges.""",
  )

  parallelism: int = Field(
      default=4,
      description="""Number of parallel inferences to run during an Eval. Few
factors to consider while changing this value:

1) Your available quota with the model. Models tend to enforce per-minute or
per-second SLAs. Using a larger value could result in the eval quickly consuming
the quota.

2) The tools used by the Agent could also have their SLA. Using a larger value
could also overwhelm those tools.""",
  )


class InferenceRequest(BaseModel):
  """Represent a request to perform inferences for the eval cases in an eval set."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  app_name: str = Field(
      description="""The name of the app to which the eval case belongs to."""
  )

  eval_set_id: str = Field(description="""ID of the eval set.""")

  eval_case_ids: Optional[list[str]] = Field(
      default=None,
      description="""ID of the eval cases for which inferences need to be
generated.

All the eval case ids should belong to the EvalSet.

If the list of eval case ids are empty or not specified, then all the eval cases
in an eval set are evaluated.
      """,
  )

  inference_config: InferenceConfig = Field(
      description="""The config to use for inferencing.""",
  )


class InferenceStatus(Enum):
  """Status of the inference."""

  UNKNOWN = 0
  SUCCESS = 1
  FAILURE = 2


class InferenceResult(BaseModel):
  """Contains inference results for a single eval case."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  app_name: str = Field(
      description="""The name of the app to which the eval case belongs to."""
  )

  eval_set_id: str = Field(description="""ID of the eval set.""")

  eval_case_id: str = Field(
      description="""ID of the eval case for which inferences were generated.""",
  )

  inferences: Optional[list[Invocation]] = Field(
      default=None,
      description="""Inferences obtained from the Agent for the eval case.""",
  )

  session_id: Optional[str] = Field(
      description="""ID of the inference session."""
  )

  status: InferenceStatus = Field(
      default=InferenceStatus.UNKNOWN,
      description="""Status of the inference.""",
  )

  error_message: Optional[str] = Field(
      default=None,
      description="""Error message if the inference failed.""",
  )


class EvaluateRequest(BaseModel):
  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  inference_results: list[InferenceResult] = Field(
      description="""A list of inferences that need to be evaluated.""",
  )

  evaluate_config: EvaluateConfig = Field(
      description="""The config to use for evaluations.""",
  )


class BaseEvalService(ABC):
  """A service to run Evals for an ADK agent."""

  @abstractmethod
  async def perform_inference(
      self,
      inference_request: InferenceRequest,
  ) -> AsyncGenerator[InferenceResult, None]:
    """Returns InferenceResult obtained from the Agent as and when they are available.

    Args:
      inference_request: The request for generating inferences.
    """

  @abstractmethod
  async def evaluate(
      self,
      evaluate_request: EvaluateRequest,
  ) -> AsyncGenerator[EvalCaseResult, None]:
    """Returns EvalCaseResult for each item as and when they are available.

    Args:
      evaluate_request: The request to perform metric evaluations on the
        inferences.
    """
