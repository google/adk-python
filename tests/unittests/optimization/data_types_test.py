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

from google.adk.optimization.data_types import AgentWithScores
from google.adk.optimization.data_types import OptimizerResult
from google.adk.optimization.data_types import SamplingResult
from google.adk.optimization.local_eval_sampler import LocalEvalSamplerConfig
from pydantic import BaseModel
from pydantic import ValidationError
import pytest


@pytest.mark.parametrize(
    ("model", "field_names"),
    [
        (SamplingResult, ["scores"]),
        (LocalEvalSamplerConfig, ["eval_config", "app_name", "train_eval_set"]),
    ],
)
def test_json_schema_marks_required_fields_at_object_level(
    model: type[BaseModel], field_names: list[str]
):
  schema = model.model_json_schema()

  for field_name in field_names:
    # JSON Schema's `required` is an array on the object, so a boolean
    # `required` inside a property schema is invalid.
    assert "required" not in schema["properties"][field_name]
    assert field_name in schema["required"]


@pytest.mark.parametrize(
    ("model", "field_name"),
    [
        (SamplingResult, "scores"),
        (AgentWithScores, "optimized_agent"),
        (OptimizerResult, "optimized_agents"),
        (LocalEvalSamplerConfig, "eval_config"),
        (LocalEvalSamplerConfig, "app_name"),
        (LocalEvalSamplerConfig, "train_eval_set"),
    ],
)
def test_fields_without_default_are_required(
    model: type[BaseModel], field_name: str
):
  with pytest.raises(ValidationError) as exc_info:
    model.model_validate({})

  missing = {
      error["loc"][0]
      for error in exc_info.value.errors()
      if error["type"] == "missing"
  }
  assert field_name in missing
