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

import enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from google.genai import types as genai_types
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic_core import ValidationError


class InjectedError(BaseModel):
  """An error to be injected into a tool call."""

  injected_http_error_code: int
  """Inject http error code to the tool call. Will present as "error_code"
  in the tool response dict."""

  error_message: str
  """Inject error message to the tool call. Will present as
  "error_message" in the tool response dict."""


class InjectionConfig(BaseModel):
  """Injection configuration for a tool."""

  injection_probability: float = 1.0
  """Probability of injecting the injected_value."""

  match_args: Optional[Dict[str, Any]] = None
  """Only apply injection if the request matches the match_args.
  If match_args is not provided, the injection will be applied to all
  requests."""

  injected_latency_seconds: float = Field(default=0.0, le=120.0)
  """Inject latency to the tool call. Please note it may not be accurate if                                                                                                                                                                                             │
  the interceptor is applied as after tool callback."""

  random_seed: Optional[int] = None
  """The random seed to use for this injection."""

  injected_error: Optional[InjectedError] = None
  """The injected error."""

  injected_response: Optional[Dict[str, Any]] = None
  """The injected response."""

  @model_validator(mode="after")
  def check_injected_error_or_response(self) -> Self:
    """Checks that either injected_error or injected_response is set."""
    if bool(self.injected_error) == bool(self.injected_response):
      raise ValueError(
          "Either injected_error or injected_response must be set, but not"
          " both, and not neither."
      )
    return self


class MockStrategy(enum.Enum):
  """Mock strategy for a tool."""

  MOCK_STRATEGY_UNSPECIFIED = 0

  MOCK_STRATEGY_TOOL_SPEC = 1
  """Use tool specifications to mock the tool response."""

  MOCK_STRATEGY_TRACING = 2
  """Use provided tracing and tool specifications to mock the tool
  response based on llm response. Need to provide tracing path in
  command."""


class ToolSimulationConfig(BaseModel):
  """Simulation configuration for a single tool."""

  tool_name: str
  """Name of the tool to be simulated."""

  injection_configs: List[InjectionConfig] = Field(default_factory=list)
  """Injection configuration for the tool. If provided, the tool will be
  injected with the injected_value with the injection_probability first,
  the mock_strategy will be applied if no injection config is hit."""

  mock_strategy_type: MockStrategy = MockStrategy.MOCK_STRATEGY_UNSPECIFIED
  """The mock strategy to use."""

  @model_validator(mode="after")
  def check_mock_strategy_type(self) -> Self:
    """Checks that mock_strategy_type is not UNSPECIFIED if no injections."""
    if (
        not self.injection_configs
        and self.mock_strategy_type == MockStrategy.MOCK_STRATEGY_UNSPECIFIED
    ):
      raise ValueError(
          "If injection_configs is empty, mock_strategy_type cannot be"
          " MOCK_STRATEGY_UNSPECIFIED."
      )
    return self


class AgentSimulatorConfig(BaseModel):
  """Configuration for AgentSimulator."""

  tool_simulation_configs: List[ToolSimulationConfig] = Field(
      default_factory=list
  )
  """A list of tool simulation configurations."""

  simulation_model: str = Field(default="gemini-2.5-flash")
  """The model to use for internal simulator LLM calls (tool analysis, mock responses)."""

  simulation_model_configuration: genai_types.GenerateContentConfig = Field(
      default_factory=lambda: genai_types.GenerateContentConfig(
          thinking_config=genai_types.ThinkingConfig(
              include_thoughts=True,
              thinking_budget=10240,
          )
      ),
  )
  """The configuration for the internal simulator LLM calls."""

  tracing_path: Optional[str] = None
  """The path to the tracing file to be used for mocking. Only used if the
  mock_strategy_type is MOCK_STRATEGY_TRACING."""

  @field_validator("tool_simulation_configs")
  @classmethod
  def check_tool_simulation_configs(cls, v: List[ToolSimulationConfig]):
    """Checks that tool_simulation_configs is not empty."""
    if not v:
      raise ValueError("tool_simulation_configs must be provided.")
    seen_tool_names = set()
    for tool_sim_config in v:
      if tool_sim_config.tool_name in seen_tool_names:
        raise ValueError(
            f"Duplicate tool_name found: {tool_sim_config.tool_name}"
        )
      seen_tool_names.add(tool_sim_config.tool_name)
    return v
