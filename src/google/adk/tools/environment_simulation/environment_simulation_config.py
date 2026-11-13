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
from pydantic import ValidationError  # noqa: F401
from typing_extensions import Self

from ..tool_args_matcher._tool_args_matcher import ToolArgsMatcher


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

  args_matcher: Optional[ToolArgsMatcher] = None
  """How the request's arguments are compared with match_args.

  If not provided, every entry of match_args must appear in the request with
  an equal value. Set it to loosen the comparison per argument, for example to
  match a free-text query semantically:

  ```json
  {"per_arg_matchers": {"query": {"type": "semantic"}}}
  ```

  Configs whose match_args equal the request are tried first, in order, before
  any args_matcher comparison runs, so an exact recording never depends on a
  model's verdict. This check stops at the first config without match_args,
  which keeps its priority over the configs after it.

  Requires match_args."""

  injected_latency_seconds: float = Field(default=0.0, le=120.0)
  """Inject latency to the tool call. Please note it may not be accurate if                                                                                                                                                                                             │
  the interceptor is applied as after tool callback."""

  random_seed: Optional[int] = None
  """Seeds this injection's own generator once when the plugin or callback is
  created. A call draws the next value only when it matches this injection and
  no injection tried before it answered it, and one plugin's sequence is shared
  by every session it serves. None draws from a shared unseeded generator."""

  injected_error: Optional[InjectedError] = None
  """The injected error."""

  injected_response: Optional[Dict[str, Any]] = None
  """The injected response."""

  @model_validator(mode="after")
  def check_injected_error_or_response(self) -> Self:
    """Checks that either injected_error or injected_response is set."""
    if (self.injected_error is None) == (self.injected_response is None):
      raise ValueError(
          "Either injected_error or injected_response must be set, but not"
          " both, and not neither."
      )
    return self

  @model_validator(mode="after")
  def check_args_matcher_has_match_args(self) -> Self:
    """Checks that args_matcher is only set together with match_args."""
    if self.args_matcher is not None and not self.match_args:
      raise ValueError(
          "args_matcher requires match_args: without match_args the injection"
          " applies to every request and args_matcher is never used."
      )
    return self


class MockStrategy(enum.Enum):
  """Mock strategy for a tool."""

  MOCK_STRATEGY_UNSPECIFIED = 0

  MOCK_STRATEGY_TOOL_SPEC = 1
  """Use tool specifications to mock the tool response."""

  MOCK_STRATEGY_TRACING = 2
  """Deprecated, please use MOCK_STRATEGY_TOOL_SPEC with tracing input."""


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


class EnvironmentSimulationConfig(BaseModel):
  """Configuration for EnvironmentSimulation."""

  tool_simulation_configs: List[ToolSimulationConfig] = Field(
      default_factory=list
  )
  """A list of tool simulation configurations."""

  simulation_model: str = Field(default="gemini-2.5-flash")
  """The model to use for internal simulator LLM calls (tool analysis, mock responses)."""

  simulation_model_configuration: genai_types.GenerateContentConfig = Field(
      default_factory=lambda: genai_types.GenerateContentConfig(
          thinking_config=genai_types.ThinkingConfig(
              include_thoughts=False,
              thinking_budget=10240,
          )
      ),
  )
  """The configuration for the internal simulator LLM calls."""

  tracing: Optional[str] = None
  """Tracing data (e.g., a prior agent run trace in JSON string format) to
  provide historical context for mock generation. Passed directly to mock
  strategies alongside environment_data."""

  environment_data: Optional[str] = None
  """Environment-specific data (e.g., a minimal database dump in JSON string
   format). This data is passed directly to mock strategies for contextual
   mock generation."""

  @field_validator("tool_simulation_configs")
  @classmethod
  def check_tool_simulation_configs(
      cls, v: List[ToolSimulationConfig]
  ) -> List[ToolSimulationConfig]:
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
