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

from typing import TYPE_CHECKING
from typing import Union

from pydantic import TypeAdapter
from typing_extensions import override

from ..examples import example_util
from ..examples.base_example_provider import BaseExampleProvider
from ..examples.example import Example
from .base_tool import BaseTool
from .tool_configs import BaseToolConfig
from .tool_configs import ToolArgsConfig
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest


class ExampleTool(BaseTool):
  """A tool that adds (few-shot) examples to the LLM request.

  Attributes:
    examples: The examples to add to the LLM request.
  """

  def __init__(self, examples: Union[list[Example], BaseExampleProvider]):
    # Name and description are not used because this tool only changes
    # llm_request.
    super().__init__(name='example_tool', description='example tool')
    self.examples = (
        TypeAdapter(list[Example]).validate_python(examples)
        if isinstance(examples, list)
        else examples
    )

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    parts = tool_context.user_content.parts
    if not parts or not parts[0].text:
      return

    llm_request.append_instructions([
        example_util.build_example_si(
            self.examples, parts[0].text, llm_request.model
        )
    ])

  @override
  @classmethod
  def from_config(
      cls: type[ExampleTool], config: ToolArgsConfig, config_abs_path: str
  ) -> ExampleTool:
    from ..agents import config_agent_utils

    example_tool_config = ExampleToolConfig.model_validate(config.model_dump())
    if isinstance(example_tool_config.examples, str):
      example_provider = config_agent_utils.resolve_fully_qualified_name(
          example_tool_config.examples
      )
      if not isinstance(example_provider, BaseExampleProvider):
        raise ValueError(
            'Example provider must be an instance of BaseExampleProvider.'
        )
      return cls(example_provider)
    elif isinstance(example_tool_config.examples, list):
      return cls(example_tool_config.examples)
    else:
      raise ValueError(
          'Example tool config must be a list of examples or a fully-qualified'
          ' name to a BaseExampleProvider object in code.'
      )


class ExampleToolConfig(BaseToolConfig):
  examples: Union[list[Example], str]
  """The examples to add to the LLM request. User can either provide a list of
  examples or a fully-qualified name to a BaseExampleProvider object in code."""
