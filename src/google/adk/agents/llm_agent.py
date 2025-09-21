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
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Type
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import override
from typing_extensions import TypeAlias

from ..code_executors.base_code_executor import BaseCodeExecutor
from ..events.event import Event
from ..flows.llm_flows.auto_flow import AutoFlow
from ..flows.llm_flows.base_llm_flow import BaseLlmFlow
from ..flows.llm_flows.single_flow import SingleFlow
from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ..planners.base_planner import BasePlanner
from ..tools.base_tool import BaseTool
from ..tools.base_toolset import BaseToolset
from ..tools.function_tool import FunctionTool
from ..tools.tool_configs import ToolConfig
from ..tools.tool_context import ToolContext
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental
from .base_agent import BaseAgent
from .base_agent_config import BaseAgentConfig
from .callback_context import CallbackContext
from .invocation_context import InvocationContext
from .llm_agent_config import LlmAgentConfig
from .readonly_context import ReadonlyContext

logger = logging.getLogger('google_adk.' + __name__)

_SingleBeforeModelCallback: TypeAlias = Callable[
    [CallbackContext, LlmRequest],
    Union[Awaitable[Optional[LlmResponse]], Optional[LlmResponse]],
]

BeforeModelCallback: TypeAlias = Union[
    _SingleBeforeModelCallback,
    list[_SingleBeforeModelCallback],
]

_SingleAfterModelCallback: TypeAlias = Callable[
    [CallbackContext, LlmResponse],
    Union[Awaitable[Optional[LlmResponse]], Optional[LlmResponse]],
]

AfterModelCallback: TypeAlias = Union[
    _SingleAfterModelCallback,
    list[_SingleAfterModelCallback],
]

_SingleBeforeToolCallback: TypeAlias = Callable[
    [BaseTool, dict[str, Any], ToolContext],
    Union[Awaitable[Optional[dict]], Optional[dict]],
]

BeforeToolCallback: TypeAlias = Union[
    _SingleBeforeToolCallback,
    list[_SingleBeforeToolCallback],
]

_SingleAfterToolCallback: TypeAlias = Callable[
    [BaseTool, dict[str, Any], ToolContext, dict],
    Union[Awaitable[Optional[dict]], Optional[dict]],
]

AfterToolCallback: TypeAlias = Union[
    _SingleAfterToolCallback,
    list[_SingleAfterToolCallback],
]

InstructionProvider: TypeAlias = Callable[
    [ReadonlyContext], Union[str, Awaitable[str]]
]

ToolUnion: TypeAlias = Union[Callable, BaseTool, BaseToolset]


async def _convert_tool_union_to_tools(
    tool_union: ToolUnion, ctx: ReadonlyContext
) -> list[BaseTool]:
  if isinstance(tool_union, BaseTool):
    return [tool_union]
  if callable(tool_union):
    return [FunctionTool(func=tool_union)]

  # At this point, tool_union must be a BaseToolset
  return await tool_union.get_tools_with_prefix(ctx)


class LlmAgent(BaseAgent):
  """LLM-based Agent."""

  model: Union[str, BaseLlm] = ''
  """The model to use for the agent.

  When not set, the agent will inherit the model from its ancestor.
  """

  config_type: ClassVar[Type[BaseAgentConfig]] = LlmAgentConfig
  """The config type for this agent."""

  instruction: Union[str, InstructionProvider] = ''
  """Dynamic instructions for the LLM model, guiding the agent's behavior.

  These instructions can contain placeholders like {variable_name} that will be
  resolved at runtime using session state and context.

  **Behavior depends on static_instruction:**
  - If static_instruction is None: instruction goes to system_instruction
  - If static_instruction is set: instruction goes to user content in the request

  This allows for context caching optimization where static content (static_instruction)
  comes first in the prompt, followed by dynamic content (instruction).
  """

  global_instruction: Union[str, InstructionProvider] = ''
  """Instructions for all the agents in the entire agent tree.

  ONLY the global_instruction in root agent will take effect.

  For example: use global_instruction to make all agents have a stable identity
  or personality.
  """

  static_instruction: Optional[types.Content] = None
  """Static instruction content sent literally as system instruction at the beginning.

  This field is for content that never changes and doesn't contain placeholders.
  It's sent directly to the model without any processing or variable substitution.

  This field is primarily for context caching optimization. Static instructions
  are sent as system instruction at the beginning of the request, allowing
  for improved performance when the static portion remains unchanged. Live API
  has its own cache mechanism, thus this field doesn't work with Live API.

  **Impact on instruction field:**
  - When static_instruction is None: instruction → system_instruction
  - When static_instruction is set: instruction → user content (after static content)

  **Context Caching:**
  - **Implicit Cache**: Automatic caching by model providers (no config needed)
  - **Explicit Cache**: Cache explicitly created by user for instructions, tools and contents

  See below for more information of Implicit Cache and Explicit Cache
  Gemini API: https://ai.google.dev/gemini-api/docs/caching?lang=python
  Vertex API: https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview

  Setting static_instruction alone does NOT enable caching automatically.
  For explicit caching control, configure context_cache_config at App level.

  **Content Support:**
  Can contain text, files, binaries, or any combination as types.Content
  supports multiple part types (text, inline_data, file_data, etc.).

  **Example:**
  ```python
  static_instruction = types.Content(
      role='user',
      parts=[
          types.Part(text='You are a helpful assistant.'),
          types.Part(file_data=types.FileData(...))
      ]
  )
  ```
  """

  tools: list[ToolUnion] = Field(default_factory=list)
  """Tools available to this agent."""

  generate_content_config: Optional[types.GenerateContentConfig] = None
  """The additional content generation configurations.

  NOTE: not all fields are usable, e.g. tools must be configured via `tools`,
  thinking_config must be configured via `planner` in LlmAgent.

  For example: use this config to adjust model temperature, configure safety
  settings, etc.
  """

  # LLM-based agent transfer configs - Start
  disallow_transfer_to_parent: bool = False
  """Disallows LLM-controlled transferring to the parent agent.

  NOTE: Setting this as True also prevents this agent to continue reply to the
  end-user. This behavior prevents one-way transfer, in which end-user may be
  stuck with one agent that cannot transfer to other agents in the agent tree.
  """
  disallow_transfer_to_peers: bool = False
  """Disallows LLM-controlled transferring to the peer agents."""
  # LLM-based agent transfer configs - End

  include_contents: Literal['default', 'none'] = 'default'
  """Controls content inclusion in model requests.

  Options:
    default: Model receives relevant conversation history
    none: Model receives no prior history, operates solely on current
    instruction and input
  """

  # Controlled input/output configurations - Start
  input_schema: Optional[type[BaseModel]] = None
  """The input schema when agent is used as a tool."""
  output_schema: Optional[type[BaseModel]] = None
  """The output schema when agent replies.

  NOTE:
    When this is set, agent can ONLY reply and CANNOT use any tools, such as
    function tools, RAGs, agent transfer, etc.
  """
  output_key: Optional[str] = None
  """The key in session state to store the output of the agent.

  Typically use cases:
  - Extracts agent reply for later use, such as in tools, callbacks, etc.
  - Connects agents to coordinate with each other.
  """
  # Controlled input/output configurations - End

  # Advance features - Start
  planner: Optional[BasePlanner] = None
  """Instructs the agent to make a plan and execute it step by step.

  NOTE:
    To use model's built-in thinking features, set the `thinking_config`
    field in `google.adk.planners.built_in_planner`.
  """

  code_executor: Optional[BaseCodeExecutor] = None
  """Allow agent to execute code blocks from model responses using the provided
  CodeExecutor.

  Check out available code executions in `google.adk.code_executor` package.

  NOTE:
    To use model's built-in code executor, use the `BuiltInCodeExecutor`.
  """
  # Advance features - End

  # Callbacks - Start
  before_model_callback: Optional[BeforeModelCallback] = None
  """Callback or list of callbacks to be called before calling the LLM.

  When a list of callbacks is provided, the callbacks will be called in the
  order they are listed until a callback does not return None.

  Args:
    callback_context: CallbackContext,
    llm_request: LlmRequest, The raw model request. Callback can mutate the
    request.

  Returns:
    The content to return to the user. When present, the model call will be
    skipped and the provided content will be returned to user.
  """
  after_model_callback: Optional[AfterModelCallback] = None
  """Callback or list of callbacks to be called after calling the LLM.

  When a list of callbacks is provided, the callbacks will be called in the
  order they are listed until a callback does not return None.

  Args:
    callback_context: CallbackContext,
    llm_response: LlmResponse, the actual model response.

  Returns:
    The content to return to the user. When present, the actual model response
    will be ignored and the provided content will be returned to user.
  """
  before_tool_callback: Optional[BeforeToolCallback] = None
  """Callback or list of callbacks to be called before calling the tool.

  When a list of callbacks is provided, the callbacks will be called in the
  order they are listed until a callback does not return None.

  Args:
    tool: The tool to be called.
    args: The arguments to the tool.
    tool_context: ToolContext,

  Returns:
    The tool response. When present, the returned tool response will be used and
    the framework will skip calling the actual tool.
  """
  after_tool_callback: Optional[AfterToolCallback] = None
  """Callback or list of callbacks to be called after calling the tool.

  When a list of callbacks is provided, the callbacks will be called in the
  order they are listed until a callback does not return None.

  Args:
    tool: The tool to be called.
    args: The arguments to the tool.
    tool_context: ToolContext,
    tool_response: The response from the tool.

  Returns:
    When present, the returned dict will be used as tool result.
  """
  # Callbacks - End

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    async with Aclosing(self._llm_flow.run_async(ctx)) as agen:
      async for event in agen:
        self.__maybe_save_output_to_state(event)
        yield event

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    async with Aclosing(self._llm_flow.run_live(ctx)) as agen:
      async for event in agen:
        self.__maybe_save_output_to_state(event)
        yield event
      if ctx.end_invocation:
        return

  @property
  def canonical_model(self) -> BaseLlm:
    """The resolved self.model field as BaseLlm.

    This method is only for use by Agent Development Kit.
    """
    if isinstance(self.model, BaseLlm):
      return self.model
    elif self.model:  # model is non-empty str
      return LLMRegistry.new_llm(self.model)
    else:  # find model from ancestors.
      ancestor_agent = self.parent_agent
      while ancestor_agent is not None:
        if isinstance(ancestor_agent, LlmAgent):
          return ancestor_agent.canonical_model
        ancestor_agent = ancestor_agent.parent_agent
      raise ValueError(f'No model found for {self.name}.')

  async def canonical_instruction(
      self, ctx: ReadonlyContext
  ) -> tuple[str, bool]:
    """The resolved self.instruction field to construct instruction for this agent.

    This method is only for use by Agent Development Kit.

    Args:
      ctx: The context to retrieve the session state.

    Returns:
      A tuple of (instruction, bypass_state_injection).
      instruction: The resolved self.instruction field.
      bypass_state_injection: Whether the instruction is based on
      InstructionProvider.
    """
    if isinstance(self.instruction, str):
      return self.instruction, False
    else:
      instruction = self.instruction(ctx)
      if inspect.isawaitable(instruction):
        instruction = await instruction
      return instruction, True

  async def canonical_global_instruction(
      self, ctx: ReadonlyContext
  ) -> tuple[str, bool]:
    """The resolved self.instruction field to construct global instruction.

    This method is only for use by Agent Development Kit.

    Args:
      ctx: The context to retrieve the session state.

    Returns:
      A tuple of (instruction, bypass_state_injection).
      instruction: The resolved self.global_instruction field.
      bypass_state_injection: Whether the instruction is based on
      InstructionProvider.
    """
    if isinstance(self.global_instruction, str):
      return self.global_instruction, False
    else:
      global_instruction = self.global_instruction(ctx)
      if inspect.isawaitable(global_instruction):
        global_instruction = await global_instruction
      return global_instruction, True

  async def canonical_tools(
      self, ctx: ReadonlyContext = None
  ) -> list[BaseTool]:
    """The resolved self.tools field as a list of BaseTool based on the context.

    This method is only for use by Agent Development Kit.
    """
    resolved_tools = []
    for tool_union in self.tools:
      resolved_tools.extend(await _convert_tool_union_to_tools(tool_union, ctx))
    return resolved_tools

  @property
  def canonical_before_model_callbacks(
      self,
  ) -> list[_SingleBeforeModelCallback]:
    """The resolved self.before_model_callback field as a list of _SingleBeforeModelCallback.

    This method is only for use by Agent Development Kit.
    """
    if not self.before_model_callback:
      return []
    if isinstance(self.before_model_callback, list):
      return self.before_model_callback
    return [self.before_model_callback]

  @property
  def canonical_after_model_callbacks(self) -> list[_SingleAfterModelCallback]:
    """The resolved self.after_model_callback field as a list of _SingleAfterModelCallback.

    This method is only for use by Agent Development Kit.
    """
    if not self.after_model_callback:
      return []
    if isinstance(self.after_model_callback, list):
      return self.after_model_callback
    return [self.after_model_callback]

  @property
  def canonical_before_tool_callbacks(
      self,
  ) -> list[BeforeToolCallback]:
    """The resolved self.before_tool_callback field as a list of BeforeToolCallback.

    This method is only for use by Agent Development Kit.
    """
    if not self.before_tool_callback:
      return []
    if isinstance(self.before_tool_callback, list):
      return self.before_tool_callback
    return [self.before_tool_callback]

  @property
  def canonical_after_tool_callbacks(
      self,
  ) -> list[AfterToolCallback]:
    """The resolved self.after_tool_callback field as a list of AfterToolCallback.

    This method is only for use by Agent Development Kit.
    """
    if not self.after_tool_callback:
      return []
    if isinstance(self.after_tool_callback, list):
      return self.after_tool_callback
    return [self.after_tool_callback]

  @property
  def _llm_flow(self) -> BaseLlmFlow:
    if (
        self.disallow_transfer_to_parent
        and self.disallow_transfer_to_peers
        and not self.sub_agents
    ):
      return SingleFlow()
    else:
      return AutoFlow()

  def __maybe_save_output_to_state(self, event: Event):
    """Saves the model output to state if needed."""
    # skip if the event was authored by some other agent (e.g. current agent
    # transferred to another agent)
    if event.author != self.name:
      logger.debug(
          'Skipping output save for agent %s: event authored by %s',
          self.name,
          event.author,
      )
      return
    if (
        self.output_key
        and event.is_final_response()
        and event.content
        and event.content.parts
    ):

      result = ''.join(
          part.text
          for part in event.content.parts
          if part.text and not part.thought
      )
      if self.output_schema:
        # If the result from the final chunk is just whitespace or empty,
        # it means this is an empty final chunk of a stream.
        # Do not attempt to parse it as JSON.
        if not result.strip():
          return
        result = self.output_schema.model_validate_json(result).model_dump(
            exclude_none=True
        )
      event.actions.state_delta[self.output_key] = result

  @model_validator(mode='after')
  def __model_validator_after(self) -> LlmAgent:
    self.__check_output_schema()
    return self

  def __check_output_schema(self):
    if not self.output_schema:
      return

    if (
        not self.disallow_transfer_to_parent
        or not self.disallow_transfer_to_peers
    ):
      logger.warning(
          'Invalid config for agent %s: output_schema cannot co-exist with'
          ' agent transfer configurations. Setting'
          ' disallow_transfer_to_parent=True, disallow_transfer_to_peers=True',
          self.name,
      )
      self.disallow_transfer_to_parent = True
      self.disallow_transfer_to_peers = True

    if self.sub_agents:
      raise ValueError(
          f'Invalid config for agent {self.name}: if output_schema is set,'
          ' sub_agents must be empty to disable agent transfer.'
      )

  @field_validator('generate_content_config', mode='after')
  @classmethod
  def validate_generate_content_config(
      cls, generate_content_config: Optional[types.GenerateContentConfig]
  ) -> types.GenerateContentConfig:
    if not generate_content_config:
      return types.GenerateContentConfig()
    if generate_content_config.thinking_config:
      raise ValueError('Thinking config should be set via LlmAgent.planner.')
    if generate_content_config.tools:
      raise ValueError('All tools must be set via LlmAgent.tools.')
    if generate_content_config.system_instruction:
      raise ValueError(
          'System instruction must be set via LlmAgent.instruction.'
      )
    if generate_content_config.response_schema:
      raise ValueError(
          'Response schema must be set via LlmAgent.output_schema.'
      )
    return generate_content_config

  @classmethod
  @experimental
  def _resolve_tools(
      cls, tool_configs: list[ToolConfig], config_abs_path: str
  ) -> list[Any]:
    """Resolve tools from configuration.

    Args:
      tool_configs: List of tool configurations (ToolConfig objects).
      config_abs_path: The absolute path to the agent config file.

    Returns:
      List of resolved tool objects.
    """

    resolved_tools = []
    for tool_config in tool_configs:
      if '.' not in tool_config.name:
        # ADK built-in tools
        module = importlib.import_module('google.adk.tools')
        obj = getattr(module, tool_config.name)
      else:
        # User-defined tools
        module_path, obj_name = tool_config.name.rsplit('.', 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)

      if isinstance(obj, BaseTool) or isinstance(obj, BaseToolset):
        logger.debug(
            'Tool %s is an instance of BaseTool/BaseToolset.', tool_config.name
        )
        resolved_tools.append(obj)
      elif inspect.isclass(obj) and (
          issubclass(obj, BaseTool) or issubclass(obj, BaseToolset)
      ):
        logger.debug(
            'Tool %s is a sub-class of BaseTool/BaseToolset.', tool_config.name
        )
        resolved_tools.append(
            obj.from_config(tool_config.args, config_abs_path)
        )
      elif callable(obj):
        if tool_config.args:
          logger.debug(
              'Tool %s is a user-defined tool-generating function.',
              tool_config.name,
          )
          resolved_tools.append(obj(tool_config.args))
        else:
          logger.debug(
              'Tool %s is a user-defined function tool.', tool_config.name
          )
          resolved_tools.append(obj)
      else:
        raise ValueError(f'Invalid tool YAML config: {tool_config}.')

    return resolved_tools

  @override
  @classmethod
  @experimental
  def _parse_config(
      cls: Type[LlmAgent],
      config: LlmAgentConfig,
      config_abs_path: str,
      kwargs: Dict[str, Any],
  ) -> Dict[str, Any]:
    from .config_agent_utils import resolve_callbacks
    from .config_agent_utils import resolve_code_reference

    if config.model:
      kwargs['model'] = config.model
    if config.instruction:
      kwargs['instruction'] = config.instruction
    if config.static_instruction:
      kwargs['static_instruction'] = config.static_instruction
    if config.disallow_transfer_to_parent:
      kwargs['disallow_transfer_to_parent'] = config.disallow_transfer_to_parent
    if config.disallow_transfer_to_peers:
      kwargs['disallow_transfer_to_peers'] = config.disallow_transfer_to_peers
    if config.include_contents != 'default':
      kwargs['include_contents'] = config.include_contents
    if config.input_schema:
      kwargs['input_schema'] = resolve_code_reference(config.input_schema)
    if config.output_schema:
      kwargs['output_schema'] = resolve_code_reference(config.output_schema)
    if config.output_key:
      kwargs['output_key'] = config.output_key
    if config.tools:
      kwargs['tools'] = cls._resolve_tools(config.tools, config_abs_path)
    if config.before_model_callbacks:
      kwargs['before_model_callback'] = resolve_callbacks(
          config.before_model_callbacks
      )
    if config.after_model_callbacks:
      kwargs['after_model_callback'] = resolve_callbacks(
          config.after_model_callbacks
      )
    if config.before_tool_callbacks:
      kwargs['before_tool_callback'] = resolve_callbacks(
          config.before_tool_callbacks
      )
    if config.after_tool_callbacks:
      kwargs['after_tool_callback'] = resolve_callbacks(
          config.after_tool_callbacks
      )
    if config.generate_content_config:
      kwargs['generate_content_config'] = config.generate_content_config

    return kwargs


Agent: TypeAlias = LlmAgent
