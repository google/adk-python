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

"""LlmAgent: drop-in replacement using Mesh + BaseLlmAgent composition.

Combines Mesh (multi-agent orchestration) with BaseLlmAgent.
Constructs internal nodes (_SingleLlmAgent for leaves, recursive
LlmAgent for non-leaves) in model_post_init.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any
from typing import AsyncGenerator
from typing import ClassVar
from typing import Dict
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import warnings

from google.genai import types
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import override
from typing_extensions import Self
from typing_extensions import TypeAlias

from ..events.event import Event
from ..events.event_actions import EventActions
from ..features import experimental
from ..features import FeatureName
from ..models.base_llm import BaseLlm
from ..models.registry import LLMRegistry
from ..tools.base_tool import BaseTool
from ..tools.base_toolset import BaseToolset
from ..tools.tool_configs import ToolConfig
from ..utils._schema_utils import validate_schema
from ..utils.context_utils import Aclosing
from ..workflow._base_node import BaseNode
from .base_agent import BaseAgentState
from .base_agent_config import BaseAgentConfig
from .context import Context
from .llm._base_llm_agent import _SingleAfterModelCallback
from .llm._base_llm_agent import _SingleAfterToolCallback
from .llm._base_llm_agent import _SingleBeforeModelCallback
from .llm._base_llm_agent import _SingleBeforeToolCallback
from .llm._base_llm_agent import _SingleOnModelErrorCallback
from .llm._base_llm_agent import _SingleOnToolErrorCallback
from .llm._base_llm_agent import AfterModelCallback
from .llm._base_llm_agent import AfterToolCallback
from .llm._base_llm_agent import BaseLlmAgent
from .llm._base_llm_agent import BeforeModelCallback
from .llm._base_llm_agent import BeforeToolCallback
from .llm._base_llm_agent import InstructionProvider
from .llm._base_llm_agent import OnModelErrorCallback
from .llm._base_llm_agent import OnToolErrorCallback
from .llm._base_llm_agent import ToolUnion
from .llm._mesh import _Mesh
from .llm._single_llm_agent import _convert_tool_union_to_tools
from .llm._single_llm_agent import _SingleLlmAgent
from .llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME
from .llm.task._finish_task_tool import FinishTaskTool
from .llm.task._request_task_tool import RequestTaskTool
from .llm_agent_config import LlmAgentConfig

if TYPE_CHECKING:
  from .invocation_context import InvocationContext as BaseInvocationContext
  from .readonly_context import ReadonlyContext

logger = logging.getLogger('google_adk.' + __name__)

# Fixed child node names from _SingleLlmAgent's internal graph
# (call_llm and execute_tools FunctionNodes).
_COORDINATOR_CHILD_NAMES = frozenset({'call_llm', 'execute_tools'})


class LlmAgent(_Mesh, BaseLlmAgent):
  """Workflow based implementation of LlmAgent. Internally constructs
  ``_SingleLlmAgent`` (for leaf agents) or nested ``LlmAgent`` (for
  agents with their own sub_agents) and delegates to ``_Mesh``
  for orchestration.

  The coordinator node (a ``_SingleLlmAgent`` with the same name as
  this ``LlmAgent``) handles this agent's own LLM reasoning loop.
  """

  rerun_on_resume: bool = True
  """LlmAgent must re-run on resume so its internal workflow
  (call_llm/execute_tools) can properly handle interrupted nodes
  like auth, confirmation, and long-running tools."""

  DEFAULT_MODEL: ClassVar[str] = 'gemini-2.5-flash'
  """System default model used when no model is set on an agent."""

  _default_model: ClassVar[Union[str, BaseLlm]] = DEFAULT_MODEL
  """Current default model used when an agent has no model set."""

  config_type: ClassVar[Type[BaseAgentConfig]] = LlmAgentConfig
  """The config type for this agent."""

  # ------------------------------------------------------------------
  # Coordinator access
  # ------------------------------------------------------------------

  @property
  def _coordinator(self) -> _SingleLlmAgent:
    """The coordinator _SingleLlmAgent (first node in the mesh)."""
    return self.nodes[0]

  def _update_mode(self, mode: str) -> None:
    """Updates mode and rebuilds internal nodes.

    The coordinator and task tools are created during model_post_init
    based on the original mode.  When the mode changes after construction
    (e.g. auto-defaulting to single_turn in a workflow), the internal
    nodes must be rebuilt so that task tools are injected and the
    coordinator is configured correctly for the new mode.
    """
    self.mode = mode
    self._build_nodes()

  # ------------------------------------------------------------------
  # canonical_* properties (delegated to coordinator)
  # ------------------------------------------------------------------

  @property
  def canonical_model(self) -> BaseLlm:
    """The resolved self.model field as BaseLlm.

    This method is only for use by Agent Development Kit.
    """
    if isinstance(self.model, BaseLlm):
      return self.model
    elif self.model:
      return LLMRegistry.new_llm(self.model)
    else:
      ancestor_agent = self.parent_agent
      while ancestor_agent is not None:
        if isinstance(ancestor_agent, LlmAgent):
          return ancestor_agent.canonical_model
        ancestor_agent = ancestor_agent.parent_agent
      return self._resolve_default_model()

  @classmethod
  def _resolve_default_model(cls) -> BaseLlm:
    """Resolves the current default model to a BaseLlm instance."""
    default_model = cls._default_model
    if isinstance(default_model, BaseLlm):
      return default_model
    return LLMRegistry.new_llm(default_model)

  async def canonical_instruction(
      self, ctx: ReadonlyContext
  ) -> tuple[str, bool]:
    """The resolved self.instruction field to construct instruction.

    This method is only for use by Agent Development Kit.
    """
    return await self._coordinator.canonical_instruction(ctx)

  async def canonical_global_instruction(
      self, ctx: ReadonlyContext
  ) -> tuple[str, bool]:
    """The resolved self.instruction field to construct global instruction.

    This method is only for use by Agent Development Kit.
    """
    return await self._coordinator.canonical_global_instruction(ctx)

  async def canonical_tools(
      self, ctx: Optional[ReadonlyContext] = None
  ) -> list[BaseTool]:
    """The resolved self.tools field as a list of BaseTool.

    This method is only for use by Agent Development Kit.
    """
    import asyncio

    from .llm._single_llm_agent import _convert_tool_union_to_tools

    multiple_tools = len(self.tools) > 1
    model = self.canonical_model

    results = await asyncio.gather(*(
        _convert_tool_union_to_tools(tool_union, ctx, model, multiple_tools)
        for tool_union in self.tools
    ))

    resolved_tools = []
    for tools in results:
      resolved_tools.extend(tools)

    return resolved_tools

  @property
  def canonical_before_model_callbacks(self) -> list:
    """The resolved self.before_model_callback as a list.

    This method is only for use by Agent Development Kit.
    """
    return self._coordinator.canonical_before_model_callbacks

  @property
  def canonical_after_model_callbacks(self) -> list:
    """The resolved self.after_model_callback as a list.

    This method is only for use by Agent Development Kit.
    """
    return self._coordinator.canonical_after_model_callbacks

  @property
  def canonical_on_model_error_callbacks(self) -> list:
    """The resolved self.on_model_error_callback as a list.

    This method is only for use by Agent Development Kit.
    """
    return self._coordinator.canonical_on_model_error_callbacks

  @property
  def canonical_before_tool_callbacks(self) -> list:
    """The resolved self.before_tool_callback as a list.

    This method is only for use by Agent Development Kit.
    """
    return self._coordinator.canonical_before_tool_callbacks

  @property
  def canonical_after_tool_callbacks(self) -> list:
    """The resolved self.after_tool_callback as a list.

    This method is only for use by Agent Development Kit.
    """
    return self._coordinator.canonical_after_tool_callbacks

  @property
  def canonical_on_tool_error_callbacks(self) -> list:
    """The resolved self.on_tool_error_callback as a list.

    This method is only for use by Agent Development Kit.
    """
    return self._coordinator.canonical_on_tool_error_callbacks

  # ------------------------------------------------------------------
  # Class methods
  # ------------------------------------------------------------------

  @classmethod
  def set_default_model(cls, model: Union[str, BaseLlm]) -> None:
    """Overrides the default model used when an agent has no model set."""
    if not isinstance(model, (str, BaseLlm)):
      raise TypeError('Default model must be a model name or BaseLlm.')
    if isinstance(model, str) and not model:
      raise ValueError('Default model must be a non-empty string.')
    cls._default_model = model

  # ------------------------------------------------------------------
  # Output key support
  # ------------------------------------------------------------------

  def _is_coordinator_event(self, event: Event, node_path: str) -> bool:
    """Returns True if the event is from this agent's coordinator scope.

    Accepts events with exact node_path match or from the
    coordinator's fixed internal nodes (call_llm, execute_tools).
    """
    if not isinstance(event, Event):
      return False
    if not event.node_info.path:
      return True
    if event.node_info.path == node_path:
      return True
    if not event.node_info.path.startswith(node_path):
      return False
    suffix = event.node_info.path[len(node_path) :]
    return suffix[:1] == '/' and suffix[1:] in _COORDINATOR_CHILD_NAMES

  def _maybe_save_output_to_state(self, event: Event, node_path: str) -> None:
    """Saves the model output to state and/or event.output if needed."""
    # Skip if the event was authored by some other agent (e.g. current
    # agent transferred to another agent).
    if event.author != self.name:
      logger.debug(
          'Skipping output save for agent %s: event authored by %s',
          self.name,
          event.author,
      )
      return

    # Verify the event originates from this agent's coordinator
    # scope (exact match or fixed internal nodes call_llm,
    # execute_tools).  Prevents same-named sub-agents from
    # leaking their output into this agent's output_key.
    if not self._is_coordinator_event(event, node_path):
      logger.debug(
          'Skipping output save for agent %s: event node_path %s'
          ' does not match expected path %s',
          self.name,
          event.node_info.path,
          node_path,
      )
      return

    # Skip interrupt events (long-running tools, HITL). These carry
    # function_call content that must be preserved for resume handling.
    # is_final_response() returns True for them, but they are not actual
    # model text responses.
    if event.long_running_tool_ids:
      return

    # Skip function_response events. These are tool execution results,
    # not model text responses. skip_summarization can make
    # is_final_response() return True for them (e.g. HITL confirmation
    # error responses), but their content must be preserved.
    if event.get_function_responses():
      return

    set_output_key = bool(self.output_key)
    set_event_output = self.mode == 'single_turn' or (self.mode == 'task' and self.output_schema)

    if not set_output_key and not set_event_output:
      return

    # Handle text responses.
    if event.is_final_response() and event.content and event.content.parts:
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
        result = validate_schema(self.output_schema, result)
      if set_output_key:
        event.actions.state_delta[self.output_key] = result
      if set_event_output:
        event.output = result
        if self.mode == 'single_turn':
          # Single_turn output goes only in event.output, not content.
          # Prevents redundant text from polluting the coordinator's
          # session history.
          event.content = None

  # ------------------------------------------------------------------
  # Config-based creation
  # ------------------------------------------------------------------

  @classmethod
  @experimental(FeatureName.AGENT_CONFIG)
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

      if isinstance(obj, (BaseTool, BaseToolset)):
        logger.debug(
            'Tool %s is an instance of BaseTool/BaseToolset.',
            tool_config.name,
        )
        resolved_tools.append(obj)
      elif inspect.isclass(obj) and (
          issubclass(obj, BaseTool) or issubclass(obj, BaseToolset)
      ):
        logger.debug(
            'Tool %s is a sub-class of BaseTool/BaseToolset.',
            tool_config.name,
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
              'Tool %s is a user-defined function tool.',
              tool_config.name,
          )
          resolved_tools.append(obj)
      else:
        raise ValueError(f'Invalid tool YAML config: {tool_config}.')

    return resolved_tools

  @override
  @classmethod
  @experimental(FeatureName.AGENT_CONFIG)
  def _parse_config(
      cls: Type[LlmAgent],
      config: LlmAgentConfig,
      config_abs_path: str,
      kwargs: Dict[str, Any],
  ) -> Dict[str, Any]:
    from .config_agent_utils import resolve_callbacks
    from .config_agent_utils import resolve_code_reference

    if config.model_code:
      kwargs['model'] = resolve_code_reference(config.model_code)
    elif config.model:
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

  # ------------------------------------------------------------------
  # Validators
  # ------------------------------------------------------------------

  @field_validator('generate_content_config', mode='after')
  @classmethod
  def validate_generate_content_config(
      cls, generate_content_config: Optional[types.GenerateContentConfig]
  ) -> types.GenerateContentConfig:
    if not generate_content_config:
      return types.GenerateContentConfig()
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

  # ------------------------------------------------------------------
  # Execution
  # ------------------------------------------------------------------

  @override
  async def run_async(
      self,
      parent_context: BaseInvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """Skips span/plugin wrapper — the coordinator handles those.

    ``LlmAgent`` is a transparent _Mesh wrapper. The internal
    ``_SingleLlmAgent`` coordinator goes through
    ``BaseAgent.run_async()`` which creates the ``invoke_agent``
    span and fires plugin callbacks. Doing it here too would
    double them.
    """
    ctx = self._create_invocation_context(parent_context)
    async with Aclosing(self._run_async_impl(ctx)) as agen:
      async for event in agen:
        yield event

  @override
  async def run_node_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Wraps ``_Mesh.run()`` to persist ``output_key`` state.

    Intercepts every event yielded by the mesh orchestration loop
    and calls ``_maybe_save_output_to_state`` with the context's
    ``node_path``.

    For single_turn mode, emits an agent-level output event (no path)
    after the loop so parent consumers (_Mesh, LlmAgentWrapper) can
    discover it without inspecting internal child paths.

    Also propagates ``escalate`` actions from the coordinator's
    internal nodes (call_llm, execute_tools) at the agent's own
    ``node_path`` so parent wrappers (e.g. LoopAgent's
    _DefaultRouteNode) can detect them.

    Args:
      ctx: The workflow context carrying ``node_path``.
      node_input: Input from the parent workflow graph or mesh.

    Yields:
      ADK events from the internal nodes.
    """
    escalated = False
    node_path = ctx.node_path
    single_turn_output = None
    async with Aclosing(
        super().run_node_impl(ctx=ctx, node_input=node_input)
    ) as run_gen:
      async for event in run_gen:
        output_before = event.output if isinstance(event, Event) else None
        self._maybe_save_output_to_state(event, node_path)
        # Capture output that _maybe_save_output_to_state just set.
        # Events with pre-existing output (e.g. CallLlmResult) are
        # skipped since their output is internal routing data.
        if (
            self.mode == 'single_turn'
            and isinstance(event, Event)
            and event.output is not None
            and output_before is None
        ):
          single_turn_output = event.output
        if (
            not escalated
            and isinstance(event, Event)
            and event.actions
            and event.actions.escalate
            and self._is_coordinator_event(event, node_path)
        ):
          escalated = True
        # Emit accumulated output before END_OF_AGENT so downstream
        # consumers see output in the correct order.
        if (
            isinstance(event, Event)
            and event.actions
            and event.actions.end_of_agent
        ):
          if single_turn_output is not None:
            yield Event(output=single_turn_output)
            single_turn_output = None
          if escalated:
            yield Event(
                actions=EventActions(escalate=True, skip_summarization=True),
            )
            escalated = False
        yield event
    # Fallback: emit if no END_OF_AGENT was produced (e.g. interrupted).
    if single_turn_output is not None:
      yield Event(output=single_turn_output)
    if escalated:
      yield Event(
          actions=EventActions(escalate=True, skip_summarization=True),
      )

  @override
  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[Any, None]:
    """Creates a root Context and delegates to ``run()``.

    Creates an initial ``Context`` wrapping the
    ``InvocationContext`` with no external transfer targets
    (this is the top-level entry point). ``run()`` then
    delegates to ``_Mesh.run()`` for orchestration and
    handles ``output_key`` persistence.

    Args:
      ctx: The current invocation context.

    Yields:
      ADK events from the internal nodes.
    """
    root_ctx = Context(
        invocation_context=ctx,
        node_path=self.name,
        execution_id='',
        local_events=[],
        transfer_targets=[],
    )
    async with Aclosing(self.run(ctx=root_ctx, node_input=None)) as agen:
      async for event in agen:
        yield event

  # ------------------------------------------------------------------
  # Validation
  # ------------------------------------------------------------------

  @model_validator(mode='after')
  def _validate_task_mode_no_sub_agents(self) -> Self:
    if self.mode in ('task', 'single_turn') and self.sub_agents:
      raise ValueError(
          f"Agent '{self.name}' has mode='{self.mode}' but also has"
          ' sub_agents. Task and single_turn agents cannot have'
          ' sub_agents.'
      )
    return self

  @model_validator(mode='after')
  def _validate_no_workflow_sub_agents(self) -> Self:
    from ..workflow._workflow import Workflow

    for sub_agent in self.sub_agents:
      if type(sub_agent) is Workflow:
        raise ValueError(
            f"LlmAgent '{self.name}' has a Workflow"
            f" ('{sub_agent.name}') as a sub_agent. Workflow cannot be"
            ' used as a sub_agent of LlmAgent. Use it as a node in'
            ' another Workflow instead.'
        )
    return self

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  @override
  def model_post_init(self, context: Any) -> None:
    """Build internal agents from sub_agents.

    Steps:
      1. Create coordinator ``_SingleLlmAgent`` for this agent's
         own reasoning loop (using all LLM fields from this
         instance).
      2. For each sub-agent, create or reuse an internal node:
         - Already a ``_SingleLlmAgent`` or ``LlmAgent`` -> reuse.
         - Has sub_agents -> create a nested ``LlmAgent``.
         - Leaf -> create a ``_SingleLlmAgent``.
      3. Populate ``nodes`` list.
      4. Call super to complete BaseAgent initialization.
    """
    # Provide a warning if multiple thinking configurations are found.
    if getattr(
        self.generate_content_config, 'thinking_config', None
    ) and getattr(self.planner, 'thinking_config', None):
      warnings.warn(
          'Both `thinking_config` in `generate_content_config` and a '
          'planner with `thinking_config` are provided. The '
          "planner's configuration will take precedence.",
          UserWarning,
          stacklevel=3,
      )

    # Allow shared sub-agent instances across multiple parents.
    # BaseAgent.model_post_init rejects sub-agents that already
    # have a parent_agent set.  The workflow LlmAgent uses Mesh
    # with node_path for scoping, so parent_agent is not required
    # for routing.  Reset it here so the validation passes; the
    # last parent to be constructed will own parent_agent.
    for sub_agent in self.sub_agents:
      sub_agent.parent_agent = None
    self._build_nodes()
    super().model_post_init(context)

  @override
  def model_copy(
      self, *, update: dict[str, Any] | None = None, deep: bool = False
  ) -> Any:
    """Overrides model_copy to rebuild internal nodes if name changes."""
    copied = super().model_copy(update=update, deep=deep)
    if update and 'name' in update:
      # If the agent name is updated (e.g., by ParallelWorker scheduling it
      # dynamically), rebuild the internal nodes so the coordinator gets the
      # updated name.
      copied._build_nodes()
    return copied

  def _build_nodes(self) -> None:
    """Construct the ``nodes`` list from ``sub_agents``.

    Creates the coordinator ``_SingleLlmAgent`` and one internal
    node per sub-agent (either ``_SingleLlmAgent`` for leaves or
    nested ``LlmAgent`` for sub-agents with their own sub-agents).
    """
    # 1. Auto-inject task tools BEFORE creating the coordinator so
    #    it gets the updated self.tools snapshot.
    self._inject_task_tools()

    # 2. Create coordinator _SingleLlmAgent for this agent's own
    #    reasoning loop. Uses all LLM fields from self.
    coordinator = _SingleLlmAgent.from_base_llm_agent(self)

    # 3. Process sub_agents into internal nodes.
    internal_nodes: list[BaseNode] = []
    for sub_agent in self.sub_agents:
      if isinstance(sub_agent, (_SingleLlmAgent, LlmAgent)):
        internal_nodes.append(sub_agent)
      elif isinstance(sub_agent, BaseLlmAgent):
        if sub_agent.sub_agents:
          internal_nodes.append(sub_agent)
        else:
          internal_nodes.append(_SingleLlmAgent.from_base_llm_agent(sub_agent))
      elif isinstance(sub_agent, BaseNode):
        internal_nodes.append(sub_agent)
      else:
        logger.warning(
            'Sub-agent %s is not a BaseNode, skipping.',
            getattr(sub_agent, 'name', sub_agent),
        )

    # 4. Populate nodes: coordinator first, then sub-agent nodes.
    self.nodes = [coordinator] + internal_nodes

  def _inject_task_tools(self) -> None:
    """Auto-inject FinishTaskTool and RequestTaskTool based on modes.

    - If this agent has ``mode == 'task'``: inject ``FinishTaskTool``
      into ``self.tools`` (unless already present). Single_turn agents
      use controlled output (response_schema) instead of finish_task.
    - For each sub-agent with ``mode in ('task', 'single_turn')``:
      inject ``RequestTaskTool`` for that sub-agent into ``self.tools``
      (unless a tool named ``{sub.name}`` already exists).
    - For each sub-agent with ``mode == 'task'``: inject
      ``FinishTaskTool`` on the sub-agent itself.
    """
    existing_tool_names = {
        t.name for t in self.tools if isinstance(t, BaseTool)
    }

    # Inject FinishTaskTool for task agents only. Single_turn agents
    # use controlled output (response_schema) instead.
    if self.mode == 'task' and FINISH_TASK_TOOL_NAME not in existing_tool_names:
      self.tools.append(FinishTaskTool(task_agent=self))

    # Inject RequestTaskTool for each task/single_turn sub-agent,
    # and FinishTaskTool on task sub-agents only.
    for sub_agent in self.sub_agents:
      if not isinstance(sub_agent, BaseLlmAgent):
        continue
      if sub_agent.mode not in ('task', 'single_turn'):
        continue
      tool_name = sub_agent.name
      if tool_name not in existing_tool_names:
        self.tools.append(RequestTaskTool(task_agent=sub_agent))
      if sub_agent.mode == 'task':
        sub_tool_names = {
            t.name for t in sub_agent.tools if isinstance(t, BaseTool)
        }
        if FINISH_TASK_TOOL_NAME not in sub_tool_names:
          sub_agent.tools.append(FinishTaskTool(task_agent=sub_agent))


Agent: TypeAlias = LlmAgent
