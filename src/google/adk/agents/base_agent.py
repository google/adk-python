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

import inspect
from typing import Any
from typing import AsyncGenerator
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import final
from typing import Mapping
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from typing_extensions import override
from typing_extensions import TypeAlias

from ..events.event import Event
from ..events.event_actions import EventActions
from ..telemetry import tracing
from ..telemetry.tracing import tracer
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental
from .base_agent_config import BaseAgentConfig
from .callback_context import CallbackContext

if TYPE_CHECKING:
  from .invocation_context import InvocationContext

_SingleAgentCallback: TypeAlias = Callable[
    [CallbackContext],
    Union[Awaitable[Optional[types.Content]], Optional[types.Content]],
]

BeforeAgentCallback: TypeAlias = Union[
    _SingleAgentCallback,
    list[_SingleAgentCallback],
]

AfterAgentCallback: TypeAlias = Union[
    _SingleAgentCallback,
    list[_SingleAgentCallback],
]

SelfAgent = TypeVar('SelfAgent', bound='BaseAgent')


@experimental
class BaseAgentState(BaseModel):
  """Base class for all agent states."""

  model_config = ConfigDict(
      extra='forbid',
  )


AgentState = TypeVar('AgentState', bound=BaseAgentState)


class BaseAgent(BaseModel):
  """Base class for all agents in Agent Development Kit."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  config_type: ClassVar[type[BaseAgentConfig]] = BaseAgentConfig
  """The config type for this agent.

  Sub-classes should override this to specify their own config type.

  Example:

  ```
  class MyAgentConfig(BaseAgentConfig):
    my_field: str = ''

  class MyAgent(BaseAgent):
    config_type: ClassVar[type[BaseAgentConfig]] = MyAgentConfig
  ```
  """

  name: str
  """The agent's name.

  Agent name must be a Python identifier and unique within the agent tree.
  Agent name cannot be "user", since it's reserved for end-user's input.
  """

  description: str = ''
  """Description about the agent's capability.

  The model uses this to determine whether to delegate control to the agent.
  One-line description is enough and preferred.
  """

  parent_agent: Optional[BaseAgent] = Field(default=None, init=False)
  """The parent agent of this agent.

  Note that an agent can ONLY be added as sub-agent once.

  If you want to add one agent twice as sub-agent, consider to create two agent
  instances with identical config, but with different name and add them to the
  agent tree.
  """
  sub_agents: list[BaseAgent] = Field(default_factory=list)
  """The sub-agents of this agent."""

  before_agent_callback: Optional[BeforeAgentCallback] = None
  """Callback or list of callbacks to be invoked before the agent run.

  When a list of callbacks is provided, the callbacks will be called in the
  order they are listed until a callback does not return None.

  Args:
    callback_context: MUST be named 'callback_context' (enforced).

  Returns:
    Optional[types.Content]: The content to return to the user.
      When the content is present, the agent run will be skipped and the
      provided content will be returned to user.
  """
  after_agent_callback: Optional[AfterAgentCallback] = None
  """Callback or list of callbacks to be invoked after the agent run.

  When a list of callbacks is provided, the callbacks will be called in the
  order they are listed until a callback does not return None.

  Args:
    callback_context: MUST be named 'callback_context' (enforced).

  Returns:
    Optional[types.Content]: The content to return to the user.
      When the content is present, the provided content will be used as agent
      response and appended to event history as agent response.
  """

  def _load_agent_state(
      self,
      ctx: InvocationContext,
      state_type: Type[AgentState],
      default_state: AgentState,
  ) -> tuple[AgentState, bool]:
    """Loads the agent state from the invocation context, handling resumption.

    Args:
      ctx: The invocation context.
      state_type: The type of the agent state.
      default_state: The default state to use if not resuming.

    Returns:
        tuple[AgentState, bool]: The current state and a boolean indicating if
        resuming.
    """
    if self.name not in ctx.agent_states:
      return default_state, False
    else:
      return state_type.model_validate(ctx.agent_states.get(self.name)), True

  def _create_agent_state_event(
      self,
      ctx: InvocationContext,
      *,
      state: Optional[BaseAgentState] = None,
      end_of_agent: bool = False,
  ) -> Event:
    """Creates an event for agent state.

    Args:
      ctx: The invocation context.
      state: The agent state to checkpoint.
      end_of_agent: Whether the agent is finished running.

    Returns:
      An Event object representing the checkpoint.
    """
    event_actions = EventActions()
    if state:
      event_actions.agent_state = state.model_dump(mode='json')
    if end_of_agent:
      event_actions.end_of_agent = True
    return Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        branch=ctx.branch,
        actions=event_actions,
    )

  def clone(
      self: SelfAgent, update: Mapping[str, Any] | None = None
  ) -> SelfAgent:
    """Creates a copy of this agent instance.

    Args:
      update: Optional mapping of new values for the fields of the cloned agent.
        The keys of the mapping are the names of the fields to be updated, and
        the values are the new values for those fields.
        For example: {"name": "cloned_agent"}

    Returns:
      A new agent instance with identical configuration as the original
      agent except for the fields specified in the update.
    """
    if update is not None and 'parent_agent' in update:
      raise ValueError(
          'Cannot update `parent_agent` field in clone. Parent agent is set'
          ' only when the parent agent is instantiated with the sub-agents.'
      )

    # Only allow updating fields that are defined in the agent class.
    allowed_fields = set(self.__class__.model_fields)
    if update is not None:
      invalid_fields = set(update) - allowed_fields
      if invalid_fields:
        raise ValueError(
            f'Cannot update non-existent fields in {self.__class__.__name__}:'
            f' {invalid_fields}'
        )

    cloned_agent = self.model_copy(update=update)

    # If any field is stored as list and not provided in the update, need to
    # shallow copy it for the cloned agent to avoid sharing the same list object
    # with the original agent.
    for field_name in cloned_agent.__class__.model_fields:
      if field_name == 'sub_agents':
        continue
      if update is not None and field_name in update:
        continue
      field = getattr(cloned_agent, field_name)
      if isinstance(field, list):
        setattr(cloned_agent, field_name, field.copy())

    if update is None or 'sub_agents' not in update:
      # If `sub_agents` is not provided in the update, need to recursively clone
      # the sub-agents to avoid sharing the sub-agents with the original agent.
      cloned_agent.sub_agents = []
      for sub_agent in self.sub_agents:
        cloned_sub_agent = sub_agent.clone()
        cloned_sub_agent.parent_agent = cloned_agent
        cloned_agent.sub_agents.append(cloned_sub_agent)
    else:
      for sub_agent in cloned_agent.sub_agents:
        sub_agent.parent_agent = cloned_agent

    # Remove the parent agent from the cloned agent to avoid sharing the parent
    # agent with the cloned agent.
    cloned_agent.parent_agent = None
    return cloned_agent

  @final
  async def run_async(
      self,
      parent_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """Entry method to run an agent via text-based conversation.

    Args:
      parent_context: InvocationContext, the invocation context of the parent
        agent.

    Yields:
      Event: the events generated by the agent.
    """

    async def _run_with_trace() -> AsyncGenerator[Event, None]:
      with tracer.start_as_current_span(f'invoke_agent {self.name}') as span:
        ctx = self._create_invocation_context(parent_context)
        tracing.trace_agent_invocation(span, self, ctx)
        if event := await self.__handle_before_agent_callback(ctx):
          yield event
        if ctx.end_invocation:
          return

        async with Aclosing(self._run_async_impl(ctx)) as agen:
          async for event in agen:
            yield event

        if ctx.end_invocation:
          return

        if event := await self.__handle_after_agent_callback(ctx):
          yield event

    async with Aclosing(_run_with_trace()) as agen:
      async for event in agen:
        yield event

  @final
  async def run_live(
      self,
      parent_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """Entry method to run an agent via video/audio-based conversation.

    Args:
      parent_context: InvocationContext, the invocation context of the parent
        agent.

    Yields:
      Event: the events generated by the agent.
    """

    async def _run_with_trace() -> AsyncGenerator[Event, None]:
      with tracer.start_as_current_span(f'invoke_agent {self.name}') as span:
        ctx = self._create_invocation_context(parent_context)
        tracing.trace_agent_invocation(span, self, ctx)
        if event := await self.__handle_before_agent_callback(ctx):
          yield event
        if ctx.end_invocation:
          return

        async with Aclosing(self._run_live_impl(ctx)) as agen:
          async for event in agen:
            yield event

        if event := await self.__handle_after_agent_callback(ctx):
          yield event

    async with Aclosing(_run_with_trace()) as agen:
      async for event in agen:
        yield event

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Core logic to run this agent via text-based conversation.

    Args:
      ctx: InvocationContext, the invocation context for this agent.

    Yields:
      Event: the events generated by the agent.
    """
    raise NotImplementedError(
        f'_run_async_impl for {type(self)} is not implemented.'
    )
    yield  # AsyncGenerator requires having at least one yield statement

  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Core logic to run this agent via video/audio-based conversation.

    Args:
      ctx: InvocationContext, the invocation context for this agent.

    Yields:
      Event: the events generated by the agent.
    """
    raise NotImplementedError(
        f'_run_live_impl for {type(self)} is not implemented.'
    )
    yield  # AsyncGenerator requires having at least one yield statement

  @property
  def root_agent(self) -> BaseAgent:
    """Gets the root agent of this agent."""
    root_agent = self
    while root_agent.parent_agent is not None:
      root_agent = root_agent.parent_agent
    return root_agent

  def find_agent(self, name: str) -> Optional[BaseAgent]:
    """Finds the agent with the given name in this agent and its descendants.

    Args:
      name: The name of the agent to find.

    Returns:
      The agent with the matching name, or None if no such agent is found.
    """
    if self.name == name:
      return self
    return self.find_sub_agent(name)

  def find_sub_agent(self, name: str) -> Optional[BaseAgent]:
    """Finds the agent with the given name in this agent's descendants.

    Args:
      name: The name of the agent to find.

    Returns:
      The agent with the matching name, or None if no such agent is found.
    """
    for sub_agent in self.sub_agents:
      if result := sub_agent.find_agent(name):
        return result
    return None

  def _create_invocation_context(
      self, parent_context: InvocationContext
  ) -> InvocationContext:
    """Creates a new invocation context for this agent."""
    invocation_context = parent_context.model_copy(update={'agent': self})
    return invocation_context

  @property
  def canonical_before_agent_callbacks(self) -> list[_SingleAgentCallback]:
    """The resolved self.before_agent_callback field as a list of _SingleAgentCallback.

    This method is only for use by Agent Development Kit.
    """
    if not self.before_agent_callback:
      return []
    if isinstance(self.before_agent_callback, list):
      return self.before_agent_callback
    return [self.before_agent_callback]

  @property
  def canonical_after_agent_callbacks(self) -> list[_SingleAgentCallback]:
    """The resolved self.after_agent_callback field as a list of _SingleAgentCallback.

    This method is only for use by Agent Development Kit.
    """
    if not self.after_agent_callback:
      return []
    if isinstance(self.after_agent_callback, list):
      return self.after_agent_callback
    return [self.after_agent_callback]

  async def __handle_before_agent_callback(
      self, ctx: InvocationContext
  ) -> Optional[Event]:
    """Runs the before_agent_callback if it exists.

    Args:
      ctx: InvocationContext, the invocation context for this agent.

    Returns:
      Optional[Event]: an event if callback provides content or changed state.
    """
    callback_context = CallbackContext(ctx)

    # Run callbacks from the plugins.
    before_agent_callback_content = (
        await ctx.plugin_manager.run_before_agent_callback(
            agent=self, callback_context=callback_context
        )
    )

    # If no overrides are provided from the plugins, further run the canonical
    # callbacks.
    if (
        not before_agent_callback_content
        and self.canonical_before_agent_callbacks
    ):
      for callback in self.canonical_before_agent_callbacks:
        before_agent_callback_content = callback(
            callback_context=callback_context
        )
        if inspect.isawaitable(before_agent_callback_content):
          before_agent_callback_content = await before_agent_callback_content
        if before_agent_callback_content:
          break

    # Process the override content if exists, and further process the state
    # change if exists.
    if before_agent_callback_content:
      ret_event = Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          branch=ctx.branch,
          content=before_agent_callback_content,
          actions=callback_context._event_actions,
      )
      ctx.end_invocation = True
      return ret_event

    if callback_context.state.has_delta():
      return Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          branch=ctx.branch,
          actions=callback_context._event_actions,
      )

    return None

  async def __handle_after_agent_callback(
      self, invocation_context: InvocationContext
  ) -> Optional[Event]:
    """Runs the after_agent_callback if it exists.

    Args:
      invocation_context: InvocationContext, the invocation context for this
        agent.

    Returns:
      Optional[Event]: an event if callback provides content or changed state.
    """

    callback_context = CallbackContext(invocation_context)

    # Run callbacks from the plugins.
    after_agent_callback_content = (
        await invocation_context.plugin_manager.run_after_agent_callback(
            agent=self, callback_context=callback_context
        )
    )

    # If no overrides are provided from the plugins, further run the canonical
    # callbacks.
    if (
        not after_agent_callback_content
        and self.canonical_after_agent_callbacks
    ):
      for callback in self.canonical_after_agent_callbacks:
        after_agent_callback_content = callback(
            callback_context=callback_context
        )
        if inspect.isawaitable(after_agent_callback_content):
          after_agent_callback_content = await after_agent_callback_content
        if after_agent_callback_content:
          break

    # Process the override content if exists, and further process the state
    # change if exists.
    if after_agent_callback_content:
      ret_event = Event(
          invocation_id=invocation_context.invocation_id,
          author=self.name,
          branch=invocation_context.branch,
          content=after_agent_callback_content,
          actions=callback_context._event_actions,
      )
      return ret_event

    if callback_context.state.has_delta():
      return Event(
          invocation_id=invocation_context.invocation_id,
          author=self.name,
          branch=invocation_context.branch,
          content=after_agent_callback_content,
          actions=callback_context._event_actions,
      )
    return None

  @override
  def model_post_init(self, __context: Any) -> None:
    self.__set_parent_agent_for_sub_agents()

  @field_validator('name', mode='after')
  @classmethod
  def validate_name(cls, value: str):
    if not value.isidentifier():
      raise ValueError(
          f'Found invalid agent name: `{value}`.'
          ' Agent name must be a valid identifier. It should start with a'
          ' letter (a-z, A-Z) or an underscore (_), and can only contain'
          ' letters, digits (0-9), and underscores.'
      )
    if value == 'user':
      raise ValueError(
          "Agent name cannot be `user`. `user` is reserved for end-user's"
          ' input.'
      )
    return value

  def __set_parent_agent_for_sub_agents(self) -> BaseAgent:
    for sub_agent in self.sub_agents:
      if sub_agent.parent_agent is not None:
        raise ValueError(
            f'Agent `{sub_agent.name}` already has a parent agent, current'
            f' parent: `{sub_agent.parent_agent.name}`, trying to add:'
            f' `{self.name}`'
        )
      sub_agent.parent_agent = self
    return self

  @final
  @classmethod
  @experimental
  def from_config(
      cls: Type[SelfAgent],
      config: BaseAgentConfig,
      config_abs_path: str,
  ) -> SelfAgent:
    """Creates an agent from a config.

    If sub-classes uses a custom agent config, override `_from_config_kwargs`
    method to return an updated kwargs for agent construstor.

    Args:
      config: The config to create the agent from.
      config_abs_path: The absolute path to the config file that contains the
        agent config.

    Returns:
      The created agent.
    """
    kwargs = cls.__create_kwargs(config, config_abs_path)
    kwargs = cls._parse_config(config, config_abs_path, kwargs)
    return cls(**kwargs)

  @classmethod
  @experimental
  def _parse_config(
      cls: Type[SelfAgent],
      config: BaseAgentConfig,
      config_abs_path: str,
      kwargs: Dict[str, Any],
  ) -> Dict[str, Any]:
    """Parses the config and returns updated kwargs to construct the agent.

    Sub-classes should override this method to use a custom agent config class.

    Args:
      config: The config to parse.
      config_abs_path: The absolute path to the config file that contains the
        agent config.
      kwargs: The keyword arguments used for agent constructor.

    Returns:
      The updated keyword arguments used for agent constructor.
    """
    return kwargs

  @classmethod
  def __create_kwargs(
      cls,
      config: BaseAgentConfig,
      config_abs_path: str,
  ) -> Dict[str, Any]:
    """Creates kwargs for the fields of BaseAgent."""

    from .config_agent_utils import resolve_agent_reference
    from .config_agent_utils import resolve_callbacks

    kwargs: Dict[str, Any] = {
        'name': config.name,
        'description': config.description,
    }
    if config.sub_agents:
      sub_agents = []
      for sub_agent_config in config.sub_agents:
        sub_agent = resolve_agent_reference(sub_agent_config, config_abs_path)
        sub_agents.append(sub_agent)
      kwargs['sub_agents'] = sub_agents

    if config.before_agent_callbacks:
      kwargs['before_agent_callback'] = resolve_callbacks(
          config.before_agent_callbacks
      )
    if config.after_agent_callbacks:
      kwargs['after_agent_callback'] = resolve_callbacks(
          config.after_agent_callbacks
      )
    return kwargs
