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

"""Runs an Antigravity SDK agent as an ADK agent.

Wraps a pre-configured ``google.antigravity.Agent`` as a native ADK
``BaseAgent`` node, delegating each turn to the Antigravity SDK runner and
streaming the harness's steps back as ADK events.

The harness runs the Antigravity agent's loop and owns its conversation, so an
``AntigravityAgent`` must run as an ADK root agent unless it declares
``mode='single_turn'``. ADK ``sub_agents`` are allowed: each child is bridged
onto the Antigravity SDK config as a client-side tool, which is the only way
the harness can reach one.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import types
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Callable
from typing import Literal
from typing import Protocol

from google.antigravity import Agent
from google.antigravity import AgentConfig
from google.antigravity.connections.local.local_connection_config import BaseLocalAgentConfig
from google.antigravity.types import SessionContinuationMode
from google.antigravity.types import Step
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ...agents.base_agent import BaseAgent
from ...agents.context import Context
from ...agents.invocation_context import InvocationContext
from ...agents.run_config import StreamingMode
from ...events.event import Event
from ...events.event_actions import EventActions
from ...utils.content_utils import to_user_content
from ._event_converter import convert_step_to_events
from ._event_converter import drain_tool_results
from ._event_converter import final_model_text
from ._sub_agent_tools import make_sub_agent_tool
from ._tool_result_capture import ToolErrorCapture
from ._tool_result_capture import ToolResultBuffer
from ._tool_result_capture import ToolResultCapture

logger = logging.getLogger('google_adk.' + __name__)

_CONVERSATION_ID_STATE_KEY_PREFIX = '_antigravity_conversation_id_'

_PARENT_REQUIRES_SINGLE_TURN_MESSAGE = (
    'AntigravityAgent may only be an ADK sub-agent when it sets '
    "mode='single_turn', where the ADK parent composes a self-contained "
    'request. Otherwise it must run as an ADK root agent.'
)


class _SdkConversation(Protocol):
  """The parts of an Antigravity SDK ``Conversation`` that a turn drives."""

  @property
  def history(self) -> list[Step]:
    ...

  async def send(self, prompt: str) -> None:
    ...

  def receive_steps(self) -> AsyncIterator[Step]:
    ...


class _SdkAgent(Protocol):
  """The parts of an Antigravity SDK ``Agent`` that a turn runs on.

  A protocol so that a subclass can run its turns on a structurally identical
  copy of ``google.antigravity.Agent``.
  """

  @property
  def conversation(self) -> _SdkConversation:
    ...

  @property
  def conversation_id(self) -> str | None:
    ...

  async def __aenter__(self) -> _SdkAgent:
    ...

  async def __aexit__(
      self,
      exc_type: type[BaseException] | None,
      exc: BaseException | None,
      traceback: types.TracebackType | None,
  ) -> bool | None:
    ...


@dataclasses.dataclass(frozen=True)
class _ActiveConversation:
  """An entered Antigravity SDK ``Agent`` and its scoped tool-result capture."""

  agent: _SdkAgent
  tool_results: ToolResultBuffer | None

  async def __aenter__(self) -> _ActiveConversation:
    # The SDK ``Agent`` is already entered by ``_enter_sdk_agent``; this exists
    # only so callers can use ``async with``, which reports the true unwinding
    # exception to ``__aexit__`` rather than ``sys.exc_info()``.
    return self

  async def __aexit__(
      self,
      exc_type: type[BaseException] | None,
      exc: BaseException | None,
      traceback: types.TracebackType | None,
  ) -> bool | None:
    return await self.agent.__aexit__(exc_type, exc, traceback)


class AntigravityAgent(BaseAgent):
  """Runs a Google Antigravity SDK agent as an ADK agent node.

  Each turn of an ADK session runs on a fresh Antigravity SDK ``Agent``,
  resuming the conversation the previous turn created. The conversation id is
  kept in ADK session state, so resumption survives a restart; under
  ``mode='single_turn'`` no id is stored. Persisting the id needs the ADK
  ``Runner``, which is what applies a yielded event's ``state_delta``.

  Any ADK ``sub_agents`` are bridged onto the Antigravity SDK config as
  client-side tools named after the child, so every child needs a non-empty
  ``description`` and a name unique among its siblings.

  Must be an ADK root agent unless ``mode='single_turn'``.
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      use_attribute_docstrings=True,
      extra='forbid',
  )

  config: AgentConfig = Field(exclude=True)
  """The ``google.antigravity.AgentConfig`` describing the Antigravity agent.

  Typically a ``LocalAgentConfig``. Excluded from serialization: it holds
  runtime wiring (e.g. callable tools) that is not JSON-serializable.
  """

  mode: Literal['single_turn'] | None = Field(default=None, frozen=True)
  """Composition mode when used as a sub-agent.

  ``'single_turn'`` is what allows this agent to have a parent at all: the
  parent ``LlmAgent`` exposes it as an inline tool taking a ``request`` string.
  The parent composes the task; session history is not forwarded, and each call
  is an independent conversation.

  Leave as ``None`` for a standalone root agent. Frozen, because the adoption
  guard only gets to check it once, at construction.
  """

  @override
  def model_post_init(self, __context: Any) -> None:
    super().model_post_init(__context)
    self._validate_sub_agents()
    self._warn_if_local_without_save_dir()

  def _warn_if_local_without_save_dir(self) -> None:
    if self.mode == 'single_turn':
      return
    # A local config with no `save_dir` mints a fresh temporary directory per
    # connection, so every turn writes somewhere the next turn will not look.
    if not isinstance(self.config, self._local_config_cls):
      return
    if self.config.save_dir:
      return
    logger.warning(
        'This Antigravity agent will not remember anything across turns: its'
        ' config runs the harness locally with no save_dir, so each turn gets'
        ' a fresh'
        ' temporary directory, and the conversation from the previous turn is'
        ' not there to resume. Set save_dir to a stable path, or set'
        ' mode="single_turn" if independent turns are what you want.'
    )

  def _validate_sub_agents(self) -> None:
    # Called again from `_build_sdk_config` because `sub_agents` can be mutated
    # or `model_copy`-ed after construction, bypassing `model_post_init`.
    # Seeded with the config's own tool names: a child is added to the same
    # `config.tools`, so one sharing a name with a tool already there collides
    # just as two children do. A `str` entry names a builtin; a callable carries
    # its name on `__name__`. Builtins enabled by name are not enumerated here.
    tool_names: set[str] = {
        tool if isinstance(tool, str) else getattr(tool, '__name__', '')
        for tool in self.config.tools
    }
    seen_names: set[str] = set()
    for child in self.sub_agents:
      if not child.description:
        raise ValueError(
            f"ADK sub-agent '{child.name}' needs a description: it is offered"
            ' to the harness as a tool, and the description is the only thing'
            ' the Antigravity model reads when deciding whether to call it.'
        )
      if child.name in tool_names:
        raise ValueError(
            f"ADK sub-agent '{child.name}' collides with a tool already on the"
            ' config: it is added to the same `config.tools`, and the harness'
            ' registers one tool per name and rejects the second with an error'
            ' naming only the tool. Rename the child or the tool.'
        )
      if child.name in seen_names:
        # BaseAgent.validate_sub_agents_unique_names only logs a warning.
        raise ValueError(
            f"Two ADK sub-agents share the name '{child.name}': the harness"
            ' registers one tool per name and rejects the second with an error'
            ' naming only the tool, not the ADK agent it came from. Rename one'
            ' of the children.'
        )
      seen_names.add(child.name)

  def __setattr__(self, name: str, value: Any) -> None:
    # `mode` is read via __dict__ because fields may still be unpopulated.
    if (
        name == 'parent_agent'
        and value is not None
        and self.__dict__.get('mode') != 'single_turn'
    ):
      raise ValueError(_PARENT_REQUIRES_SINGLE_TURN_MESSAGE)
    super().__setattr__(name, value)

  def _extract_user_prompt(self, ctx: InvocationContext) -> str:
    if ctx.user_content and ctx.user_content.parts:
      for part in ctx.user_content.parts:
        if part.text:
          return str(part.text)
    return ''

  @property
  def _sdk_agent_cls(self) -> Callable[[AgentConfig], _SdkAgent]:
    """The Antigravity SDK ``Agent`` class each turn runs on.

    Override to bind a different copy of the Antigravity SDK.
    """
    return Agent  # type: ignore[no-any-return]

  @property
  def _local_config_cls(self) -> type[AgentConfig]:
    """The config class meaning "runs the harness as a local subprocess"."""
    # The **base** local config, not the default subclass: the temporary
    # directory is minted by `BaseLocalAgentConfig._get_or_create_save_dir`, so
    # every subclass of it forgets without a `save_dir`, not just the usual one.
    return BaseLocalAgentConfig  # type: ignore[no-any-return]

  @property
  def _tool_result_capture_cls(self) -> type[ToolResultBuffer]:
    # A seam so another Antigravity SDK copy can supply its own capture: a hook
    # is classified by `isinstance` against the hook classes of the SDK it came
    # from, so the caller cannot use `ToolResultCapture` directly -- it is bound
    # to this copy's `PostToolCallHook`. Typed as the non-hook base
    # `ToolResultBuffer`, not this subclass, so an override can subclass that
    # base plus its own hook rather than dragging this copy's hook base in.
    return ToolResultCapture

  @property
  def _tool_error_capture_cls(self) -> type[ToolErrorCapture]:
    # The same seam, overridden for the same reason: each SDK copy binds its own
    # `OnToolErrorHook`. Separate from the success capture because both hook
    # interfaces name their entry point `run`. No buffer-only base to widen to
    # here -- this holds a buffer rather than being one -- so the type stays the
    # concrete class.
    return ToolErrorCapture

  def _build_sdk_config(
      self,
      tool_results: ToolResultBuffer | None = None,
  ) -> AgentConfig:
    self._validate_sub_agents()
    # Copied because the Antigravity SDK `Agent`'s AsyncExitStack is
    # single-use, and to avoid mutating the caller's config.
    config = self.config.model_copy(deep=True)
    if self.sub_agents:
      config.tools = list(config.tools) + [
          make_sub_agent_tool(child) for child in self.sub_agents
      ]
    if tool_results is not None:
      # Both halves: success reports on `post_tool_call`, failure on
      # `on_tool_error`, and registering the error hook is what puts
      # `LIFECYCLE_HOOK_ON_TOOL_ERROR` on the wire at all.
      # Runtime-safe: both values are real hooks. `tool_results` is typed as
      # the widened non-hook seam base `ToolResultBuffer` (see
      # `_tool_result_capture_cls`), so pyrefly cannot see it is a `Hook`.
      config.hooks = list(config.hooks) + [  # pyrefly: ignore[bad-assignment]
          tool_results,
          self._tool_error_capture_cls(tool_results),
      ]
    return config

  def _conversation_id_state_key(self) -> str:
    # Scoped by agent name so two `AntigravityAgent`s in one ADK session do not
    # resume each other's conversation.
    return _CONVERSATION_ID_STATE_KEY_PREFIX + self.name

  def _conversation_id_event(
      self, ctx: InvocationContext, conversation_id: str | None
  ) -> Event:
    # Its own event rather than folded into a model event, because a partial
    # event is not appended to the session -- which is where `state_delta` is
    # applied. A None `conversation_id` clears the stored id.
    state_delta: dict[str, str | None] = {
        self._conversation_id_state_key(): conversation_id
    }
    return Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        branch=ctx.branch,
        actions=EventActions(state_delta=state_delta),
    )

  def _id_delta_if_changed(
      self,
      ctx: InvocationContext,
      active: _ActiveConversation,
      stored_id: str | None,
  ) -> Event | None:
    """Returns an event persisting the conversation id, if it is new."""
    # Asked per event rather than at connect, because the runtime need not have
    # published an id yet by then, nor by the first step.
    conversation_id = active.agent.conversation_id
    if not conversation_id or conversation_id == stored_id:
      return None
    return self._conversation_id_event(ctx, conversation_id)

  async def _enter_sdk_agent(
      self, conversation_id: str | None = None
  ) -> _ActiveConversation:
    # Gated on `sub_agents`: ADK children are the only client tools, and a
    # post_tool_call hook costs a blocking round trip per successful call. Its
    # call ids only mean anything within this conversation.
    tool_results = self._tool_result_capture_cls() if self.sub_agents else None
    config = self._build_sdk_config(tool_results)
    if conversation_id:
      config.conversation_id = conversation_id
      # CREATE_OR_RESUME is the only mode that survives a store that is no
      # longer there; under the default a missing store is a hard error raised
      # from inside the runtime.
      config.session_continuation_mode = (
          SessionContinuationMode.CREATE_OR_RESUME
      )
    agent = self._sdk_agent_cls(config)
    try:
      entered = await agent.__aenter__()
    except BaseException:
      # BaseException is right *here*: this arm awaits, it does not yield, and
      # a cancelled connect would otherwise orphan the harness subprocess.
      await agent.__aexit__(*sys.exc_info())
      raise
    return _ActiveConversation(entered, tool_results)

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if self.mode == 'single_turn':
      active = await self._enter_sdk_agent()
      async with active:
        async for event in self._run_turn(active, ctx):
          yield event
      return

    stored_id: str | None = ctx.session.state.get(
        self._conversation_id_state_key()
    )
    active = await self._enter_sdk_agent(stored_id)

    async with active:
      if stored_id and self._resume_was_silently_dropped(active):
        yield self._conversation_id_event(ctx, None)
        raise RuntimeError(
            f'Could not resume conversation {stored_id!r}: it is no longer'
            ' available. The stored id has been cleared, so the next turn will'
            ' start a new conversation, but the earlier turns of this session'
            ' are not recoverable.'
        )
      # Record the id from inside the loop as soon as it changes. "Did we yield
      # an event" is not the same question as "does this conversation have
      # history": a turn whose steps carry no user-visible content (e.g.
      # compaction) yields no events yet still has history, and must record its
      # id or the next turn orphans it. A genuinely empty turn (no history) must
      # NOT record, or the next resume looks silently dropped.
      recorded = False
      try:
        async for event in self._run_turn(active, ctx):
          yield event
          if not recorded:
            delta = self._id_delta_if_changed(ctx, active, stored_id)
            if delta is not None:
              recorded = True
              yield delta
        if not recorded and active.agent.conversation.history:
          delta = self._id_delta_if_changed(ctx, active, stored_id)
          if delta is not None:
            yield delta
      except Exception:
        # A harness error mid-turn leaves a resumable conversation behind, so
        # the block after the loop never runs on that path; persist the id here
        # before re-raising, or the next turn orphans it. Only Exception, never
        # GeneratorExit: answering an abandoned consumer with a yield would
        # raise "async generator ignored GeneratorExit".
        if not recorded and active.agent.conversation.history:
          delta = self._id_delta_if_changed(ctx, active, stored_id)
          if delta is not None:
            yield delta
        raise

  def _resume_was_silently_dropped(self, active: _ActiveConversation) -> bool:
    """Whether we asked to resume and got a new conversation instead."""
    # An empty history after a resume is the only silent-drop signal we have,
    # and it is only reliable for the local connection: verified there, it
    # creates a fresh conversation when the stored one is missing and reports
    # success, seeding no prior history. A remote backend that quietly starts a
    # new conversation on a stale id (rather than raising) cannot be
    # distinguished here without Antigravity SDK support, so this is gated to
    # the local config to avoid falsely failing every remote resume.
    if not isinstance(self.config, self._local_config_cls):
      return False
    return not active.agent.conversation.history

  async def _run_turn(
      self, active: _ActiveConversation, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    seen_tool_calls: set[str] = set()
    seen_tool_results: set[str] = set()
    streaming = bool(
        ctx.run_config and ctx.run_config.streaming_mode == StreamingMode.SSE
    )

    await active.agent.conversation.send(self._extract_user_prompt(ctx))

    async for step in active.agent.conversation.receive_steps():
      for event in convert_step_to_events(
          step,
          ctx=ctx,
          author=self.name,
          seen_tool_calls=seen_tool_calls,
          seen_tool_results=seen_tool_results,
          tool_results=active.tool_results,
          streaming=streaming,
      ):
        yield event

    # A client tool's terminal step carries empty `tool_calls`: its result
    # arrives on the `post_tool_call` hook, not in a step, so pair it up after
    # the loop. The hook is a blocking round trip the harness completes before
    # the turn goes idle, so every result owed a response is buffered by now.
    for event in drain_tool_results(
        ctx=ctx,
        seen_tool_calls=seen_tool_calls,
        seen_tool_results=seen_tool_results,
        tool_results=active.tool_results,
    ):
      yield event

    if active.tool_results is not None:
      # Whatever is left was never owed a response.
      active.tool_results.clear()

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Event, None]:
    """Runs the agent as a node, threading node_input in and output out."""
    parent_context = ctx.get_invocation_context()
    # A None node_input means a classic agent-tree run.
    if node_input is not None:
      parent_context = parent_context.model_copy(
          update={'user_content': to_user_content(node_input)}
      )

    last_text: str | None = None
    # Keep in sync with BaseAgent._run_impl: super() cannot be delegated to,
    # since it re-derives the invocation context and would drop node_input.
    async for event in self.run_async(parent_context=parent_context):
      if event.author:
        ctx.event_author = event.author
      if not event.node_info.path and event.author == self.name:
        event.node_info.path = ctx.node_path
      text = final_model_text(event, self.name)
      if text is not None:
        last_text = text
      yield event

    # Both assignments are needed: NodeRunner._enrich_event reads
    # ctx.event_author, and a direct consumer reads author=.
    ctx.event_author = self.name
    yield Event(
        invocation_id=parent_context.invocation_id,
        author=self.name,
        branch=parent_context.branch,
        output=last_text or '',
    )
