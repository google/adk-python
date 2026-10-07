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

import asyncio
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
from google.genai import types as genai_types
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
from ._tool_result_capture import SubagentCallCapture
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
    pass

  @property
  def last_turn_usage(self) -> Any | None:
    pass

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
    # The Antigravity SDK ``Agent`` is already entered by ``_enter_sdk_agent``;
    # this exists only so callers can use ``async with``, which reports the true
    # unwinding exception to ``__aexit__`` rather than ``sys.exc_info()``.
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
    # is classified by `isinstance` against the hook classes of the Antigravity
    # SDK it came from, so the caller cannot use `ToolResultCapture` directly --
    # it is bound to this copy's `PostToolCallHook`. Typed as the non-hook base
    # `ToolResultBuffer`, not this subclass, so an override can subclass that
    # base plus its own hook rather than dragging this copy's hook base in.
    return ToolResultCapture

  @property
  def _tool_error_capture_cls(self) -> type[ToolErrorCapture]:
    # The same seam, overridden for the same reason: each Antigravity SDK copy
    # binds its own `OnToolErrorHook`. Separate from the success capture because
    # both hook interfaces name their entry point `run`. No buffer-only base to
    # widen to here -- this holds a buffer rather than being one -- so the type
    # stays the concrete class.
    return ToolErrorCapture

  @property
  def _subagent_call_capture_cls(self) -> type[SubagentCallCapture]:
    # Pre-tool capture seam for recording `start_subagent` `TypeName` / `Role`
    # arguments before the harness sends the empty `ActionInvokeSubagent` step.
    return SubagentCallCapture

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
      hooks: list[Any] = [
          tool_results,
          self._tool_error_capture_cls(tool_results),
      ]
      if self.config.subagents:
        hooks.append(self._subagent_call_capture_cls(tool_results))
      config.hooks = list(config.hooks) + hooks
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
    # Enable tool-result capture whenever ADK sub_agents, SDK client tools, or
    # SDK subagents are configured.
    config_tools = self.config.tools or ()
    has_client_tools = bool(
        self.sub_agents
        or any(not isinstance(t, str) for t in config_tools)
        or self.config.subagents
    )
    tool_results = self._tool_result_capture_cls() if has_client_tools else None
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

  def _extract_tool_name(self, tool: Any) -> str:
    """Extracts the tool name safely without dynamic reflection."""
    if isinstance(tool, str):
      return tool
    try:
      return str(tool.__name__)
    except AttributeError:
      return ''

  def _matching_subagent_by_tool(self, step: Step) -> str | None:
    """Returns the SDK subagent name if a tool call belongs to one subagent."""
    tool_calls = (
        step.tool_calls if isinstance(step.tool_calls, (list, tuple)) else ()
    )
    sdk_subagents = (
        self.config.subagents
        if isinstance(self.config.subagents, (list, tuple))
        else ()
    )
    for call in tool_calls:
      matching = [
          subagent.name
          for subagent in sdk_subagents
          if isinstance(subagent.name, str)
          and subagent.name
          and any(
              self._extract_tool_name(tool) == call.name
              for tool in subagent.tools or ()
          )
      ]
      if len(matching) == 1:
        return matching[0]
    return None

  def _extract_session_subagent_authors(
      self, ctx: InvocationContext
  ) -> dict[str, str]:
    """Returns prior sub-agent trajectory-to-author mappings from session."""
    session = ctx.session
    session_events = (
        session.events
        if session is not None and isinstance(session.events, (list, tuple))
        else ()
    )
    authors: dict[str, str] = {}
    excluded_authors = {self.name, 'user', f'{self.name}_subagent'}
    for past_event in session_events:
      custom_metadata = past_event.custom_metadata
      author = past_event.author
      if not isinstance(custom_metadata, dict) or not isinstance(author, str):
        continue
      trajectory_id = custom_metadata.get('trajectory_id')
      depth = custom_metadata.get('depth')
      has_subagent_meta = (
          isinstance(depth, int) and not isinstance(depth, bool) and depth > 0
      ) or bool(custom_metadata.get('parent_trajectory_id'))
      if (
          isinstance(trajectory_id, str)
          and trajectory_id
          and has_subagent_meta
          and author
          and author not in excluded_authors
      ):
        authors[trajectory_id] = author
    return authors

  def _resolve_step_context(
      self,
      step: Step,
      *,
      main_trajectory_id: str,
      subagent_authors: dict[str, str],
      pending_invocations: list[str],
  ) -> tuple[str, str, dict[str, Any] | None]:
    """Returns ``(author, trajectory_key, custom_metadata)`` for ``step``."""
    # 1. Classify whether the step belongs to a sub-agent trajectory.
    trajectory_id = (
        step.trajectory_id if isinstance(step.trajectory_id, str) else ''
    )
    depth = step.depth if isinstance(step.depth, int) and step.depth > 0 else 0
    parent_trajectory_id = (
        step.parent_trajectory_id
        if isinstance(step.parent_trajectory_id, str)
        else ''
    )
    matched_by_tool = self._matching_subagent_by_tool(step)
    is_subagent = (
        depth > 0
        or bool(parent_trajectory_id)
        or (bool(trajectory_id) and trajectory_id in subagent_authors)
        or (
            not trajectory_id
            and matched_by_tool is not None
            and matched_by_tool in subagent_authors.values()
        )
    )

    # 2. Resolve event author and normalized trajectory hierarchy metadata.
    author = self.name
    trajectory_key = main_trajectory_id or trajectory_id
    if is_subagent:
      sdk_subagents = (
          self.config.subagents
          if isinstance(self.config.subagents, (list, tuple))
          else ()
      )
      if trajectory_id and trajectory_id in subagent_authors:
        author = subagent_authors[trajectory_id]
      elif matched_by_tool is not None:
        author = matched_by_tool
      elif pending_invocations:
        author = pending_invocations.pop(0)
      elif (
          len(sdk_subagents) == 1
          and isinstance(sdk_subagents[0].name, str)
          and sdk_subagents[0].name
      ):
        author = sdk_subagents[0].name
      else:
        author = f'{self.name}_subagent'
      if trajectory_id:
        if author != f'{self.name}_subagent':
          subagent_authors[trajectory_id] = author
        trajectory_key = trajectory_id
      else:
        trajectory_key = next(
            (
                sub_traj_id
                for sub_traj_id, sub_author in reversed(
                    list(subagent_authors.items())
                )
                if sub_author == author
            ),
            '',
        )
      depth = depth if depth > 0 else 1
      parent_trajectory_id = parent_trajectory_id or main_trajectory_id

    if not trajectory_key and not parent_trajectory_id and depth <= 0:
      return author, trajectory_key, None
    step_index = step.step_index if isinstance(step.step_index, int) else 0
    metadata: dict[str, Any] = {
        'trajectory_id': trajectory_key,
        'step_index': step_index,
        'depth': depth,
    }
    if parent_trajectory_id:
      metadata['parent_trajectory_id'] = parent_trajectory_id
    return author, trajectory_key, metadata

  async def _run_turn(
      self, active: _ActiveConversation, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    # 1. Initialize per-turn tracking buffers and send the user prompt.
    seen_tool_calls: set[str] = set()
    seen_tool_results: set[str] = set()
    pending_function_calls: dict[str, list[Event]] = {}
    seen_thought_steps: dict[str, str] = {}
    latest_thoughts: dict[str, tuple[str, Any]] = {}
    subagent_authors: dict[str, str] = self._extract_session_subagent_authors(
        ctx
    )
    pending_invocations: list[str] = (
        active.tool_results.pending_subagents
        if active.tool_results is not None
        else []
    )
    main_trajectory_id = ''
    streaming = bool(
        ctx.run_config and ctx.run_config.streaming_mode == StreamingMode.SSE
    )

    await active.agent.conversation.send(self._extract_user_prompt(ctx))

    # 2. Stream trajectory steps and convert each into ADK events.
    async def _stream_raw_events() -> AsyncGenerator[Event, None]:
      nonlocal main_trajectory_id
      async for step in active.agent.conversation.receive_steps():
        if active.tool_results is not None:
          # Yield to the event loop so backgrounded post_tool_call hook
          # tasks can record results before convert_step_to_events drains them.
          await asyncio.sleep(0)
        trajectory_id = (
            step.trajectory_id if isinstance(step.trajectory_id, str) else ''
        )
        if not main_trajectory_id:
          step_depth = (
              step.depth
              if isinstance(step.depth, int) and step.depth > 0
              else 0
          )
          parent_traj_id = (
              step.parent_trajectory_id
              if isinstance(step.parent_trajectory_id, str)
              else ''
          )
          if trajectory_id and step_depth == 0 and not parent_traj_id:
            main_trajectory_id = trajectory_id
          elif parent_traj_id and step_depth <= 1:
            main_trajectory_id = parent_traj_id
        if main_trajectory_id and '' in pending_function_calls:
          orphan_calls = pending_function_calls.pop('')
          pending_function_calls.setdefault(main_trajectory_id, []).extend(
              orphan_calls
          )
        if main_trajectory_id and '' in latest_thoughts:
          orphan_key, orphan_part = latest_thoughts.pop('')
          if orphan_key.startswith(':'):
            orphan_key = f'{main_trajectory_id}{orphan_key}'
          latest_thoughts.setdefault(
              main_trajectory_id, (orphan_key, orphan_part)
          )
        step_author, trajectory_key, step_metadata = self._resolve_step_context(
            step,
            main_trajectory_id=main_trajectory_id,
            subagent_authors=subagent_authors,
            pending_invocations=pending_invocations,
        )
        if (
            not trajectory_id
            and not step.tool_calls
            and not pending_function_calls.get(trajectory_key)
        ):
          active_traj_key = next(
              (
                  cand_key
                  for cand_key, cand_events in reversed(
                      list(pending_function_calls.items())
                  )
                  if cand_events
              ),
              None,
          )
          if active_traj_key is not None:
            trajectory_key = active_traj_key
            pending_meta = pending_function_calls[active_traj_key][
                0
            ].custom_metadata
            if isinstance(pending_meta, dict):
              step_metadata = {
                  **pending_meta,
                  'step_index': (
                      step.step_index if isinstance(step.step_index, int) else 0
                  ),
              }
        pending_trajectory_events = pending_function_calls.setdefault(
            trajectory_key, []
        )
        for event in convert_step_to_events(
            step,
            ctx=ctx,
            author=step_author,
            seen_tool_calls=seen_tool_calls,
            seen_tool_results=seen_tool_results,
            tool_results=active.tool_results,
            streaming=streaming,
            pending_function_calls=pending_trajectory_events,
            seen_thought_steps=seen_thought_steps,
            latest_thoughts=latest_thoughts,
            custom_metadata=step_metadata,
        ):
          yield event

      # 3. Flush remaining buffered calls and drain any trailing client results.
      for traj_key, pending_trajectory_events in pending_function_calls.items():
        if not pending_trajectory_events:
          continue
        cached = latest_thoughts.pop(traj_key, None)
        if cached is not None and pending_trajectory_events[0].content:
          parts = pending_trajectory_events[0].content.parts
          if parts is not None and not any(
              bool(part.thought) for part in parts
          ):
            parts.insert(0, cached[1])
        for event in pending_trajectory_events:
          if event.content and event.content.parts:
            for part in event.content.parts:
              if part.function_call and part.function_call.id:
                seen_tool_calls.add(part.function_call.id)
          yield event
      pending_function_calls.clear()

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

    has_emitted_events = False
    async for event in _stream_raw_events():
      has_emitted_events = True
      yield event

    if has_emitted_events:
      usage = active.agent.conversation.last_turn_usage
      if usage is not None:
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=usage.prompt_token_count,
                candidates_token_count=usage.candidates_token_count,
                total_token_count=usage.total_token_count,
                thoughts_token_count=usage.thoughts_token_count,
            ),
        )

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
