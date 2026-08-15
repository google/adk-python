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

"""Antigravity SDK agent wrapper for ADK.

Wraps a pre-configured ``google.antigravity.Agent`` as a native ADK
``BaseAgent`` node, delegating each turn to the Antigravity runner and
streaming its trajectory steps back as ADK events.

The SDK harness runs its own agent loop and owns its own conversation, so an
``AntigravityAgent`` can never be given ADK ``sub_agents``, and it must run as
a standalone root agent unless it declares ``mode='single_turn'``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Literal

from google.antigravity import Agent
from google.antigravity import AgentConfig
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from . import _event_converter
from . import _trajectory_files
from ...agents.base_agent import BaseAgent
from ...agents.context import Context
from ...agents.invocation_context import InvocationContext
from ...agents.run_config import StreamingMode
from ...events.event import Event
from ...utils.content_utils import to_user_content

logger = logging.getLogger('google_adk.' + __name__)

_NO_SUB_AGENTS_MESSAGE = (
    'AntigravityAgent cannot be given sub_agents: the Antigravity SDK harness '
    'runs its own agent loop and would never dispatch to an ADK child.'
)
_PARENT_REQUIRES_SINGLE_TURN_MESSAGE = (
    "AntigravityAgent may only be a sub-agent when it sets mode='single_turn', "
    'where the parent composes a self-contained request. Otherwise it must run '
    'as a standalone root agent.'
)


def _derive_conversation_id(session_id: str, agent_name: str) -> str:
  """Returns a deterministic conversation id (>=32 chars, [a-zA-Z0-9-])."""
  # Hashing keeps the id stable across turns while satisfying the SDK's length
  # and character constraints.
  return hashlib.sha256(f'{session_id}/{agent_name}'.encode()).hexdigest()


def _final_model_text(event: Event, author: str) -> str | None:
  """Returns an event's user-visible model text, or None if it carries none.

  Partials, other authors, and thought/function parts do not count.

  Args:
    event: The event to inspect.
    author: The agent name whose events count as model output.

  Returns:
    The concatenated user-visible text, or None if the event carries none.
  """
  if event.partial or event.author != author or not event.content:
    return None
  parts = event.content.parts or []
  chunks = [
      part.text
      for part in parts
      if part.text
      and not part.thought
      and not part.function_call
      and not part.function_response
  ]
  return ''.join(chunks) if chunks else None


class AntigravityAgent(BaseAgent):
  """Runs a Google Antigravity SDK agent as an ADK agent.

  Each turn spins up a fresh SDK ``Agent`` from ``config`` and exposes its
  trajectory steps as standard ADK events recorded in the session.

  Must be a standalone root agent unless ``mode='single_turn'``; see the module
  docstring.
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      use_attribute_docstrings=True,
      extra='forbid',
  )

  config: AgentConfig = Field(exclude=True)
  """The ``google.antigravity.AgentConfig`` describing the SDK agent.

  Typically a ``LocalAgentConfig``. Excluded from serialization: it holds
  runtime wiring (e.g. callable tools) that is not JSON-serializable.
  """

  mode: Literal['single_turn'] | None = Field(default=None, frozen=True)
  """Composition mode when used as a sub-agent.

  ``'single_turn'`` is what allows this agent to have a parent at all: the
  parent ``LlmAgent`` exposes it as an inline tool taking a ``request`` string,
  rather than as an LLM-transfer target. The parent composes the task; session
  history is not forwarded. Each call is an independent conversation: nothing
  is resumed, and ``config.save_dir`` is not required.

  Leave as ``None`` for a standalone root agent. Frozen, because the adoption
  guard only gets to check it once, at construction.
  """

  @override
  def model_post_init(self, __context: Any) -> None:
    super().model_post_init(__context)
    if self.sub_agents:
      raise ValueError(_NO_SUB_AGENTS_MESSAGE)

  def __setattr__(self, name: str, value: Any) -> None:
    # A parent assigns `parent_agent` on adoption; rejecting it here is what
    # enforces the restriction. `mode` via __dict__: fields may be unpopulated.
    if (
        name == 'parent_agent'
        and value is not None
        and self.__dict__.get('mode') != 'single_turn'
    ):
      raise ValueError(_PARENT_REQUIRES_SINGLE_TURN_MESSAGE)
    super().__setattr__(name, value)

  def _extract_user_prompt(self, ctx: InvocationContext) -> str:
    """Returns the user text that started this invocation."""
    if ctx.user_content and ctx.user_content.parts:
      for part in ctx.user_content.parts:
        if part.text:
          return str(part.text)
    return ''

  @property
  def _sdk_agent_cls(self) -> type[Agent]:
    """The SDK Agent class each turn runs on.

    Override in a subclass to run turns on a different Agent implementation.
    """
    # The ignore is for the SDK being untyped to mypy, not for the return.
    return Agent  # type: ignore[no-any-return]

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    # A single-turn call neither resumes an earlier conversation nor leaves
    # one behind, so it needs no folder to keep them in.
    single_turn = self.mode == 'single_turn'
    save_dir = self.config.save_dir
    if not single_turn and not save_dir:
      raise ValueError(
          'AntigravityAgent requires config.save_dir to persist and resume '
          'conversation trajectories across turns.'
      )

    prompt = self._extract_user_prompt(ctx)

    # The SDK Agent's AsyncExitStack is single-use, so each turn needs a fresh
    # one; copying also avoids mutating the caller's config.
    config = self.config.model_copy(deep=True)
    # The id is keyed on the ADK session, so the turns of one session share a
    # conversation. Single-turn calls are not, so they get no id.
    conversation_id: str | None = None
    resumed = False
    resume_step_index = -1  # Highest step_index already emitted; -1 = none.
    # `and save_dir` is redundant at runtime (the guard above raised already);
    # it narrows save_dir to str for the type checker.
    if not single_turn and save_dir:
      conversation_id = _derive_conversation_id(ctx.session.id, self.name)
      # Resume only when a trajectory already exists; the harness errors if a
      # conversation_id is given with no matching file on disk.
      resumed = _trajectory_files.has_trajectory(save_dir, conversation_id)
      if resumed:
        # On resume the harness replays the whole trajectory; skip steps
        # already emitted in earlier turns and track the new max to persist.
        resume_step_index = _trajectory_files.load_resume_step_index(
            save_dir, conversation_id
        )
    config.conversation_id = conversation_id if resumed else None

    max_step_index = resume_step_index

    seen_tool_calls: set[str] = set()
    seen_tool_results: set[str] = set()
    streaming = bool(
        ctx.run_config and ctx.run_config.streaming_mode == StreamingMode.SSE
    )

    async with self._sdk_agent_cls(config) as active_agent:
      await active_agent.conversation.send(prompt)

      async for step in active_agent.conversation.receive_steps():
        if step.step_index <= resume_step_index:
          continue
        max_step_index = max(max_step_index, step.step_index)
        for event in _event_converter.convert_step_to_events(
            step,
            ctx=ctx,
            author=self.name,
            seen_tool_calls=seen_tool_calls,
            seen_tool_results=seen_tool_results,
            streaming=streaming,
        ):
          yield event

      harness_conversation_id = active_agent.conversation_id

    # On a fresh turn the harness wrote traj-<random> (flushed when the session
    # exits above); rename it. No id under single-turn, so that case skips.
    if save_dir and conversation_id:
      if not resumed and harness_conversation_id:
        _trajectory_files.rename_trajectory(
            save_dir, conversation_id, harness_conversation_id
        )
      _trajectory_files.save_resume_step_index(
          save_dir, conversation_id, max_step_index
      )

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Event, None]:
    """Runs the agent as a node, threading node_input in and output out.

    Unlike ``BaseAgent._run_impl``, the parent's composed request is used as
    the prompt, and the final model text is reported as the node's output.

    Args:
      ctx: The node context for this run.
      node_input: The parent's composed request, or None for a classic
        agent-tree run.

    Yields:
      The agent's events, followed by a trailing event whose ``output`` is the
      final model text, or the empty string if there was none.
    """
    parent_context = ctx.get_invocation_context()
    if node_input is not None:
      parent_context = parent_context.model_copy(
          update={'user_content': to_user_content(node_input)}
      )

    last_text: str | None = None
    # Keep in sync with BaseAgent._run_impl: super() cannot be delegated to,
    # since it re-derives the invocation context and would drop node_input.
    async for event in self.run_async(parent_context=parent_context):
      # Preserve author by setting it in context for NodeRunner.
      if event.author:
        ctx.event_author = event.author
      if not event.node_info.path and event.author == self.name:
        event.node_info.path = ctx.node_path
      if (text := _final_model_text(event, self.name)) is not None:
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
