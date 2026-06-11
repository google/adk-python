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

The Antigravity SDK currently only supports its local (in-process Go harness)
mode. That mode owns its own session lifecycle and cannot participate in ADK's
multi-agent delegation, so an ``AntigravityAgent`` is restricted to running as a
standalone root agent. This restriction is expected to be lifted once the SDK
gains a remote connection mode.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import AsyncGenerator

from google.antigravity import Agent
from google.antigravity import AgentConfig
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from . import _event_converter
from . import _trajectory_files
from ...agents.base_agent import BaseAgent
from ...agents.invocation_context import InvocationContext
from ...agents.run_config import StreamingMode
from ...events.event import Event

logger = logging.getLogger('google_adk.' + __name__)

_ROOT_ONLY_MESSAGE = (
    'AntigravityAgent currently only supports the Antigravity SDK local mode, '
    'which must run as a standalone root agent. Using it as a sub-agent or '
    'giving it sub-agents is not supported yet (this restriction is temporary '
    'and will be lifted once the SDK supports remote connection modes).'
)


class AntigravityAgent(BaseAgent):
  """Runs a Google Antigravity SDK agent as an ADK root agent.

  Each turn spins up a fresh SDK ``Agent`` from ``config`` and exposes its
  trajectory steps as standard ADK events recorded in the session.
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      use_attribute_docstrings=True,
      extra='forbid',
  )

  config: AgentConfig = Field(exclude=True)
  """The ``google.antigravity.AgentConfig`` describing the SDK agent.

  Typically a ``LocalAgentConfig``. Excluded from serialization because it holds
  runtime wiring (e.g. callable tools) that is not JSON-serializable.
  """

  @override
  def model_post_init(self, __context: Any) -> None:
    super().model_post_init(__context)
    if self.sub_agents:
      raise ValueError(_ROOT_ONLY_MESSAGE)

  def __setattr__(self, name: str, value: Any) -> None:
    # `parent_agent` is assigned by a parent agent when it adopts this agent as
    # a sub-agent (see BaseAgent.__set_parent_agent_for_sub_agents). Rejecting a
    # non-None assignment here is what enforces the root-only restriction for
    # the "used as a sub-agent" direction at construction time.
    if name == 'parent_agent' and value is not None:
      raise ValueError(_ROOT_ONLY_MESSAGE)
    super().__setattr__(name, value)

  def _extract_user_prompt(self, ctx: InvocationContext) -> str:
    """Returns the user text that started this invocation."""
    if ctx.user_content and ctx.user_content.parts:
      for part in ctx.user_content.parts:
        if part.text:
          return str(part.text)
    return ''

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    save_dir = self.config.save_dir
    if not save_dir:
      raise ValueError(
          'AntigravityAgent requires config.save_dir to persist and resume '
          'conversation trajectories across turns.'
      )

    prompt = self._extract_user_prompt(ctx)

    # Deep-copy the config so each turn gets an independent, fresh SDK Agent.
    # The SDK Agent's AsyncExitStack is single-use, so a new instance is needed
    # per turn; copying also avoids mutating the caller's config.
    config = self.config.model_copy(deep=True)
    conversation_id = f'{ctx.session.id}_{self.name}'

    # Resume only when a trajectory already exists; the harness errors if a
    # conversation_id is given with no matching file on disk.
    resumed = _trajectory_files.has_trajectory(save_dir, conversation_id)
    config.conversation_id = conversation_id if resumed else None

    # On resume the harness replays the whole trajectory; skip steps already
    # emitted in earlier turns and track the new max index to persist.
    resume_step_index = (
        _trajectory_files.load_resume_step_index(save_dir, conversation_id)
        if resumed
        else -1
    )
    max_step_index = resume_step_index

    seen_tool_calls: set[str] = set()
    seen_tool_results: set[str] = set()
    streaming = bool(
        ctx.run_config and ctx.run_config.streaming_mode == StreamingMode.SSE
    )

    async with Agent(config) as active_agent:
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

    # On a fresh turn the harness wrote traj-<random>; rename it to our
    # deterministic name (the file is flushed once the session above exits).
    if not resumed and harness_conversation_id:
      _trajectory_files.rename_trajectory(
          save_dir, conversation_id, harness_conversation_id
      )
    _trajectory_files.save_resume_step_index(
        save_dir, conversation_id, max_step_index
    )
