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

"""Exposes ADK sub-agents to an Antigravity SDK harness as client tools.

The harness runs the Antigravity agent's loop over plain Python callables, not
ADK nodes, and binds them once per conversation rather than once per turn.
Each call therefore runs the ADK child in isolation — its own ``Runner`` and an
in-memory session — and returns only its final text. Close to ADK's own
``AgentTool``, except that code-execution output and executable code are not
returned.
"""

from __future__ import annotations

from typing import Protocol

from google.genai import types as genai_types

from ...agents.base_agent import BaseAgent
from ...utils.context_utils import Aclosing
from ._event_converter import final_model_text

_SUB_AGENT_USER_ID = 'antigravity_sub_agent'


class SubAgentTool(Protocol):
  """The callable the Antigravity harness binds, plus what it is bound as."""

  # The model reads both: `__name__` is the tool name, `__doc__` its
  # description.
  __name__: str
  __doc__: str | None

  async def __call__(self, request: str) -> str:
    ...


def make_sub_agent_tool(child: BaseAgent) -> SubAgentTool:
  """Wraps ``child`` as a client tool answering with its final text.

  Args:
    child: The ADK agent to run. Each call runs it once, in a fresh session.

  Returns:
    An async callable carrying ``child``'s name as ``__name__`` and its
    description as ``__doc__``. It answers with the child's last user-visible
    text, falling back to the child's last error message and then to ``''`` --
    never ``None``.

  Raises:
    Exception: Whatever the child raises propagates to the caller, unlike
      ``AgentTool``, which reports failures as an error string.
  """

  async def call(request: str) -> str:
    # Imported here, not at module scope: ``runners`` pulls in most of ADK,
    # and importing it eagerly from a labs module is a cycle waiting to
    # happen. ``AgentTool`` defers the same import for the same reason.
    from ...runners import Runner  # pylint: disable=g-import-not-at-top
    from ...sessions.in_memory_session_service import InMemorySessionService  # pylint: disable=g-import-not-at-top

    runner = Runner(
        app_name=child.name,
        agent=child,
        session_service=InMemorySessionService(),
    )
    try:
      session = await runner.session_service.create_session(
          app_name=child.name, user_id=_SUB_AGENT_USER_ID
      )
      message = genai_types.Content(
          role='user', parts=[genai_types.Part.from_text(text=request)]
      )
      last_text: str | None = None
      last_error: str | None = None
      # ``Aclosing`` as ``AgentTool`` does: if the caller is cancelled mid-turn
      # the generator is closed here rather than whenever it is collected,
      # which is what keeps the close below on the same task that opened it.
      async with Aclosing(
          runner.run_async(
              user_id=_SUB_AGENT_USER_ID,
              session_id=session.id,
              new_message=message,
          )
      ) as agen:
        async for event in agen:
          if event.error_message:
            last_error = event.error_message
          # Not filtered by author: a composite ADK child yields its own
          # sub-agents' events under their names, still part of its answer.
          text = final_model_text(event)
          if text is not None:
            last_text = text
      # A blocked or cut-off turn carries an error and no content at all,
      # which the Antigravity model must tell apart from a silent ADK child.
      return last_text or last_error or ''
    finally:
      # Closes the caller's own child: its toolsets, plugins and session
      # service are in scope here, so a later call reconnects them. Deliberate,
      # and what ``AgentTool`` does, because an MCP session left open
      # resurfaces later as "Attempted to exit cancel scope in a different
      # task". ``finally`` rather than AgentTool's straight-line close, so a
      # child that raises still closes; it does not swallow the exception.
      await runner.close()

  # What the Antigravity model reads: the ADK child's name becomes the tool
  # name it calls, and the child's description the tool description.
  call.__name__ = child.name
  call.__doc__ = child.description
  return call
