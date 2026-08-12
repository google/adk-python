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

import hashlib
from typing import Any
from typing import AsyncGenerator
from typing import Union

from google.genai import types
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from pydantic import ConfigDict
from typing_extensions import override

from ..events.event import Event
from .base_agent import BaseAgent
from .invocation_context import InvocationContext


def _get_thread_id(app_name: str, user_id: str, session_id: str) -> str:
  """Derives the LangGraph checkpointer thread id for a session.

  Session ids are caller-chosen and are only unique within an
  (app_name, user_id) pair, so all three components have to take part in the
  thread id. Each component is length-prefixed before hashing so that a
  component containing the separator cannot stand in for a different triple.
  The composite is hashed rather than used verbatim so that the thread id is a
  fixed-length token no checkpointer backend has to escape, and so the user id
  is not written into checkpointer storage; the cost is that a stored row can
  only be tied back to a session by recomputing the digest.

  Args:
    app_name: the app the session belongs to
    user_id: the user the session belongs to
    session_id: the session id

  Returns:
    a deterministic thread id for the session
  """
  key = '|'.join(
      f'{len(component)}:{component}'
      for component in (app_name, user_id, session_id)
  )
  return hashlib.sha256(key.encode('utf-8')).hexdigest()


def _get_last_human_messages(
    events: list[Event],
) -> list[Union[HumanMessage, AIMessage]]:
  """Extracts last human messages from given list of events.

  Args:
    events: the list of events

  Returns:
    list of last human messages
  """
  messages: list[Union[HumanMessage, AIMessage]] = []
  for event in reversed(events):
    if messages and event.author != 'user':
      break
    if event.author == 'user' and event.content and event.content.parts:
      messages.append(HumanMessage(content=event.content.parts[0].text))
  return list(reversed(messages))


class LangGraphAgent(BaseAgent):
  """Adapts a compiled LangGraph state graph for single or multi-turn use.

  When using a persistent checkpointer, set ``LANGGRAPH_STRICT_MSGPACK=true``
  before importing LangGraph and compiling the graph. LangGraph's patched
  releases provide schema-derived checkpoint allowlisting, but do not enable
  strict deserialization by default.

  The checkpointer thread id is derived from the session's app name, user id
  and id together, because session ids are only unique within an
  (app name, user id) pair. Checkpoints written by earlier releases, which
  keyed the thread on the session id alone, are not reused: with a persistent
  checkpointer the first turn after upgrading resumes from empty graph state.
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
  )
  """The pydantic model config."""

  graph: CompiledStateGraph

  instruction: str = ''

  @override
  async def _run_async_impl(
      self,
      ctx: InvocationContext,
  ) -> AsyncGenerator[Event, None]:

    # Needed for langgraph checkpointer (for subsequent invocations; multi-turn)
    config: RunnableConfig = {
        'configurable': {
            'thread_id': _get_thread_id(
                ctx.session.app_name, ctx.session.user_id, ctx.session.id
            )
        }
    }

    # Add instruction as SystemMessage if graph state is empty. State lookup is
    # only valid when the compiled graph has a checkpointer.
    graph_messages: list[Any] = []
    if self.graph.checkpointer:
      current_graph_state = await self.graph.aget_state(config)
      graph_messages = (
          current_graph_state.values.get('messages', [])
          if current_graph_state.values
          else []
      )
    messages: list[BaseMessage] = (
        [SystemMessage(content=self.instruction)]
        if self.instruction and not graph_messages
        else []
    )
    # Add events to messages (evaluating the memory used; parent agent vs checkpointer)
    messages += self._get_messages(ctx.session.events)

    # Use the Runnable
    final_state = await self.graph.ainvoke({'messages': messages}, config)
    result = final_state['messages'][-1].content

    result_event = Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        branch=ctx.branch,
        content=types.Content(
            role='model',
            parts=[types.Part.from_text(text=result)],
        ),
    )
    yield result_event

  def _get_messages(
      self, events: list[Event]
  ) -> list[Union[HumanMessage, AIMessage]]:
    """Extracts messages from given list of events.

    If the developer provides their own memory within langgraph, we return the
    last user messages only. Otherwise, we return all messages between the user
    and the agent.

    Args:
      events: the list of events

    Returns:
      list of messages
    """
    if self.graph.checkpointer:
      return _get_last_human_messages(events)
    else:
      return self._get_conversation_with_agent(events)

  def _get_conversation_with_agent(
      self, events: list[Event]
  ) -> list[Union[HumanMessage, AIMessage]]:
    """Extracts messages from given list of events.

    Args:
      events: the list of events

    Returns:
      list of messages
    """

    messages: list[Union[HumanMessage, AIMessage]] = []
    for event in events:
      if not event.content or not event.content.parts:
        continue
      if event.author == 'user':
        messages.append(HumanMessage(content=event.content.parts[0].text))
      elif event.author == self.name:
        messages.append(AIMessage(content=event.content.parts[0].text))
    return messages
