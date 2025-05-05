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

from typing import AsyncGenerator
from typing import Union

from google.genai import types
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from pydantic import ConfigDict
from typing_extensions import override

from ..events.event import Event
from .base_agent import BaseAgent
from .invocation_context import InvocationContext


def _get_last_human_messages(events: list[Event]) -> list[HumanMessage]:
  """Extracts last human messages from given list of events.

  Args:
    events: the list of events

  Returns:
    list of last human messages
  """
  messages = []
  for event in reversed(events):
    if messages and event.author != 'user':
      break
    if event.author == 'user' and event.content and event.content.parts:
      messages.append(HumanMessage(content=event.content.parts[0].text))
  return list(reversed(messages))


class LangGraphAgent(BaseAgent):
  """Currently a concept implementation, supports single and multi-turn."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
  )
  """The pydantic model config."""

  graph: CompiledGraph

  instruction: str = ''

  @override
  async def _run_async_impl(
      self,
      ctx: InvocationContext,
  ) -> AsyncGenerator[Event, None]:

    # Needed for langgraph checkpointer (for subsequent invocations; multi-turn)
    config: RunnableConfig = {'configurable': {'thread_id': ctx.session.id}}

    # Add instruction as SystemMessage if graph state is empty
    current_graph_state = self.graph.get_state(config)
    graph_messages = (
        current_graph_state.values.get('messages', [])
        if current_graph_state.values
        else []
    )
    messages = (
        [SystemMessage(content=self.instruction)]
        if self.instruction and not graph_messages
        else []
    )
    # Add events to messages (evaluating the memory used; parent agent vs checkpointer)
    messages += self._get_messages(ctx.session.events)

    # Use the Runnable
    final_state = self.graph.invoke({'messages': messages}, config)
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

    messages = []
    for event in events:
      if not event.content or not event.content.parts:
        continue
      if event.author == 'user':
        messages.append(HumanMessage(content=event.content.parts[0].text))
      elif event.author == self.name:
        messages.append(AIMessage(content=event.content.parts[0].text))
    return messages
