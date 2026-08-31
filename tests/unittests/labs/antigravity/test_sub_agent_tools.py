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

"""Tests for the bridge exposing ADK sub-agents as Antigravity SDK tools."""

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.events.event import Event
from google.adk.labs.antigravity import _sub_agent_tools
from google.genai import types as genai_types
import pytest


class _EchoAgent(BaseAgent):
  """Replies with the request it was given, prefixed."""

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    text = ''
    for part in (ctx.user_content.parts if ctx.user_content else []) or []:
      if part.text:
        text = part.text
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=genai_types.Content(
            role='model',
            parts=[genai_types.Part.from_text(text=f'echoed: {text}')],
        ),
    )


def test_the_tool_takes_its_name_and_docstring_from_the_child():
  """The harness's model sees the child's own name and description."""
  child = _EchoAgent(name='reviewer', description='Reviews a diff.')

  tool = _sub_agent_tools.make_sub_agent_tool(child)

  assert tool.__name__ == 'reviewer'
  assert tool.__doc__ == 'Reviews a diff.'


@pytest.mark.asyncio
async def test_the_request_reaches_the_child_and_its_text_comes_back():
  child = _EchoAgent(name='reviewer', description='Reviews a diff.')

  tool = _sub_agent_tools.make_sub_agent_tool(child)
  result = await tool(request='look at cl/1')

  assert result == 'echoed: look at cl/1'


@pytest.mark.asyncio
async def test_the_last_text_the_child_emits_is_the_one_returned():
  """The child's final final-text event wins over the earlier ones."""

  class _ChattyAgent(BaseAgent):
    """Emits three separate final-text events, so only the last should win."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      for text in ('first', 'second', 'third'):
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=genai_types.Content(
                role='model', parts=[genai_types.Part.from_text(text=text)]
            ),
        )

  child = _ChattyAgent(name='reviewer', description='Reviews a diff.')

  tool = _sub_agent_tools.make_sub_agent_tool(child)
  result = await tool(request='look at cl/1')

  assert result == 'third'


@pytest.mark.asyncio
async def test_multiple_text_parts_are_joined_with_newlines():
  """As ``AgentTool`` does, not glued together into one run of text."""

  class _TwoPartAgent(BaseAgent):
    """Emits one event whose content carries two separate text parts."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      yield Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          content=genai_types.Content(
              role='model',
              parts=[
                  genai_types.Part.from_text(text='Hello'),
                  genai_types.Part.from_text(text='world'),
              ],
          ),
      )

  child = _TwoPartAgent(name='reviewer', description='Reviews a diff.')

  tool = _sub_agent_tools.make_sub_agent_tool(child)
  result = await tool(request='look at cl/1')

  assert result == 'Hello\nworld'


@pytest.mark.asyncio
async def test_a_child_with_no_visible_text_returns_the_empty_string():
  """A child that emits no user-visible text answers with ``''``."""

  class _SilentAgent(BaseAgent):
    """Emits only a thought, which is not user-visible model text."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      yield Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          content=genai_types.Content(
              role='model',
              parts=[genai_types.Part(text='thinking out loud', thought=True)],
          ),
      )

  child = _SilentAgent(name='reviewer', description='Reviews a diff.')

  tool = _sub_agent_tools.make_sub_agent_tool(child)
  result = await tool(request='look at cl/1')

  # Explicit, not `assert not result`: the harness needs a str, and None --
  # the regression this test exists to catch -- satisfies `not result` too.
  assert result == ''  # pylint: disable=g-explicit-bool-comparison


@pytest.mark.asyncio
async def test_a_composite_child_answers_with_its_sub_agents_text():
  """Events come back authored 'inner', never 'pipeline'.

  Filtering on the child's own name would match nothing and quietly answer ''.
  """
  child = SequentialAgent(
      name='pipeline',
      description='Reviews a diff in stages.',
      sub_agents=[_EchoAgent(name='inner', description='Echoes.')],
  )

  tool = _sub_agent_tools.make_sub_agent_tool(child)
  result = await tool(request='look at cl/1')

  assert result == 'echoed: look at cl/1'


@pytest.mark.asyncio
async def test_a_blocked_child_answers_with_its_error_message():
  """A turn blocked with an error and no content surfaces the error."""

  class _BlockedAgent(BaseAgent):
    """Emits an error and no content at all, as a blocked model turn does."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      yield Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          error_message='blocked by the safety filter',
      )

  child = _BlockedAgent(name='reviewer', description='Reviews a diff.')

  tool = _sub_agent_tools.make_sub_agent_tool(child)
  result = await tool(request='look at cl/1')

  assert result == 'blocked by the safety filter'


@pytest.mark.asyncio
async def test_a_failing_child_propagates():
  """A child that raises is not swallowed: the harness must see the error."""

  class _AngryAgent(BaseAgent):
    """Always fails, to prove failures are not swallowed."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      raise RuntimeError('child exploded')
      yield  # pylint: disable=unreachable

  child = _AngryAgent(name='reviewer', description='Reviews a diff.')

  tool = _sub_agent_tools.make_sub_agent_tool(child)

  with pytest.raises(RuntimeError, match='child exploded'):
    await tool(request='look at cl/1')
