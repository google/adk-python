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

"""Tests for SnowflakeCortexAgent.

Verifies the configuration surface: composition guards, credential exclusion
from ``repr`` and every serialization path, and per-agent session state keys.
"""

# `_state_key` is the only seam for the state key until the run loop lands and
# surfaces it in a `state_delta`.
# pylint: disable=protected-access

from __future__ import annotations

import functools
import json
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.run_config import RunConfig
from google.adk.cli.utils.graph_serialization import serialize_agent
from google.adk.events.event import Event
from google.adk.labs.snowflake import SnowflakeCortexAgent
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types as genai_types
from pydantic import ValidationError
import pytest

_TOKEN = 'pat-secret-token-value'


def _bearer_headers(ctx: ReadonlyContext, *, token: str) -> dict[str, str]:
  del ctx
  return {'Authorization': f'Bearer {token}'}


# A `functools.partial` rather than a closure, because its `repr` prints the
# bound token. That makes "the token is absent" a real check: it would appear
# if the field were not excluded from `repr` and serialization.
_HEADER_PROVIDER = functools.partial(_bearer_headers, token=_TOKEN)


def _make_agent(name: str = 'cortex', **kwargs: object) -> SnowflakeCortexAgent:
  """A minimal SnowflakeCortexAgent pointing at a fake account."""
  return SnowflakeCortexAgent(
      name=name,
      account_url='https://example.snowflakecomputing.com',
      database='SALES_DB',
      schema_name='ANALYTICS',
      cortex_agent_name='SALES_AGENT',
      header_provider=_HEADER_PROVIDER,
      **kwargs,
  )


class _StubChild(BaseAgent):
  """A runnable ADK child agent."""

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(invocation_id=ctx.invocation_id, author=self.name)


async def _invocation_context(agent: BaseAgent) -> InvocationContext:
  """A real InvocationContext rooted at `agent`."""
  session_service = InMemorySessionService()
  return InvocationContext(
      session_service=session_service,
      invocation_id='inv_1',
      agent=agent,
      session=await session_service.create_session(
          app_name='test_app', user_id='test_user'
      ),
      user_content=genai_types.Content(
          role='user', parts=[genai_types.Part.from_text(text='hello')]
      ),
      run_config=RunConfig(),
  )


def test_standalone_agent_is_allowed():
  """An agent with neither parent nor children constructs cleanly."""
  agent = _make_agent()

  assert agent.parent_agent is None
  assert agent.sub_agents == []


def test_defaults_are_the_documented_values():
  """Options not passed take the documented defaults."""
  agent = _make_agent()

  assert agent.timeout == 900.0
  assert agent.cancel_on_disconnect is True
  assert agent.max_tool_result_bytes == 32 * 1024
  assert agent.include_thinking_in_final_event is False


@pytest.mark.parametrize('field', ['timeout', 'max_tool_result_bytes'])
def test_non_positive_bounds_are_rejected(field: str):
  """A zero timeout or result size limit fails validation."""
  with pytest.raises(ValidationError, match='greater than 0'):
    _make_agent(**{field: 0})


def test_sub_agents_are_rejected():
  """Declaring `sub_agents` fails at construction."""
  child = _StubChild(name='reviewer')

  with pytest.raises(ValueError, match='sub_agents'):
    _make_agent(sub_agents=[child])


def test_using_as_sub_agent_is_rejected():
  """A parent listing this agent in `sub_agents` fails to construct."""
  agent = _make_agent()

  with pytest.raises(ValueError, match='root agent'):
    BaseAgent(name='parent', sub_agents=[agent])

  assert agent.parent_agent is None


async def test_sub_agents_added_after_construction_are_rejected_at_run():
  """Mutating `sub_agents` past validation still fails, at the first turn."""
  agent = _make_agent()
  agent.sub_agents.append(_StubChild(name='late'))
  ctx = await _invocation_context(agent)

  with pytest.raises(ValueError, match='sub_agents'):
    async for _ in agent.run_async(ctx):
      pass


async def test_sub_agents_added_by_clone_are_rejected_at_run():
  """A clone given `sub_agents` skips construction checks but cannot run."""
  agent = _make_agent()
  cloned = agent.clone(update={'sub_agents': [_StubChild(name='late')]})
  ctx = await _invocation_context(cloned)

  with pytest.raises(ValueError, match='sub_agents'):
    async for _ in cloned.run_async(ctx):
      pass


async def test_running_is_not_implemented_yet():
  """Running a turn raises until the Snowflake client lands."""
  agent = _make_agent()
  ctx = await _invocation_context(agent)

  with pytest.raises(NotImplementedError):
    async for _ in agent.run_async(ctx):
      pass


def test_header_provider_is_hidden_from_repr():
  """`repr` shows neither the provider nor the token it carries."""
  agent = _make_agent()

  text = repr(agent)

  assert 'header_provider' not in text
  assert _TOKEN not in text


def test_header_provider_is_excluded_from_model_dump():
  """`model_dump` omits the provider and the token it carries."""
  agent = _make_agent()

  dumped = agent.model_dump()

  assert 'header_provider' not in dumped
  assert _TOKEN not in str(dumped)
  assert dumped['cortex_agent_name'] == 'SALES_AGENT'


def test_header_provider_is_hidden_from_the_adk_web_agent_graph():
  """The `adk web` agent graph omits the provider and the token it carries."""
  agent = _make_agent()

  serialized = json.dumps(serialize_agent(agent), default=str)

  assert 'header_provider' not in serialized
  assert _TOKEN not in serialized


async def test_header_provider_stays_callable_on_the_instance():
  """Exclusion from output leaves the provider itself in place."""
  agent = _make_agent()
  ctx = await _invocation_context(agent)

  headers = agent.header_provider(ReadonlyContext(ctx))

  assert headers == {'Authorization': f'Bearer {_TOKEN}'}


def test_clone_keeps_the_header_provider():
  """A clone can still authenticate: exclusion is from output, not copies."""
  agent = _make_agent()

  cloned = agent.clone(update={'name': 'copy'})

  assert cloned.header_provider is agent.header_provider


def test_state_key_is_scoped_by_agent_name():
  """Two agents with different names keep separate Snowflake threads."""
  first = _make_agent(name='first')
  second = _make_agent(name='second')

  assert first._state_key() != second._state_key()
  assert first._state_key() == _make_agent(name='first')._state_key()
