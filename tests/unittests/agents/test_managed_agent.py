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

import asyncio
from typing import Any
from unittest.mock import MagicMock

from google.adk.agents._managed_agent import ManagedAgent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events.event import Event
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import google_search
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from google.genai import types as genai_types
import pytest


class _FakeClient:
  vertexai = False


class _FakeApiClient:

  def __init__(self, location, vertexai=None):
    self.location = location
    self.vertexai = vertexai


class _FakeClientWithLocation:
  """Mimics a genai Client: public vertexai, private _api_client.location."""

  def __init__(self, location, vertexai=None):
    self.vertexai = bool(vertexai)
    self._api_client = _FakeApiClient(location, vertexai)


def test_construction_sets_fields_and_injectable_client():
  client = _FakeClient()
  agent = ManagedAgent(
      name='mgr',
      agent_id='antigravity-preview-05-2026',
      environment={'type': 'remote'},
      api_client=client,
  )

  assert agent.name == 'mgr'
  assert agent.agent_id == 'antigravity-preview-05-2026'
  assert agent.environment == {'type': 'remote'}
  assert agent.tools == []
  # Injected client is returned without constructing a real genai client.
  assert agent.api_client is client


def test_lazy_client_enterprise_uses_global_location(monkeypatch):
  import google.genai as genai

  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '1')
  captured = {}

  def _fake_client(**kwargs):
    captured.update(kwargs)
    return _FakeClient()

  monkeypatch.setattr(genai, 'Client', _fake_client)

  agent = ManagedAgent(name='mgr', agent_id='agents/a')
  _ = agent.api_client  # triggers lazy construction

  assert captured['enterprise'] is True
  assert captured['location'] == 'global'


def test_lazy_client_dev_api_omits_location(monkeypatch):
  import google.genai as genai

  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '0')
  captured = {}

  def _fake_client(**kwargs):
    captured.update(kwargs)
    return _FakeClient()

  monkeypatch.setattr(genai, 'Client', _fake_client)

  agent = ManagedAgent(name='mgr', agent_id='agents/a')
  _ = agent.api_client  # triggers lazy construction

  assert captured['enterprise'] is False
  assert 'location' not in captured


def test_injected_non_global_enterprise_client_raises():
  client = _FakeClientWithLocation('us-central1', vertexai=True)

  with pytest.raises(ValueError, match='global'):
    ManagedAgent(name='mgr', agent_id='agents/a', api_client=client)


def test_injected_global_enterprise_client_is_accepted():
  client = _FakeClientWithLocation('global', vertexai=True)

  agent = ManagedAgent(name='mgr', agent_id='agents/a', api_client=client)

  assert agent.api_client is client


def test_injected_regional_dev_api_client_is_accepted():
  # Developer API clients have no meaningful location; genai still stamps
  # GOOGLE_CLOUD_LOCATION onto _api_client.location, so a regional value must
  # NOT be rejected for a non-enterprise client.
  client = _FakeClientWithLocation('us-central1', vertexai=False)

  agent = ManagedAgent(name='mgr', agent_id='agents/a', api_client=client)

  assert agent.api_client is client


def test_injected_client_without_location_is_accepted():
  client = _FakeClient()

  agent = ManagedAgent(name='mgr', agent_id='agents/a', api_client=client)

  assert agent.api_client is client


def test_validate_uses_public_vertexai_property():
  # The enterprise decision must come from the PUBLIC `Client.vertexai`
  # property, not the private `_api_client.vertexai`. This client reports
  # enterprise via the public property while its private `_api_client.vertexai`
  # is unset; a non-global location must therefore be rejected.
  class _PublicVertexClient:
    vertexai = True  # public property says enterprise

    def __init__(self):
      self._api_client = _FakeApiClient('us-central1', vertexai=None)

  with pytest.raises(ValueError, match='global'):
    ManagedAgent(
        name='mgr', agent_id='agents/a', api_client=_PublicVertexClient()
    )


def _ctx() -> Any:
  # _resolve_backend_tools only needs an InvocationContext to build a
  # ToolContext; a MagicMock satisfies the built-in tools used here.
  return MagicMock()


def test_resolve_builtin_google_search():
  agent = ManagedAgent(
      name='mgr',
      agent_id='agents/a',
      tools=[google_search],
      api_client=_FakeClient(),
  )

  tool_params = asyncio.run(agent._resolve_backend_tools(_ctx()))

  assert {'type': 'google_search'} in tool_params


def test_resolve_raw_tool_passthrough():
  raw = types.Tool(url_context=types.UrlContext())
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', tools=[raw], api_client=_FakeClient()
  )

  tool_params = asyncio.run(agent._resolve_backend_tools(_ctx()))

  assert {'type': 'url_context'} in tool_params


def test_resolve_rejects_raw_mcp_server():
  raw = types.Tool(
      mcp_servers=[
          types.McpServer(
              name='db',
              streamable_http_transport=types.StreamableHttpTransport(
                  url='https://x'
              ),
          )
      ]
  )
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', tools=[raw], api_client=_FakeClient()
  )

  with pytest.raises(NotImplementedError, match='mcp'):
    asyncio.run(agent._resolve_backend_tools(_ctx()))


def test_resolve_rejects_function_tool():
  def my_fn(x: str) -> str:
    return x

  agent = ManagedAgent(
      name='mgr',
      agent_id='agents/a',
      tools=[FunctionTool(func=my_fn)],
      api_client=_FakeClient(),
  )

  with pytest.raises(NotImplementedError, match='client-executed'):
    asyncio.run(agent._resolve_backend_tools(_ctx()))


def test_resolve_rejects_plain_callable():
  def my_fn(x: str) -> str:
    return x

  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', tools=[my_fn], api_client=_FakeClient()
  )

  with pytest.raises(NotImplementedError, match='client-executed'):
    asyncio.run(agent._resolve_backend_tools(_ctx()))


def test_resolve_rejects_raw_tool_with_function_declarations():
  raw = types.Tool(
      function_declarations=[
          types.FunctionDeclaration(name='my_fn', description='d')
      ]
  )
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', tools=[raw], api_client=_FakeClient()
  )

  with pytest.raises(NotImplementedError, match='client-executed'):
    asyncio.run(agent._resolve_backend_tools(_ctx()))


def test_resolve_rejects_unsupported_raw_tool():
  raw = types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', tools=[raw], api_client=_FakeClient()
  )

  with pytest.raises(NotImplementedError, match='Unsupported raw'):
    asyncio.run(agent._resolve_backend_tools(_ctx()))


def test_resolve_combines_multiple_tools():
  agent = ManagedAgent(
      name='mgr',
      agent_id='agents/a',
      tools=[google_search, types.Tool(url_context=types.UrlContext())],
      api_client=_FakeClient(),
  )

  tool_params = asyncio.run(agent._resolve_backend_tools(_ctx()))

  assert {'type': 'google_search'} in tool_params
  assert {'type': 'url_context'} in tool_params


def test_resolve_empty_tools_returns_empty():
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', api_client=_FakeClient()
  )

  assert asyncio.run(agent._resolve_backend_tools(_ctx())) == []


def test_resolve_passes_managed_agent_flag_and_no_model():
  from google.adk.tools.base_tool import BaseTool

  class _RecordingTool(BaseTool):

    def __init__(self):
      super().__init__(name='rec', description='rec')
      self.captured = {}

    async def process_llm_request(self, *, tool_context, llm_request):
      self.captured['is_managed_agent'] = llm_request._is_managed_agent
      self.captured['model'] = llm_request.model

  rec = _RecordingTool()
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', tools=[rec], api_client=_FakeClient()
  )

  asyncio.run(agent._resolve_backend_tools(_ctx()))

  assert rec.captured['is_managed_agent'] is True
  assert rec.captured['model'] is None


class _RecordingInteractions:

  def __init__(self, responses_per_call):
    self._responses_per_call = list(responses_per_call)
    self.calls = []

  async def create(self, **kwargs):
    self.calls.append(kwargs)
    responses = self._responses_per_call.pop(0)

    class _Iter:

      def __init__(self, items):
        self._it = iter(items)

      def __aiter__(self):
        return self

      async def __anext__(self):
        try:
          return next(self._it)
        except StopIteration as exc:
          raise StopAsyncIteration from exc

    return _Iter(responses)


class _RecordingClient:
  vertexai = False

  def __init__(self, responses_per_call):
    self.aio = MagicMock()
    self.aio.interactions = _RecordingInteractions(responses_per_call)


def _user_ctx(text, *, session_events=None, invocation_id='inv1', branch=None):
  ctx = MagicMock()
  ctx.user_content = genai_types.Content(
      role='user', parts=[genai_types.Part(text=text)]
  )
  ctx.invocation_id = invocation_id
  ctx.branch = branch
  ctx.session.events = session_events or []
  return ctx


def _make_llm_response(text, interaction_id, environment_id):
  return LlmResponse(
      content=genai_types.Content(
          role='model', parts=[genai_types.Part(text=text)]
      ),
      interaction_id=interaction_id,
      environment_id=environment_id,
  )


def _partial_text_response(text):
  return LlmResponse(
      content=genai_types.Content(
          role='model', parts=[genai_types.Part(text=text)]
      ),
      partial=True,
  )


def _final_text_response(text):
  return LlmResponse(
      content=genai_types.Content(
          role='model', parts=[genai_types.Part(text=text)]
      ),
      partial=False,
      turn_complete=True,
  )


def test_run_async_yields_events_with_ids(monkeypatch):
  from google.adk.agents import _managed_agent as mod

  async def _fake_stream(api_client, *, create_kwargs, stream):
    yield _make_llm_response('Hello!', 'int_1', 'env_1')

  monkeypatch.setattr(mod, '_create_interactions', _fake_stream)

  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', api_client=_FakeClient()
  )

  async def _collect():
    out = []
    async for e in agent._run_async_impl(_user_ctx('hi')):
      out.append(e)
    return out

  events = asyncio.run(_collect())
  assert len(events) == 1
  assert events[0].author == 'mgr'
  assert events[0].content.parts[0].text == 'Hello!'
  assert events[0].interaction_id == 'int_1'
  assert events[0].environment_id == 'env_1'


def test_run_async_recovers_previous_state():
  prior = Event(
      author='mgr', interaction_id='int_prev', environment_id='env_prev'
  )
  ctx = _user_ctx('again', session_events=[prior])

  client = _RecordingClient([[]])
  agent = ManagedAgent(name='mgr', agent_id='agents/a', api_client=client)

  asyncio.run(_drain(agent._run_async_impl(ctx)))

  create_kwargs = client.aio.interactions.calls[0]
  assert create_kwargs['previous_interaction_id'] == 'int_prev'
  assert create_kwargs['environment'] == 'env_prev'
  assert create_kwargs['agent'] == 'agents/a'
  assert create_kwargs['stream'] is True
  assert create_kwargs['background'] is True


def test_run_async_forwards_tools_and_agent_config():
  from google.adk.tools import google_search

  client = _RecordingClient([[]])
  agent = ManagedAgent(
      name='mgr',
      agent_id='agents/a',
      tools=[google_search],
      agent_config={'type': 'dynamic'},
      api_client=client,
  )

  asyncio.run(_drain(agent._run_async_impl(_user_ctx('hi'))))

  create_kwargs = client.aio.interactions.calls[0]
  assert {'type': 'google_search'} in create_kwargs['tools']
  assert create_kwargs['agent_config'] == {'type': 'dynamic'}


def test_run_async_sets_background_true():
  client = _RecordingClient([[]])
  agent = ManagedAgent(name='mgr', agent_id='agents/a', api_client=client)

  asyncio.run(_drain(agent._run_async_impl(_user_ctx('hi'))))

  create_kwargs = client.aio.interactions.calls[0]
  assert create_kwargs['background'] is True


def test_run_async_yields_multiple_events_in_order(monkeypatch):
  from google.adk.agents import _managed_agent as mod

  async def _fake_stream(api_client, *, create_kwargs, stream):
    yield _make_llm_response('one', 'int_1', 'env_1')
    yield _make_llm_response('two', 'int_1', 'env_1')

  monkeypatch.setattr(mod, '_create_interactions', _fake_stream)

  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', api_client=_FakeClient()
  )

  events = asyncio.run(_drain_collect(agent._run_async_impl(_user_ctx('hi'))))
  assert [e.content.parts[0].text for e in events] == ['one', 'two']


def test_run_async_error_yields_error_event(monkeypatch):
  from google.adk.agents import _managed_agent as mod

  async def _boom(api_client, *, create_kwargs, stream):
    raise RuntimeError('api exploded')
    yield  # pragma: no cover

  monkeypatch.setattr(mod, '_create_interactions', _boom)

  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', api_client=_FakeClient()
  )

  events = asyncio.run(_drain_collect(agent._run_async_impl(_user_ctx('hi'))))
  assert len(events) == 1
  assert events[0].author == 'mgr'
  assert 'api exploded' in (events[0].error_message or '')
  assert events[0].error_code == 'UNKNOWN_ERROR'
  assert events[0].turn_complete is True


def test_run_async_api_error_surfaces_backend_status_and_message(monkeypatch):
  from google.adk.agents import _managed_agent as mod
  from google.genai import errors

  async def _boom(api_client, *, create_kwargs, stream):
    raise errors.ClientError(
        429,
        {
            'error': {
                'code': 429,
                'status': 'RESOURCE_EXHAUSTED',
                'message': 'Quota exceeded.',
            }
        },
    )
    yield  # pragma: no cover

  monkeypatch.setattr(mod, '_create_interactions', _boom)
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', api_client=_FakeClient()
  )

  events = asyncio.run(_drain_collect(agent._run_async_impl(_user_ctx('hi'))))
  assert len(events) == 1
  assert events[0].author == 'mgr'
  assert events[0].error_code == 'RESOURCE_EXHAUSTED'
  assert events[0].error_message == 'Quota exceeded.'
  assert events[0].turn_complete is True


def test_run_async_uses_self_environment_when_no_prior():
  client = _RecordingClient([[]])
  agent = ManagedAgent(
      name='mgr',
      agent_id='agents/a',
      environment={'type': 'remote'},
      api_client=client,
  )

  asyncio.run(_drain(agent._run_async_impl(_user_ctx('hi'))))

  create_kwargs = client.aio.interactions.calls[0]
  assert create_kwargs['environment'] == {'type': 'remote'}
  assert 'previous_interaction_id' not in create_kwargs


def test_run_async_raises_on_unsupported_tool():
  def my_fn(x: str) -> str:
    return x

  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', tools=[my_fn], api_client=_FakeClient()
  )

  with pytest.raises(NotImplementedError, match='client-executed'):
    asyncio.run(_drain(agent._run_async_impl(_user_ctx('hi'))))


def test_managed_agent_exported_from_package():
  import google.adk.agents as agents_pkg

  assert agents_pkg.ManagedAgent is ManagedAgent
  assert 'ManagedAgent' in agents_pkg.__all__


def test_run_async_non_streaming_suppresses_partials(monkeypatch):
  from google.adk.agents import _managed_agent as mod

  async def _fake_stream(api_client, *, create_kwargs, stream):
    yield _partial_text_response('thinking')
    yield _partial_text_response('searching')
    yield _final_text_response('Final answer.')

  monkeypatch.setattr(mod, '_create_interactions', _fake_stream)
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', api_client=_FakeClient()
  )
  ctx = _user_ctx('hi')
  ctx.run_config.streaming_mode = StreamingMode.NONE

  events = asyncio.run(_drain_collect(agent._run_async_impl(ctx)))

  assert len(events) == 1
  assert events[0].content.parts[0].text == 'Final answer.'
  assert not events[0].partial


def test_run_async_sse_yields_all_partials(monkeypatch):
  from google.adk.agents import _managed_agent as mod

  async def _fake_stream(api_client, *, create_kwargs, stream):
    yield _partial_text_response('thinking')
    yield _partial_text_response('searching')
    yield _final_text_response('Final answer.')

  monkeypatch.setattr(mod, '_create_interactions', _fake_stream)
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', api_client=_FakeClient()
  )
  ctx = _user_ctx('hi')
  ctx.run_config.streaming_mode = StreamingMode.SSE

  events = asyncio.run(_drain_collect(agent._run_async_impl(ctx)))

  assert [e.content.parts[0].text for e in events] == [
      'thinking',
      'searching',
      'Final answer.',
  ]


def test_run_async_non_streaming_surfaces_error_event(monkeypatch):
  from google.adk.agents import _managed_agent as mod

  async def _fake_stream(api_client, *, create_kwargs, stream):
    yield _partial_text_response('thinking')
    yield LlmResponse(
        error_code='UNKNOWN_ERROR', error_message='boom', turn_complete=True
    )

  monkeypatch.setattr(mod, '_create_interactions', _fake_stream)
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', api_client=_FakeClient()
  )
  ctx = _user_ctx('hi')
  ctx.run_config.streaming_mode = StreamingMode.NONE

  events = asyncio.run(_drain_collect(agent._run_async_impl(ctx)))

  assert len(events) == 1
  assert events[0].error_code == 'UNKNOWN_ERROR'
  assert events[0].error_message == 'boom'


def test_run_async_default_run_config_suppresses_partials(monkeypatch):
  from google.adk.agents import _managed_agent as mod

  async def _fake_stream(api_client, *, create_kwargs, stream):
    yield _partial_text_response('thinking')
    yield _final_text_response('Final answer.')

  monkeypatch.setattr(mod, '_create_interactions', _fake_stream)
  agent = ManagedAgent(
      name='mgr', agent_id='agents/a', api_client=_FakeClient()
  )
  ctx = _user_ctx('hi')
  ctx.run_config = RunConfig()  # default streaming_mode == StreamingMode.NONE

  events = asyncio.run(_drain_collect(agent._run_async_impl(ctx)))

  assert len(events) == 1
  assert events[0].content.parts[0].text == 'Final answer.'


def test_run_async_non_streaming_final_event_carries_grounding_and_usage():
  from google.genai.interactions import InteractionCompletedEvent
  from google.genai.interactions import InteractionSseEventInteraction
  from google.genai.interactions import StepDelta
  from google.genai.interactions import Usage

  sse_events = [
      StepDelta(
          event_type='step.delta',
          index=0,
          delta={
              'type': 'google_search_call',
              'arguments': {'queries': ['q1']},
          },
      ),
      StepDelta(
          event_type='step.delta',
          index=0,
          delta={'type': 'text', 'text': 'Final answer.'},
      ),
      InteractionCompletedEvent(
          event_type='interaction.completed',
          interaction=InteractionSseEventInteraction(
              id='int_e2e',
              status='completed',
              steps=[],
              usage=Usage(total_input_tokens=12, total_output_tokens=7),
          ),
      ),
  ]
  client = _RecordingClient([sse_events])
  agent = ManagedAgent(name='mgr', agent_id='agents/a', api_client=client)
  ctx = _user_ctx('hi')
  ctx.run_config.streaming_mode = StreamingMode.NONE

  events = asyncio.run(_drain_collect(agent._run_async_impl(ctx)))

  assert len(events) == 1
  final = events[0]
  assert not final.partial
  assert final.content.parts[-1].text == 'Final answer.'
  assert final.grounding_metadata.web_search_queries == ['q1']
  assert final.usage_metadata.prompt_token_count == 12
  assert final.usage_metadata.candidates_token_count == 7


async def _drain(agen):
  async for _ in agen:
    pass


async def _drain_collect(agen):
  out = []
  async for e in agen:
    out.append(e)
  return out
