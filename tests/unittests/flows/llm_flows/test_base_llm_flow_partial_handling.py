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

from typing import AsyncGenerator

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import _SingleAfterModelCallback
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import _handle_after_model_callback
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
import pytest

from ... import testing_utils


class BaseLlmFlowForTesting(BaseLlmFlow):
  """Test implementation of BaseLlmFlow for testing purposes."""

  pass


@pytest.mark.asyncio
async def test_run_async_breaks_on_partial_event():
  """Test that run_async breaks when the last event is partial."""
  # Create a mock model that returns partial responses
  partial_response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Partial response')]
      ),
      partial=True,
  )

  mock_model = testing_utils.MockModel.create(responses=[partial_response])

  agent = Agent(name='test_agent', model=mock_model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()
  events = []

  # Collect events from the flow
  async for event in flow.run_async(invocation_context):
    events.append(event)

  # Should have one event (the partial response)
  assert len(events) == 1
  assert events[0].partial is True
  assert events[0].content.parts[0].text == 'Partial response'


@pytest.mark.asyncio
async def test_run_async_breaks_on_final_response():
  """Test that run_async breaks when the last event is a final response."""
  # Create a mock model that returns a final response
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Final response')]
      ),
      partial=False,
      error_code=types.FinishReason.STOP,
  )

  mock_model = testing_utils.MockModel.create(responses=[final_response])

  agent = Agent(name='test_agent', model=mock_model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()
  events = []

  # Collect events from the flow
  async for event in flow.run_async(invocation_context):
    events.append(event)

  # Should have one event (the final response)
  assert len(events) == 1
  assert events[0].partial is False
  assert events[0].content.parts[0].text == 'Final response'


@pytest.mark.asyncio
async def test_run_async_breaks_on_no_last_event():
  """Test that run_async breaks when there is no last event."""
  # Create a mock model that returns an empty response (no content)
  empty_response = LlmResponse(content=None, partial=False)

  mock_model = testing_utils.MockModel.create(responses=[empty_response])

  agent = Agent(name='test_agent', model=mock_model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()
  events = []

  # Collect events from the flow
  async for event in flow.run_async(invocation_context):
    events.append(event)

  # Should have no events because empty responses are filtered out
  assert len(events) == 0


@pytest.mark.asyncio
async def test_run_async_breaks_on_first_partial_response():
  """Test run_async breaks on the first partial response."""
  # Create responses with mixed partial states
  partial_response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Partial response')]
      ),
      partial=True,
  )

  # These won't be reached because the flow breaks on the first partial
  non_partial_response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Non-partial response')],
      ),
      partial=False,
  )

  final_partial_response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Final partial response')],
      ),
      partial=True,
  )

  mock_model = testing_utils.MockModel.create(
      responses=[partial_response, non_partial_response, final_partial_response]
  )

  agent = Agent(name='test_agent', model=mock_model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()
  events = []

  # Collect events from the flow
  async for event in flow.run_async(invocation_context):
    events.append(event)

  # Should have only one event, breaking on the first partial response
  assert len(events) == 1
  assert events[0].partial is True
  assert events[0].content.parts[0].text == 'Partial response'


def _text_response(
    text: str,
    *,
    partial: bool | None = None,
    turn_complete: bool | None = None,
) -> LlmResponse:
  """Builds a single-part text response with the given streaming flags."""
  return LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text=text)]
      ),
      partial=partial,
      turn_complete=turn_complete,
  )


def _text_response_without_streaming_fields(text: str) -> LlmResponse:
  """Builds a text response that omits partial and turn_complete."""
  return LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text=text)]
      )
  )


def _rebuilding_callback(
    replacement: LlmResponse,
) -> _SingleAfterModelCallback:
  """Returns a callback that rebuilds the response, dropping control fields."""

  async def _callback(
      callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse:
    del callback_context, llm_response
    return replacement

  return _callback


class _ReplacementPlugin(BasePlugin):
  """Plugin that replaces the response with a rebuilt response."""

  def __init__(self, replacement: LlmResponse):
    super().__init__(name='replacement_plugin')
    self._replacement = replacement

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse:
    del callback_context, llm_response
    return self._replacement


async def _run_handle_after_model_callback(
    agent: Agent,
    llm_response: LlmResponse,
    *,
    plugins: list[BasePlugin] | None = None,
) -> LlmResponse | None:
  """Runs the after-model callback chain for the agent over one response."""
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, plugins=plugins or []
  )
  event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
  )
  return await _handle_after_model_callback(
      invocation_context, llm_response, event
  )


@pytest.mark.asyncio
async def test_after_model_callback_replacement_inherits_partial():
  """A rebuilt replacement keeps the delta's partial flag."""
  replacement = _text_response('Hello ')
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('Hello ', partial=True)
  )

  assert result.partial is True
  assert result.content.parts[0].text == 'Hello '
  # The callback's object is not mutated; inheritance returns a copy.
  assert replacement.partial is None


@pytest.mark.asyncio
async def test_after_model_callback_replacement_inherits_turn_complete():
  """A rebuilt replacement keeps turn_complete from the replaced response."""
  replacement = _text_response('done')
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('done', turn_complete=True)
  )

  assert result.turn_complete is True


@pytest.mark.asyncio
async def test_after_model_callback_replacement_explicit_partial_respected():
  """An explicitly set partial=False on the replacement is not overridden."""
  agent = Agent(
      name='test_agent',
      after_model_callback=[
          _rebuilding_callback(_text_response('final', partial=False))
      ],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('final', partial=True)
  )

  assert result.partial is False


@pytest.mark.asyncio
async def test_after_model_callback_replacement_explicit_turn_complete_respected():
  """An explicitly set turn_complete=False is not overridden."""
  agent = Agent(
      name='test_agent',
      after_model_callback=[
          _rebuilding_callback(_text_response('final', turn_complete=False))
      ],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('final', turn_complete=True)
  )

  assert result.turn_complete is False


@pytest.mark.parametrize('field', ['partial', 'turn_complete'])
@pytest.mark.asyncio
async def test_after_model_callback_explicit_none_does_not_clear_lifecycle(
    field: str,
):
  """Explicit None does not clear framework-owned lifecycle fields."""
  replacement = _text_response('scrubbed', **{field: None})
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('original', **{field: True})
  )

  assert getattr(result, field) is True
  # Inheritance is copy-on-write; the callback's object remains unchanged.
  assert getattr(replacement, field) is None
  assert result is not replacement


@pytest.mark.parametrize('field', ['partial', 'turn_complete'])
@pytest.mark.asyncio
async def test_after_model_callback_omitted_lifecycle_field_is_inherited(
    field: str,
):
  """An omitted lifecycle field is filled from the replaced response."""
  replacement = _text_response_without_streaming_fields('scrubbed')
  assert field not in replacement.model_fields_set
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('original', **{field: True})
  )

  assert getattr(result, field) is True
  assert result is not replacement


@pytest.mark.asyncio
async def test_after_model_callback_replacement_without_streaming_keeps_identity():
  """Non-streaming replacements pass through untouched (no copy)."""
  replacement = _text_response('final')
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('final')
  )

  assert result is replacement


@pytest.mark.asyncio
async def test_after_model_callback_returning_its_argument_keeps_identity():
  """A callback that hands back what it was given has nothing to inherit."""
  llm_response = _text_response('Hello ', partial=True)
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(llm_response)],
  )

  result = await _run_handle_after_model_callback(agent, llm_response)

  assert result is llm_response


class _ResponseLookalike:
  """A response object without the streaming fields, as some models yield."""

  def __init__(self, text: str):
    self.content = types.Content(
        role='model', parts=[types.Part.from_text(text=text)]
    )
    self.partial = False


@pytest.mark.asyncio
async def test_after_model_callback_response_without_streaming_fields():
  """A response that is not an LlmResponse leaves the replacement alone."""
  replacement = _text_response('final')
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )

  result = await _run_handle_after_model_callback(
      agent, _ResponseLookalike('final')
  )

  assert result is replacement


class _StreamingFakeModel(BaseLlm):
  """Yields two partial deltas then the aggregated final response."""

  model: str = 'fake-streaming'

  @classmethod
  def create(cls) -> _StreamingFakeModel:
    return cls(model='fake-streaming')

  @classmethod
  def supported_models(cls) -> list[str]:
    return ['.*']

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    deltas = ['Hello ', 'world.']
    if stream:
      for delta in deltas:
        yield _text_response(delta, partial=True)
    yield _text_response(''.join(deltas))


@pytest.mark.asyncio
async def test_run_async_sse_rebuilt_responses_stay_partial():
  """End-to-end: rebuilding callbacks must not flip SSE deltas to final."""
  agent = Agent(
      name='test_agent',
      model=_StreamingFakeModel.create(),
      after_model_callback=[_rebuilding_callback(_text_response('scrubbed'))],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test message',
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  )

  flow = BaseLlmFlowForTesting()
  events = []
  async for event in flow.run_async(invocation_context):
    events.append(event)

  assert [event.partial for event in events] == [True, True, None]
  # The callback ran on every response (content replaced), but the streaming
  # semantics of the originals survived.
  assert [event.content.parts[0].text for event in events] == [
      'scrubbed',
      'scrubbed',
      'scrubbed',
  ]


_USAGE = types.GenerateContentResponseUsageMetadata(
    prompt_token_count=120,
    candidates_token_count=8,
    total_token_count=128,
)


@pytest.mark.asyncio
async def test_after_model_callback_replacement_inherits_usage_metadata():
  """A rebuilt replacement keeps the model call's token accounting."""
  replacement = _text_response('scrubbed')
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )
  original = _text_response('Hello world.')
  original.usage_metadata = _USAGE

  result = await _run_handle_after_model_callback(agent, original)

  assert result.usage_metadata == _USAGE
  assert result.content.parts[0].text == 'scrubbed'
  # The callback's object is not mutated; inheritance returns a copy.
  assert replacement.usage_metadata is None


@pytest.mark.asyncio
async def test_after_model_callback_replacement_explicit_usage_respected():
  """A replacement that reports its own usage is not overridden."""
  own_usage = types.GenerateContentResponseUsageMetadata(
      prompt_token_count=1, candidates_token_count=1, total_token_count=2
  )
  replacement = _text_response('scrubbed')
  replacement.usage_metadata = own_usage
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )
  original = _text_response('Hello world.')
  original.usage_metadata = _USAGE

  result = await _run_handle_after_model_callback(agent, original)

  assert result.usage_metadata == own_usage


@pytest.mark.asyncio
async def test_after_model_callback_default_none_constructor_preserves_usage():
  """Regression guard for #7451: a callback helper that forwards None by default
  must not erase the original model call's usage."""

  # Mirrors #7451's `_resp(text, partial=None, usage=None)` helper, which
  # always passes usage_metadata explicitly, even when the caller omits it.
  def _resp(text, partial=None, usage=None):
    return LlmResponse(
        content=types.Content(
            role='model', parts=[types.Part.from_text(text=text)]
        ),
        partial=partial,
        usage_metadata=usage,
    )

  replacement = _resp('scrubbed')
  assert 'usage_metadata' in replacement.model_fields_set
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )
  original = _text_response('Hello world.')
  original.usage_metadata = _USAGE

  result = await _run_handle_after_model_callback(agent, original)

  assert result.usage_metadata == _USAGE
  assert result.content.parts[0].text == 'scrubbed'


@pytest.mark.asyncio
async def test_plugin_omitted_fields_inherit_lifecycle_and_usage():
  """Plugin replacement inherits omitted lifecycle fields and usage."""
  replacement = _text_response_without_streaming_fields('scrubbed')
  plugin = _ReplacementPlugin(replacement)
  agent = Agent(name='test_agent')
  original = _text_response('Hello world.', partial=True, turn_complete=True)
  original.usage_metadata = _USAGE

  result = await _run_handle_after_model_callback(
      agent, original, plugins=[plugin]
  )

  assert result.partial is True
  assert result.turn_complete is True
  assert result.usage_metadata == _USAGE
  assert result.content.parts[0].text == 'scrubbed'
  assert replacement.partial is None
  assert replacement.usage_metadata is None


@pytest.mark.asyncio
async def test_plugin_default_none_constructor_preserves_lifecycle_and_usage():
  """#7451 guard: a plugin helper forwarding None by default must not erase
  the original lifecycle markers or usage."""

  def _resp(text, partial=None, usage=None):
    return LlmResponse(
        content=types.Content(
            role='model', parts=[types.Part.from_text(text=text)]
        ),
        partial=partial,
        usage_metadata=usage,
    )

  plugin = _ReplacementPlugin(_resp('scrubbed'))
  agent = Agent(name='test_agent')
  original = _text_response('Hello world.', partial=True, turn_complete=True)
  original.usage_metadata = _USAGE

  result = await _run_handle_after_model_callback(
      agent, original, plugins=[plugin]
  )

  assert result.partial is True
  assert result.turn_complete is True
  assert result.usage_metadata == _USAGE
  assert result.content.parts[0].text == 'scrubbed'


class _StreamingFakeModelWithUsage(_StreamingFakeModel):
  """Like _StreamingFakeModel, but reports usage on the final response only."""

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    deltas = ['Hello ', 'world.']
    if stream:
      for delta in deltas:
        yield _text_response(delta, partial=True)
    final = _text_response(''.join(deltas))
    final.usage_metadata = _USAGE
    yield final


@pytest.mark.asyncio
async def test_run_async_sse_rebuilt_final_keeps_usage_metadata():
  """End-to-end: the rebuilt final event still carries the call's usage."""
  agent = Agent(
      name='test_agent',
      model=_StreamingFakeModelWithUsage(model='fake-streaming'),
      after_model_callback=[_rebuilding_callback(_text_response('scrubbed'))],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test message',
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  )

  flow = BaseLlmFlowForTesting()
  events = []
  async for event in flow.run_async(invocation_context):
    events.append(event)

  assert [event.partial for event in events] == [True, True, None]
  assert [event.usage_metadata for event in events] == [None, None, _USAGE]


@pytest.mark.asyncio
async def test_runner_persists_only_final_sse_event_with_explicit_partial_none():
  """Explicit partial=None cannot make partial deltas persist as finals."""
  agent = Agent(
      name='test_agent',
      model=_StreamingFakeModel.create(),
      after_model_callback=[
          _rebuilding_callback(_text_response('scrubbed', partial=None))
      ],
  )
  harness = testing_utils.InMemoryRunner(agent)

  events = []
  async for event in harness.runner.run_async(
      user_id=harness.session.user_id,
      session_id=harness.session.id,
      new_message=testing_utils.get_user_content('test message'),
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  ):
    events.append(event)

  assert [event.partial for event in events] == [True, True, None]
  persisted_model_events = [
      event for event in harness.session.events if event.author == agent.name
  ]
  assert len(persisted_model_events) == 1
  assert persisted_model_events[0].partial is None
  assert persisted_model_events[0].content.parts[0].text == 'scrubbed'


_TOOL_CALLS: list[str] = []


def record_tool() -> dict[str, str]:
  """Tool used to verify partial function calls run only once."""
  _TOOL_CALLS.append('record_tool')
  return {'result': 'ok'}


def _function_call_response(partial: bool | None = None) -> LlmResponse:
  """Builds a response whose only content is a record_tool call."""
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(name='record_tool', args={})
              )
          ],
      ),
      partial=partial,
  )


class _ToolStreamingFakeModel(BaseLlm):
  """Streams function-call chunks, then answers the result with text."""

  model: str = 'fake-tool-streaming'
  turn_index: int = 0

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del llm_request, stream
    self.turn_index += 1
    if self.turn_index == 1:
      yield _function_call_response(partial=True)
      yield _function_call_response(partial=True)
      yield _function_call_response(partial=None)
    else:
      yield _text_response('done')


@pytest.mark.asyncio
async def test_explicit_partial_none_replacement_does_not_execute_partial_tool():
  """Partial function-call deltas stay partial after callback replacement."""
  _TOOL_CALLS.clear()

  async def _rebuild_function_call(
      callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse | None:
    del callback_context
    if not llm_response.get_function_calls():
      return None
    return _function_call_response(partial=None)

  agent = Agent(
      name='test_agent',
      model=_ToolStreamingFakeModel(model='fake-tool-streaming'),
      tools=[record_tool],
      after_model_callback=[_rebuild_function_call],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='call the tool',
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(invocation_context):
    events.append(event)

  # Only the streamed model-authored function-call chunks carry the
  # record_tool call; the post-tool text response is also model-authored but
  # has no function call. Assert the partial lifecycle for those chunks only.
  function_call_events = [
      _event
      for _event in events
      if _event.author == agent.name
      and any(
          function_call.name == 'record_tool'
          for function_call in _event.get_function_calls()
      )
  ]
  assert [event.partial for event in function_call_events] == [True, True, None]
  assert _TOOL_CALLS == ['record_tool']
