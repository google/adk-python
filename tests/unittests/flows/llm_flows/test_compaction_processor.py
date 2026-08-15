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

"""Tests for request-phase token compaction processor."""

from unittest.mock import AsyncMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.flows.llm_flows import compaction
from google.adk.flows.llm_flows import contents
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest


def _create_event(
    *,
    timestamp: float,
    invocation_id: str,
    text: str,
    prompt_token_count: int | None = None,
) -> Event:
  usage_metadata = None
  if prompt_token_count is not None:
    usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt_token_count
    )
  return Event(
      timestamp=timestamp,
      invocation_id=invocation_id,
      author='user',
      content=Content(role='user', parts=[Part(text=text)]),
      usage_metadata=usage_metadata,
  )


def test_single_flow_includes_compaction_before_contents():
  flow = SingleFlow()

  compaction_index = flow.request_processors.index(compaction.request_processor)
  contents_index = flow.request_processors.index(contents.request_processor)

  assert compaction_index < contents_index


@pytest.mark.asyncio
async def test_compaction_request_processor_no_token_config():
  session = Session(app_name='app', user_id='user', id='session', events=[])
  session_service = AsyncMock(spec=BaseSessionService)
  invocation_context = InvocationContext(
      invocation_id='invocation',
      agent=LlmAgent(name='agent'),
      session=session,
      session_service=session_service,
      events_compaction_config=EventsCompactionConfig(
          compaction_interval=2,
          overlap_size=0,
      ),
  )

  llm_request = LlmRequest()
  processor = compaction.CompactionRequestProcessor()

  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  assert not events
  assert not invocation_context.token_compaction_checked
  session_service.append_event.assert_not_called()


@pytest.mark.asyncio
async def test_compaction_request_processor_runs_token_compaction():
  session = Session(
      app_name='app',
      user_id='user',
      id='session',
      events=[
          _create_event(timestamp=1.0, invocation_id='inv1', text='e1'),
          _create_event(timestamp=2.0, invocation_id='inv2', text='e2'),
          _create_event(
              timestamp=3.0,
              invocation_id='inv3',
              text='e3',
              prompt_token_count=100,
          ),
      ],
  )
  session_service = AsyncMock(spec=BaseSessionService)
  mock_summarizer = AsyncMock(spec=LlmEventSummarizer)
  compacted_event = Event(author='compactor', invocation_id=Event.new_id())
  mock_summarizer.maybe_summarize_events.return_value = compacted_event

  invocation_context = InvocationContext(
      invocation_id='invocation',
      agent=LlmAgent(name='agent'),
      session=session,
      session_service=session_service,
      events_compaction_config=EventsCompactionConfig(
          summarizer=mock_summarizer,
          compaction_interval=999,
          overlap_size=0,
          token_threshold=50,
          event_retention_size=1,
      ),
  )

  llm_request = LlmRequest()
  processor = compaction.CompactionRequestProcessor()

  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  assert not events
  assert invocation_context.token_compaction_checked

  compacted_events_arg = mock_summarizer.maybe_summarize_events.call_args[1][
      'events'
  ]
  assert [event.invocation_id for event in compacted_events_arg] == [
      'inv1',
      'inv2',
  ]
  session_service.append_event.assert_called_once_with(
      session=session, event=compacted_event
  )


@pytest.mark.asyncio
async def test_compaction_request_processor_compacts_with_latest_tool_response():
  session = Session(
      app_name='app',
      user_id='user',
      id='session',
      events=[
          _create_event(timestamp=1.0, invocation_id='inv1', text='e1'),
          _create_event(timestamp=2.0, invocation_id='inv2', text='e2'),
          Event(
              timestamp=3.0,
              invocation_id='current-inv',
              author='agent',
              content=Content(
                  role='model',
                  parts=[
                      Part(
                          function_call=types.FunctionCall(
                              id='call-1', name='tool', args={}
                          )
                      )
                  ],
              ),
          ),
          Event(
              timestamp=4.0,
              invocation_id='current-inv',
              author='agent',
              content=Content(
                  role='user',
                  parts=[
                      Part(
                          function_response=types.FunctionResponse(
                              id='call-1',
                              name='tool',
                              response={'result': 'ok'},
                          )
                      )
                  ],
              ),
              usage_metadata=types.GenerateContentResponseUsageMetadata(
                  prompt_token_count=100
              ),
          ),
      ],
  )
  session_service = AsyncMock(spec=BaseSessionService)
  mock_summarizer = AsyncMock(spec=LlmEventSummarizer)
  compacted_event = Event(author='compactor', invocation_id=Event.new_id())
  mock_summarizer.maybe_summarize_events.return_value = compacted_event

  invocation_context = InvocationContext(
      invocation_id='current-inv',
      agent=LlmAgent(name='agent'),
      session=session,
      session_service=session_service,
      events_compaction_config=EventsCompactionConfig(
          summarizer=mock_summarizer,
          compaction_interval=999,
          overlap_size=0,
          token_threshold=50,
          event_retention_size=1,
      ),
  )

  llm_request = LlmRequest()
  processor = compaction.CompactionRequestProcessor()

  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  assert not events
  assert invocation_context.token_compaction_checked
  compacted_events_arg = mock_summarizer.maybe_summarize_events.call_args[1][
      'events'
  ]
  assert [event.invocation_id for event in compacted_events_arg] == [
      'inv1',
      'inv2',
  ]
  session_service.append_event.assert_called_once_with(
      session=session, event=compacted_event
  )


@pytest.mark.asyncio
async def test_compaction_request_processor_can_compact_current_user_event():
  session = Session(
      app_name='app',
      user_id='user',
      id='session',
      events=[
          _create_event(timestamp=1.0, invocation_id='inv1', text='e1'),
          Event(
              timestamp=2.0,
              invocation_id='current-inv',
              author='user',
              content=Content(
                  role='user',
                  parts=[Part(text='latest user message')],
              ),
              usage_metadata=types.GenerateContentResponseUsageMetadata(
                  prompt_token_count=100
              ),
          ),
      ],
  )
  session_service = AsyncMock(spec=BaseSessionService)
  mock_summarizer = AsyncMock(spec=LlmEventSummarizer)
  compacted_event = Event(author='compactor', invocation_id=Event.new_id())
  mock_summarizer.maybe_summarize_events.return_value = compacted_event

  invocation_context = InvocationContext(
      invocation_id='current-inv',
      agent=LlmAgent(name='agent'),
      session=session,
      session_service=session_service,
      events_compaction_config=EventsCompactionConfig(
          summarizer=mock_summarizer,
          compaction_interval=999,
          overlap_size=0,
          token_threshold=50,
          event_retention_size=0,
      ),
  )

  llm_request = LlmRequest()
  processor = compaction.CompactionRequestProcessor()

  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  assert not events
  assert invocation_context.token_compaction_checked
  compacted_events_arg = mock_summarizer.maybe_summarize_events.call_args[1][
      'events'
  ]
  assert [event.invocation_id for event in compacted_events_arg] == [
      'inv1',
      'current-inv',
  ]
  session_service.append_event.assert_called_once_with(
      session=session, event=compacted_event
  )


@pytest.mark.asyncio
async def test_compaction_request_processor_not_marked_when_not_compacted():
  session = Session(
      app_name='app',
      user_id='user',
      id='session',
      events=[
          _create_event(timestamp=1.0, invocation_id='inv1', text='e1'),
          _create_event(
              timestamp=2.0,
              invocation_id='inv2',
              text='e2',
              prompt_token_count=40,
          ),
      ],
  )
  session_service = AsyncMock(spec=BaseSessionService)
  mock_summarizer = AsyncMock(spec=LlmEventSummarizer)
  mock_summarizer.maybe_summarize_events.return_value = Event(
      author='compactor',
      invocation_id=Event.new_id(),
  )

  invocation_context = InvocationContext(
      invocation_id='invocation',
      agent=LlmAgent(name='agent'),
      session=session,
      session_service=session_service,
      events_compaction_config=EventsCompactionConfig(
          summarizer=mock_summarizer,
          compaction_interval=999,
          overlap_size=0,
          token_threshold=50,
          event_retention_size=1,
      ),
  )

  llm_request = LlmRequest()
  processor = compaction.CompactionRequestProcessor()

  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  assert not events
  assert not invocation_context.token_compaction_checked
  mock_summarizer.maybe_summarize_events.assert_not_called()
  session_service.append_event.assert_not_called()
