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

import unittest
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.genai.types import Content
from google.genai.types import FunctionCall
from google.genai.types import FunctionResponse
from google.genai.types import Part
import pytest


@pytest.mark.parametrize(
    'env_variables', ['GOOGLE_AI', 'VERTEX'], indirect=True
)
class TestLlmEventSummarizer(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.mock_llm = AsyncMock(spec=BaseLlm)
    self.mock_llm.model = 'test-model'
    self.compactor = LlmEventSummarizer(llm=self.mock_llm)

  def _create_event(
      self, timestamp: float, text: str, author: str = 'user'
  ) -> Event:
    return Event(
        timestamp=timestamp,
        author=author,
        content=Content(parts=[Part(text=text)]),
    )

  async def test_maybe_compact_events_success(self):
    events = [
        self._create_event(1.0, 'Hello', 'user'),
        self._create_event(2.0, 'Hi there!', 'model'),
    ]
    expected_conversation_history = 'user: Hello\\nmodel: Hi there!'
    expected_prompt = self.compactor._DEFAULT_PROMPT_TEMPLATE.format(
        conversation_history=expected_conversation_history
    )
    mock_llm_response = Mock(content=Content(parts=[Part(text='Summary')]))

    async def async_gen():
      yield mock_llm_response

    self.mock_llm.generate_content_async.return_value = async_gen()

    compacted_event = await self.compactor.maybe_summarize_events(events=events)

    self.assertIsNotNone(compacted_event)
    self.assertEqual(
        compacted_event.actions.compaction.compacted_content.parts[0].text,
        'Summary',
    )
    self.assertEqual(compacted_event.author, 'user')
    self.assertIsNotNone(compacted_event.actions)
    self.assertIsNotNone(compacted_event.actions.compaction)
    self.assertEqual(compacted_event.actions.compaction.start_timestamp, 1.0)
    self.assertEqual(compacted_event.actions.compaction.end_timestamp, 2.0)
    self.assertEqual(
        compacted_event.actions.compaction.compacted_content.parts[0].text,
        'Summary',
    )

    self.mock_llm.generate_content_async.assert_called_once()
    args, kwargs = self.mock_llm.generate_content_async.call_args
    llm_request = args[0]
    self.assertIsInstance(llm_request, LlmRequest)
    self.assertEqual(llm_request.model, 'test-model')
    self.assertEqual(llm_request.contents[0].role, 'user')
    self.assertEqual(llm_request.contents[0].parts[0].text, expected_prompt)
    self.assertFalse(kwargs['stream'])

  async def test_maybe_compact_events_empty_llm_response(self):
    events = [
        self._create_event(1.0, 'Hello', 'user'),
    ]
    mock_llm_response = Mock(content=None)

    async def async_gen():
      yield mock_llm_response

    self.mock_llm.generate_content_async.return_value = async_gen()

    compacted_event = await self.compactor.maybe_summarize_events(events=events)
    self.assertIsNone(compacted_event)

  async def test_maybe_compact_events_empty_input(self):
    compacted_event = await self.compactor.maybe_summarize_events(events=[])
    self.assertIsNone(compacted_event)
    self.mock_llm.generate_content_async.assert_not_called()

  def test_format_events_for_prompt(self):
    events = [
        self._create_event(1.0, 'User says...', 'user'),
        self._create_event(2.0, 'Model replies...', 'model'),
        self._create_event(3.0, 'Another user input', 'user'),
        self._create_event(4.0, 'More model text', 'model'),
        # Event with no content
        Event(timestamp=5.0, author='user'),
        # Event with empty content part
        Event(
            timestamp=6.0,
            author='model',
            content=Content(parts=[Part(text='')]),
        ),
        # Event with function call
        Event(
            timestamp=7.0,
            author='model',
            content=Content(
                parts=[
                    Part(
                        function_call=FunctionCall(
                            id='call_1', name='tool', args={}
                        )
                    )
                ]
            ),
        ),
        # Event with function response
        Event(
            timestamp=8.0,
            author='model',
            content=Content(
                parts=[
                    Part(
                        function_response=FunctionResponse(
                            id='call_1',
                            name='tool',
                            response={'result': 'done'},
                        )
                    )
                ]
            ),
        ),
    ]
    expected_formatted_history = (
        'user: User says...\\nmodel: Model replies...\\nuser: Another user'
        ' input\\nmodel: More model text'
    )
    formatted_history = self.compactor._format_events_for_prompt(events)
    self.assertEqual(formatted_history, expected_formatted_history)
