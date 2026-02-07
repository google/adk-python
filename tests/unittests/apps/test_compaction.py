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

from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.compaction import _run_compaction_for_sliding_window
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.flows.llm_flows import contents
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
from google.genai.types import Content
from google.genai.types import Part
import pytest


@pytest.mark.parametrize(
    'env_variables', ['GOOGLE_AI', 'VERTEX'], indirect=True
)
class TestCompaction(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.mock_session_service = AsyncMock(spec=BaseSessionService)
    self.mock_compactor = AsyncMock(spec=LlmEventSummarizer)

  def _create_event(
      self, timestamp: float, invocation_id: str, text: str
  ) -> Event:
    return Event(
        timestamp=timestamp,
        invocation_id=invocation_id,
        author='user',
        content=Content(role='user', parts=[Part(text=text)]),
    )

  def _create_compacted_event(
      self, start_ts: float, end_ts: float, summary_text: str
  ) -> Event:
    compaction = EventCompaction(
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        compacted_content=Content(
            role='model', parts=[Part(text=summary_text)]
        ),
    )
    return Event(
        timestamp=end_ts,
        author='compactor',
        content=compaction.compacted_content,
        actions=EventActions(compaction=compaction),
        invocation_id=Event.new_id(),
    )

  async def test_run_compaction_for_sliding_window_no_events(self):
    app = App(name='test', root_agent=Mock(spec=BaseAgent))
    session = Session(app_name='test', user_id='u1', id='s1', events=[])
    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )
    self.mock_compactor.maybe_summarize_events.assert_not_called()
    self.mock_session_service.append_event.assert_not_called()

  async def test_run_compaction_for_sliding_window_not_enough_new_invocations(
      self,
  ):
    app = App(
        name='test',
        root_agent=Mock(spec=BaseAgent),
        events_compaction_config=EventsCompactionConfig(
            summarizer=self.mock_compactor,
            compaction_interval=3,
            overlap_size=1,
        ),
    )
    # Only two new invocations ('inv1', 'inv2'), less than compaction_interval=3.
    session = Session(
        app_name='test',
        user_id='u1',
        id='s1',
        events=[
            self._create_event(1.0, 'inv1', 'e1'),
            self._create_event(2.0, 'inv2', 'e2'),
        ],
    )
    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )
    self.mock_compactor.maybe_summarize_events.assert_not_called()
    self.mock_session_service.append_event.assert_not_called()

  async def test_run_compaction_for_sliding_window_first_compaction(self):
    app = App(
        name='test',
        root_agent=Mock(spec=BaseAgent),
        events_compaction_config=EventsCompactionConfig(
            summarizer=self.mock_compactor,
            compaction_interval=2,
            overlap_size=1,
        ),
    )
    events = [
        self._create_event(1.0, 'inv1', 'e1'),
        self._create_event(2.0, 'inv2', 'e2'),
        self._create_event(3.0, 'inv3', 'e3'),
        self._create_event(4.0, 'inv4', 'e4'),
    ]
    session = Session(app_name='test', user_id='u1', id='s1', events=events)

    mock_compacted_event = self._create_compacted_event(
        1.0, 4.0, 'Summary inv1-inv4'
    )
    self.mock_compactor.maybe_summarize_events.return_value = (
        mock_compacted_event
    )

    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )

    # Expected events to compact: inv1, inv2, inv3, inv4
    compacted_events_arg = self.mock_compactor.maybe_summarize_events.call_args[
        1
    ]['events']
    self.assertEqual(
        [e.invocation_id for e in compacted_events_arg],
        ['inv1', 'inv2', 'inv3', 'inv4'],
    )
    self.mock_session_service.append_event.assert_called_once_with(
        session=session, event=mock_compacted_event
    )

  async def test_run_compaction_for_sliding_window_with_overlap(self):
    app = App(
        name='test',
        root_agent=Mock(spec=BaseAgent),
        events_compaction_config=EventsCompactionConfig(
            summarizer=self.mock_compactor,
            compaction_interval=2,
            overlap_size=1,
        ),
    )
    # inv1-inv2 are already compacted. Last compacted end timestamp is 2.0.
    initial_events = [
        self._create_event(1.0, 'inv1', 'e1'),
        self._create_event(2.0, 'inv2', 'e2'),
        self._create_compacted_event(1.0, 2.0, 'Summary inv1-inv2'),
    ]
    # Add new invocations inv3, inv4, inv5
    new_events = [
        self._create_event(3.0, 'inv3', 'e3'),
        self._create_event(4.0, 'inv4', 'e4'),
        self._create_event(5.0, 'inv5', 'e5'),
    ]
    session = Session(
        app_name='test',
        user_id='u1',
        id='s1',
        events=initial_events + new_events,
    )

    mock_compacted_event = self._create_compacted_event(
        2.0, 5.0, 'Summary inv2-inv5'
    )
    self.mock_compactor.maybe_summarize_events.return_value = (
        mock_compacted_event
    )

    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )

    # New invocations are inv3, inv4, inv5 (3 new) > threshold (2).
    # Overlap size is 1, so start from 1 inv before inv3, which is inv2.
    # Compact range: inv2 to inv5.
    compacted_events_arg = self.mock_compactor.maybe_summarize_events.call_args[
        1
    ]['events']
    self.assertEqual(
        [e.invocation_id for e in compacted_events_arg],
        ['inv2', 'inv3', 'inv4', 'inv5'],
    )
    self.mock_session_service.append_event.assert_called_once_with(
        session=session, event=mock_compacted_event
    )

  async def test_run_compaction_for_sliding_window_no_compaction_event_returned(
      self,
  ):
    app = App(
        name='test',
        root_agent=Mock(spec=BaseAgent),
        events_compaction_config=EventsCompactionConfig(
            summarizer=self.mock_compactor,
            compaction_interval=1,
            overlap_size=0,
        ),
    )
    events = [self._create_event(1.0, 'inv1', 'e1')]
    session = Session(app_name='test', user_id='u1', id='s1', events=events)

    self.mock_compactor.maybe_summarize_events.return_value = None

    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )

    self.mock_compactor.maybe_summarize_events.assert_called_once()
    self.mock_session_service.append_event.assert_not_called()

  def test_get_contents_with_multiple_compactions(self):

    # Event timestamps: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
    # Compaction 1: covers 1.0 to 4.0 (summary at 4.0)
    # Compaction 2: covers 6.0 to 9.0 (summary at 9.0)
    events = [
        self._create_event(1.0, 'inv1', 'Event 1'),
        self._create_event(2.0, 'inv2', 'Event 2'),
        self._create_event(3.0, 'inv3', 'Event 3'),
        self._create_event(4.0, 'inv4', 'Event 4'),
        self._create_compacted_event(1.0, 4.0, 'Summary 1-4'),
        self._create_event(5.0, 'inv5', 'Event 5'),
        self._create_event(6.0, 'inv6', 'Event 6'),
        self._create_event(7.0, 'inv7', 'Event 7'),
        self._create_event(8.0, 'inv8', 'Event 8'),
        self._create_event(9.0, 'inv9', 'Event 9'),
        self._create_compacted_event(6.0, 9.0, 'Summary 6-9'),
        self._create_event(10.0, 'inv10', 'Event 10'),
    ]

    result_contents = contents._get_contents(None, events)

    # Expected contents:
    # Summary 1-4 (at timestamp 4.0)
    # Event 5 (at timestamp 5.0)
    # Summary 6-9 (at timestamp 9.0)
    # Event 10 (at timestamp 10.0)
    expected_texts = [
        'Summary 1-4',
        'Event 5',
        'Summary 6-9',
        'Event 10',
    ]
    actual_texts = [c.parts[0].text for c in result_contents]
    self.assertEqual(actual_texts, expected_texts)
    # Verify timestamps are in order

  def test_get_contents_no_compaction(self):

    events = [
        self._create_event(1.0, 'inv1', 'Event 1'),
        self._create_event(2.0, 'inv2', 'Event 2'),
        self._create_event(3.0, 'inv3', 'Event 3'),
    ]

    result_contents = contents._get_contents(None, events)
    expected_texts = ['Event 1', 'Event 2', 'Event 3']
    actual_texts = [c.parts[0].text for c in result_contents]
    self.assertEqual(actual_texts, expected_texts)

  def test_get_contents_single_compaction_at_start(self):

    events = [
        self._create_event(1.0, 'inv1', 'Event 1'),
        self._create_event(2.0, 'inv2', 'Event 2'),
        self._create_compacted_event(1.0, 2.0, 'Summary 1-2'),
        self._create_event(3.0, 'inv3', 'Event 3'),
    ]

    result_contents = contents._get_contents(None, events)
    expected_texts = ['Summary 1-2', 'Event 3']
    actual_texts = [c.parts[0].text for c in result_contents]
    self.assertEqual(actual_texts, expected_texts)

  def test_get_contents_single_compaction_in_middle(self):

    events = [
        self._create_event(1.0, 'inv1', 'Event 1'),
        self._create_event(2.0, 'inv2', 'Event 2'),
        self._create_compacted_event(1.0, 2.0, 'Summary 1-2'),
        self._create_event(3.0, 'inv3', 'Event 3'),
        self._create_event(4.0, 'inv4', 'Event 4'),
        self._create_compacted_event(3.0, 4.0, 'Summary 3-4'),
        self._create_event(5.0, 'inv5', 'Event 5'),
    ]

    result_contents = contents._get_contents(None, events)
    expected_texts = ['Summary 1-2', 'Summary 3-4', 'Event 5']
    actual_texts = [c.parts[0].text for c in result_contents]
    self.assertEqual(actual_texts, expected_texts)

  def test_get_contents_compaction_at_end(self):

    events = [
        self._create_event(1.0, 'inv1', 'Event 1'),
        self._create_event(2.0, 'inv2', 'Event 2'),
        self._create_event(3.0, 'inv3', 'Event 3'),
        self._create_compacted_event(2.0, 3.0, 'Summary 2-3'),
    ]

    result_contents = contents._get_contents(None, events)
    expected_texts = ['Event 1', 'Summary 2-3']
    actual_texts = [c.parts[0].text for c in result_contents]
    self.assertEqual(actual_texts, expected_texts)

  def test_get_contents_compaction_at_beginning(self):

    events = [
        self._create_compacted_event(1.0, 2.0, 'Summary 1-2'),
        self._create_event(3.0, 'inv3', 'Event 3'),
        self._create_event(4.0, 'inv4', 'Event 4'),
    ]

    result_contents = contents._get_contents(None, events)
    expected_texts = ['Summary 1-2', 'Event 3', 'Event 4']
    actual_texts = [c.parts[0].text for c in result_contents]
    self.assertEqual(actual_texts, expected_texts)
