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
import google.adk.apps.compaction as compaction_module
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.flows.llm_flows import contents
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
from pydantic import ValidationError
import pytest


@pytest.mark.parametrize(
    'env_variables', ['GOOGLE_AI', 'VERTEX'], indirect=True
)
class TestCompaction(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.mock_session_service = AsyncMock(spec=BaseSessionService)
    self.mock_compactor = AsyncMock(spec=LlmEventSummarizer)

  def _create_event(
      self,
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

  def _create_compacted_event(
      self,
      start_ts: float,
      end_ts: float,
      summary_text: str,
      appended_ts: float | None = None,
  ) -> Event:
    compaction = EventCompaction(
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        compacted_content=Content(
            role='model', parts=[Part(text=summary_text)]
        ),
    )
    return Event(
        timestamp=appended_ts if appended_ts is not None else end_ts,
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

  def test_events_compaction_config_accepts_token_fields(self):
    config = EventsCompactionConfig(
        compaction_interval=2,
        overlap_size=1,
        token_threshold=50_000,
        event_retention_size=5,
    )
    self.assertEqual(config.token_threshold, 50_000)
    self.assertEqual(config.event_retention_size, 5)

  def test_events_compaction_config_rejects_partial_token_fields(
      self,
  ):
    with pytest.raises(ValidationError):
      EventsCompactionConfig(
          compaction_interval=2,
          overlap_size=1,
          token_threshold=50_000,
      )

  def test_latest_prompt_token_count_fallback_applies_compaction(self):
    events = [
        self._create_event(1.0, 'inv1', 'a' * 40),
        self._create_event(2.0, 'inv2', 'b' * 40),
        self._create_compacted_event(1.0, 2.0, 'S'),
        self._create_event(3.0, 'inv3', 'c' * 20),
    ]

    estimated_token_count = compaction_module._latest_prompt_token_count(events)

    # Visible text after compaction is: 'S' + ('c' * 20) = 21 chars.
    self.assertEqual(estimated_token_count, 21 // 4)

  async def test_run_compaction_for_token_threshold_keeps_retention_events(
      self,
  ):
    app = App(
        name='test',
        root_agent=Mock(spec=BaseAgent),
        events_compaction_config=EventsCompactionConfig(
            summarizer=self.mock_compactor,
            compaction_interval=999,
            overlap_size=0,
            token_threshold=50,
            event_retention_size=2,
        ),
    )
    session = Session(
        app_name='test',
        user_id='u1',
        id='s1',
        events=[
            self._create_event(1.0, 'inv1', 'e1'),
            self._create_event(2.0, 'inv2', 'e2'),
            self._create_event(3.0, 'inv3', 'e3'),
            self._create_event(4.0, 'inv4', 'e4'),
            self._create_event(5.0, 'inv5', 'e5', prompt_token_count=100),
        ],
    )

    mock_compacted_event = self._create_compacted_event(
        1.0, 3.0, 'Summary inv1-inv3'
    )
    self.mock_compactor.maybe_summarize_events.return_value = (
        mock_compacted_event
    )

    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )

    compacted_events_arg = self.mock_compactor.maybe_summarize_events.call_args[
        1
    ]['events']
    self.assertEqual(
        [e.invocation_id for e in compacted_events_arg],
        ['inv1', 'inv2', 'inv3'],
    )
    self.mock_session_service.append_event.assert_called_once_with(
        session=session, event=mock_compacted_event
    )

  async def test_run_compaction_for_token_threshold_seeds_previous_compaction(
      self,
  ):
    app = App(
        name='test',
        root_agent=Mock(spec=BaseAgent),
        events_compaction_config=EventsCompactionConfig(
            summarizer=self.mock_compactor,
            compaction_interval=999,
            overlap_size=0,
            token_threshold=50,
            event_retention_size=2,
        ),
    )
    session = Session(
        app_name='test',
        user_id='u1',
        id='s1',
        events=[
            self._create_event(1.0, 'inv1', 'e1'),
            self._create_event(2.0, 'inv2', 'e2'),
            self._create_compacted_event(1.0, 2.0, 'Summary 1-2'),
            self._create_event(3.0, 'inv3', 'e3'),
            self._create_event(4.0, 'inv4', 'e4'),
            self._create_event(5.0, 'inv5', 'e5'),
            self._create_event(6.0, 'inv6', 'e6', prompt_token_count=100),
        ],
    )

    mock_compacted_event = self._create_compacted_event(1.0, 4.0, 'Summary 1-4')
    self.mock_compactor.maybe_summarize_events.return_value = (
        mock_compacted_event
    )

    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )

    compacted_events_arg = self.mock_compactor.maybe_summarize_events.call_args[
        1
    ]['events']
    self.assertEqual(
        [e.content.parts[0].text for e in compacted_events_arg],
        ['Summary 1-2', 'e3', 'e4'],
    )
    self.assertEqual(compacted_events_arg[0].timestamp, 1.0)
    self.assertEqual(
        [e.invocation_id for e in compacted_events_arg[1:]],
        ['inv3', 'inv4'],
    )
    self.mock_session_service.append_event.assert_called_once_with(
        session=session, event=mock_compacted_event
    )

  async def test_run_compaction_for_token_threshold_with_zero_retention(
      self,
  ):
    app = App(
        name='test',
        root_agent=Mock(spec=BaseAgent),
        events_compaction_config=EventsCompactionConfig(
            summarizer=self.mock_compactor,
            compaction_interval=999,
            overlap_size=0,
            token_threshold=50,
            event_retention_size=0,
        ),
    )
    session = Session(
        app_name='test',
        user_id='u1',
        id='s1',
        events=[
            self._create_event(1.0, 'inv1', 'e1'),
            self._create_event(2.0, 'inv2', 'e2'),
            self._create_event(3.0, 'inv3', 'e3', prompt_token_count=100),
        ],
    )

    mock_compacted_event = self._create_compacted_event(
        1.0, 3.0, 'Summary inv1-inv3'
    )
    self.mock_compactor.maybe_summarize_events.return_value = (
        mock_compacted_event
    )

    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )

    compacted_events_arg = self.mock_compactor.maybe_summarize_events.call_args[
        1
    ]['events']
    self.assertEqual(
        [e.invocation_id for e in compacted_events_arg],
        ['inv1', 'inv2', 'inv3'],
    )
    self.mock_session_service.append_event.assert_called_once_with(
        session=session, event=mock_compacted_event
    )

  async def test_run_compaction_for_token_threshold_with_retention_and_overlap(
      self,
  ):
    app = App(
        name='test',
        root_agent=Mock(spec=BaseAgent),
        events_compaction_config=EventsCompactionConfig(
            summarizer=self.mock_compactor,
            compaction_interval=999,
            overlap_size=0,
            token_threshold=50,
            event_retention_size=3,
        ),
    )
    session = Session(
        app_name='test',
        user_id='u1',
        id='s1',
        events=[
            self._create_event(1.0, 'inv1', 'e1'),
            self._create_event(2.0, 'inv2', 'e2'),
            self._create_event(3.0, 'inv3', 'e3'),
            self._create_event(4.0, 'inv4', 'e4'),
            self._create_compacted_event(
                1.0, 1.0, 'Summary 1', appended_ts=5.0
            ),
            self._create_event(6.0, 'inv6', 'e6'),
            self._create_event(7.0, 'inv7', 'e7'),
            self._create_compacted_event(
                1.0, 3.0, 'Summary 1-3', appended_ts=8.0
            ),
            self._create_event(9.0, 'inv9', 'e9', prompt_token_count=100),
        ],
    )

    mock_compacted_event = self._create_compacted_event(1.0, 4.0, 'Summary 1-4')
    self.mock_compactor.maybe_summarize_events.return_value = (
        mock_compacted_event
    )

    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )

    compacted_events_arg = self.mock_compactor.maybe_summarize_events.call_args[
        1
    ]['events']
    self.assertEqual(
        [e.content.parts[0].text for e in compacted_events_arg],
        ['Summary 1-3', 'e4'],
    )
    self.assertEqual(compacted_events_arg[0].timestamp, 1.0)
    self.assertEqual(compacted_events_arg[1].invocation_id, 'inv4')
    self.mock_session_service.append_event.assert_called_once_with(
        session=session, event=mock_compacted_event
    )

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

  def test_get_contents_subsumed_compaction_is_hidden(self):
    events = [
        self._create_event(1.0, 'inv1', 'Event 1'),
        self._create_event(2.0, 'inv2', 'Event 2'),
        self._create_event(3.0, 'inv3', 'Event 3'),
        self._create_event(4.0, 'inv4', 'Event 4'),
        self._create_compacted_event(1.0, 1.0, 'Summary 1'),
        self._create_event(6.0, 'inv6', 'Event 6'),
        self._create_event(7.0, 'inv7', 'Event 7'),
        self._create_compacted_event(1.0, 3.0, 'Summary 1-3'),
        self._create_event(9.0, 'inv9', 'Event 9'),
    ]

    result_contents = contents._get_contents(None, events)
    expected_texts = [
        'Summary 1-3',
        'Event 4',
        'Event 6',
        'Event 7',
        'Event 9',
    ]
    actual_texts = [c.parts[0].text for c in result_contents]
    self.assertEqual(actual_texts, expected_texts)

  def test_get_contents_compaction_appended_late_keeps_newer_events(self):
    events = [
        self._create_event(1.0, 'inv1', 'Event 1'),
        self._create_event(2.0, 'inv2', 'Event 2'),
        self._create_event(3.0, 'inv3', 'Event 3'),
        self._create_event(4.0, 'inv4', 'Event 4'),
        self._create_event(5.0, 'inv5', 'Event 5'),
        self._create_compacted_event(1.0, 3.0, 'Summary 1-3', appended_ts=6.0),
    ]

    result_contents = contents._get_contents(None, events)
    expected_texts = ['Summary 1-3', 'Event 4', 'Event 5']
    actual_texts = [c.parts[0].text for c in result_contents]
    self.assertEqual(actual_texts, expected_texts)

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
