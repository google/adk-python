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

import unittest
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.compaction import _run_compaction_for_sliding_window
from google.adk.apps.sliding_window_compactor import SlidingWindowCompactor
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
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
    self.mock_compactor = AsyncMock(spec=SlidingWindowCompactor)

  def _create_event(
      self, timestamp: float, invocation_id: str, text: str
  ) -> Event:
    return Event(
        timestamp=timestamp,
        invocation_id=invocation_id,
        author='user',
        content=Content(parts=[Part(text=text)]),
    )

  def _create_compacted_event(
      self, start_ts: float, end_ts: float, summary_text: str
  ) -> Event:
    compaction = EventCompaction(
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        compacted_content=Content(parts=[Part(text=summary_text)]),
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
    self.mock_compactor.maybe_compact_events.assert_not_called()
    self.mock_session_service.append_event.assert_not_called()

  async def test_run_compaction_for_sliding_window_not_enough_new_invocations(
      self,
  ):
    app = App(
        name='test',
        root_agent=Mock(spec=BaseAgent),
        events_compaction_config=EventsCompactionConfig(
            compactor=self.mock_compactor,
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
    self.mock_compactor.maybe_compact_events.assert_not_called()
    self.mock_session_service.append_event.assert_not_called()

  async def test_run_compaction_for_sliding_window_first_compaction(self):
    app = App(
        name='test',
        root_agent=Mock(spec=BaseAgent),
        events_compaction_config=EventsCompactionConfig(
            compactor=self.mock_compactor,
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
    self.mock_compactor.maybe_compact_events.return_value = mock_compacted_event

    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )

    # Expected events to compact: inv1, inv2, inv3, inv4
    compacted_events_arg = self.mock_compactor.maybe_compact_events.call_args[
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
            compactor=self.mock_compactor,
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
    self.mock_compactor.maybe_compact_events.return_value = mock_compacted_event

    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )

    # New invocations are inv3, inv4, inv5 (3 new) > threshold (2).
    # Overlap size is 1, so start from 1 inv before inv3, which is inv2.
    # Compact range: inv2 to inv5.
    compacted_events_arg = self.mock_compactor.maybe_compact_events.call_args[
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
            compactor=self.mock_compactor,
            compaction_interval=1,
            overlap_size=0,
        ),
    )
    events = [self._create_event(1.0, 'inv1', 'e1')]
    session = Session(app_name='test', user_id='u1', id='s1', events=events)

    self.mock_compactor.maybe_compact_events.return_value = None

    await _run_compaction_for_sliding_window(
        app, session, self.mock_session_service
    )

    self.mock_compactor.maybe_compact_events.assert_called_once()
    self.mock_session_service.append_event.assert_not_called()
