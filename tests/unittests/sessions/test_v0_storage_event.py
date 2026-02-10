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

from datetime import datetime
from datetime import timezone

from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.sessions.schemas.v0 import StorageEvent
from google.genai import types


def test_storage_event_v0_to_event_rehydrates_compaction_model():
  compaction = EventCompaction(
      start_timestamp=1.0,
      end_timestamp=2.0,
      compacted_content=types.Content(
          role="user",
          parts=[types.Part(text="compacted")],
      ),
  )
  actions = EventActions(compaction=compaction)
  storage_event = StorageEvent(
      id="event_id",
      invocation_id="invocation_id",
      author="author",
      actions=actions,
      session_id="session_id",
      app_name="app_name",
      user_id="user_id",
      timestamp=datetime.fromtimestamp(3.0, tz=timezone.utc),
  )

  event = storage_event.to_event()

  assert event.actions is not None
  assert isinstance(event.actions.compaction, EventCompaction)
  assert event.actions.compaction.start_timestamp == 1.0
  assert event.actions.compaction.end_timestamp == 2.0
