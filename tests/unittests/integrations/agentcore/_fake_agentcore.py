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

"""In-memory stand-in for the bedrock-agentcore client methods ADK calls."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import Any


class FakeAgentCoreClient:
  """Stores Actor/session/event records the way AgentCore Memory does."""

  def __init__(self) -> None:
    # (actor_id, session_id) -> list of event dicts
    self._events: dict[tuple[str, str], list[dict[str, Any]]] = {}
    self._counter = 0

  def create_event(self, **kwargs: Any) -> dict[str, Any]:
    actor_id = kwargs['actorId']
    session_id = kwargs['sessionId']
    self._counter += 1
    timestamp = kwargs.get('eventTimestamp') or datetime.now(timezone.utc)
    event = {
        'memoryId': kwargs['memoryId'],
        'actorId': actor_id,
        'sessionId': session_id,
        'eventId': f'evt-{self._counter}',
        'eventTimestamp': timestamp,
        'payload': kwargs.get('payload') or [],
    }
    if 'extractionMode' in kwargs:
      event['extractionMode'] = kwargs['extractionMode']
    key = (actor_id, session_id)
    self._events.setdefault(key, []).append(event)
    return {'event': dict(event)}

  def list_events(self, **kwargs: Any) -> dict[str, Any]:
    key = (kwargs['actorId'], kwargs['sessionId'])
    events = list(self._events.get(key, []))
    include_payloads = kwargs.get('includePayloads', True)
    if not include_payloads:
      events = [
          {k: v for k, v in event.items() if k != 'payload'} for event in events
      ]
    return {'events': events}

  def list_sessions(self, **kwargs: Any) -> dict[str, Any]:
    actor_id = kwargs['actorId']
    summaries: list[dict[str, Any]] = []
    for (stored_actor, session_id), events in self._events.items():
      if stored_actor != actor_id or not events:
        continue
      summaries.append({
          'sessionId': session_id,
          'actorId': stored_actor,
          'createdAt': events[0]['eventTimestamp'],
      })
    return {'sessionSummaries': summaries}

  def list_actors(self, **kwargs: Any) -> dict[str, Any]:
    actor_ids = sorted({actor_id for actor_id, _ in self._events})
    return {'actorSummaries': [{'actorId': a} for a in actor_ids]}

  def delete_event(self, **kwargs: Any) -> dict[str, Any]:
    key = (kwargs['actorId'], kwargs['sessionId'])
    event_id = kwargs['eventId']
    stored = self._events.get(key, [])
    self._events[key] = [e for e in stored if e.get('eventId') != event_id]
    if not self._events[key]:
      self._events.pop(key, None)
    return {'eventId': event_id}
