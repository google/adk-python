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

"""AgentCore Memory-backed session service implementation for ADK."""

from __future__ import annotations

import asyncio
from datetime import datetime
from datetime import timezone
import json
import logging
from typing import Any
from typing import Optional

from ...errors.already_exists_error import AlreadyExistsError
from ...events.event import Event
from ...platform import uuid as platform_uuid
from ...sessions.base_session_service import BaseSessionService
from ...sessions.base_session_service import GetSessionConfig
from ...sessions.base_session_service import ListSessionsResponse
from ...sessions.session import Session
from ...sessions.state import State
from ...utils._dependency import missing_extra
from ._config import AgentCoreSessionServiceConfig

logger = logging.getLogger('google_adk.' + __name__)

# Stored as a blob so get_session can tell "this session exists" from
# "this session was never created". AgentCore has no CreateSession API.
_BOOTSTRAP_KIND = 'adk.session.bootstrap'
_ACTOR_SEPARATOR = ':'
_PAGE_SIZE = 100


def _unix_timestamp(value: object) -> float:
  """Converts an AgentCore eventTimestamp to a unix timestamp."""
  if isinstance(value, datetime):
    if value.tzinfo is None:
      value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()
  if isinstance(value, (int, float)):
    return float(value)
  return 0.0


class AgentCoreSessionService(BaseSessionService):
  """Session service backed by AWS Bedrock AgentCore Memory short-term memory.

  AgentCore has no session resource. ADK sessions are mapped as:

  - ``actorId``: ``{app_name}:{user_id}``
  - ``sessionId``: the ADK session id
  - each ADK ``Event``: one AgentCore event whose payload is the conversational
    text (when present) plus a blob of the full ADK event JSON, so history can
    be reconstructed without losing tool calls, state deltas, or metadata.

  Partial (streaming) events are not written. ``delete_session`` deletes every
  AgentCore event for that session.
  """

  def __init__(
      self,
      *,
      memory_id: Optional[str] = None,
      config: Optional[AgentCoreSessionServiceConfig] = None,
      client: Optional[Any] = None,
      region_name: Optional[str] = None,
  ):
    """Initializes the AgentCoreSessionService.

    Args:
      memory_id: AgentCore Memory resource id. Ignored when ``config`` is set.
      config: Optional full config. If omitted, ``memory_id`` is required.
      client: Optional pre-configured ``bedrock-agentcore`` boto3 client (or a
        test double). When omitted, a client is created lazily.
      region_name: AWS region. Ignored when ``config`` is set.
    """
    if config is None:
      if not memory_id:
        raise ValueError('memory_id is required when config is not provided.')
      config = AgentCoreSessionServiceConfig(
          memory_id=memory_id, region_name=region_name
      )
    self.config = config
    self._client = client

  def _get_client(self) -> Any:
    """Lazily initializes and returns the bedrock-agentcore client."""
    if self._client is not None:
      return self._client
    try:
      import boto3  # type: ignore
    except ImportError as e:
      raise missing_extra('boto3', 'agentcore') from e
    kwargs: dict[str, Any] = {}
    if self.config.region_name:
      kwargs['region_name'] = self.config.region_name
    self._client = boto3.client('bedrock-agentcore', **kwargs)
    return self._client

  async def _invoke(self, method_name: str, **kwargs: Any) -> Any:
    client = self._get_client()
    method = getattr(client, method_name)
    return await asyncio.to_thread(method, **kwargs)

  def _actor_id(self, app_name: str, user_id: str) -> str:
    return f'{app_name}{_ACTOR_SEPARATOR}{user_id}'

  def _user_id_from_actor(self, actor_id: str, app_name: str) -> Optional[str]:
    prefix = f'{app_name}{_ACTOR_SEPARATOR}'
    if not actor_id.startswith(prefix):
      return None
    return actor_id[len(prefix) :]

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    """Creates a new session.

    AgentCore has no create-session API, so this writes a bootstrap event
    (extraction skipped) so later ``get_session`` / ``list_sessions`` see it.
    """
    sid = session_id or platform_uuid.new_uuid()
    existing = await self._list_all_events(
        app_name=app_name,
        user_id=user_id,
        session_id=sid,
        include_payloads=False,
    )
    if existing:
      raise AlreadyExistsError(
          f'Session {sid} already exists for user {user_id} in app {app_name}.'
      )

    initial_state = dict(state or {})
    persisted_state = {
        k: v
        for k, v in initial_state.items()
        if not k.startswith(State.TEMP_PREFIX)
    }
    now = datetime.now(timezone.utc)
    await self._invoke(
        'create_event',
        memoryId=self.config.memory_id,
        actorId=self._actor_id(app_name, user_id),
        sessionId=sid,
        eventTimestamp=now,
        extractionMode='SKIP',
        payload=[{
            'blob': json.dumps({
                'adk_kind': _BOOTSTRAP_KIND,
                'state': persisted_state,
            })
        }],
    )
    return Session(
        id=sid,
        app_name=app_name,
        user_id=user_id,
        state=initial_state,
        events=[],
        last_update_time=now.timestamp(),
    )

  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    """Gets a session and reconstructs ADK events from AgentCore payloads."""
    raw_events = await self._list_all_events(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        include_payloads=True,
    )
    if not raw_events:
      return None

    raw_events.sort(key=lambda e: _unix_timestamp(e.get('eventTimestamp')))

    bootstrap_state: dict[str, Any] = {}
    events: list[Event] = []
    last_update = _unix_timestamp(raw_events[-1].get('eventTimestamp'))
    for raw in raw_events:
      kind, payload_state, adk_event = self._parse_payload(raw)
      if kind == _BOOTSTRAP_KIND:
        bootstrap_state = dict(payload_state or {})
        continue
      if adk_event is not None:
        events.append(adk_event)

    session_state = dict(bootstrap_state)
    for event in events:
      if event.actions and event.actions.state_delta:
        for key, value in event.actions.state_delta.items():
          if not key.startswith(State.TEMP_PREFIX):
            session_state[key] = value

    if config is not None:
      if config.after_timestamp is not None:
        events = [e for e in events if e.timestamp >= config.after_timestamp]
      if config.num_recent_events is not None:
        events = (
            events[-config.num_recent_events :]
            if config.num_recent_events
            else []
        )

    return Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=session_state,
        events=events,
        last_update_time=last_update,
    )

  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    """Lists sessions for a user, or all users of the app when user_id is None."""
    if user_id is not None:
      actor_ids = [self._actor_id(app_name, user_id)]
    else:
      actor_ids = await self._list_actor_ids(app_name)

    sessions: list[Session] = []
    for actor_id in actor_ids:
      listed_user = (
          user_id
          if user_id is not None
          else self._user_id_from_actor(actor_id, app_name)
      )
      if listed_user is None:
        continue
      summaries = await self._list_session_summaries(actor_id)
      for item in summaries:
        created = item.get('createdAt')
        sessions.append(
            Session(
                id=item['sessionId'],
                app_name=app_name,
                user_id=listed_user,
                events=[],
                last_update_time=_unix_timestamp(created),
            )
        )

    sessions.sort(key=lambda s: s.last_update_time)
    return ListSessionsResponse(sessions=sessions)

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Deletes a session by deleting every AgentCore event in it."""
    raw_events = await self._list_all_events(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        include_payloads=False,
    )
    actor_id = self._actor_id(app_name, user_id)
    for raw in raw_events:
      event_id = raw.get('eventId')
      if not event_id:
        continue
      await self._invoke(
          'delete_event',
          memoryId=self.config.memory_id,
          actorId=actor_id,
          sessionId=session_id,
          eventId=event_id,
      )

  async def append_event(self, session: Session, event: Event) -> Event:
    """Appends an event locally, then flushes it to AgentCore STM."""
    if event.partial:
      return event
    event = await super().append_event(session, event)
    payload: list[dict[str, Any]] = [{'blob': event.model_dump_json()}]
    text = self._extract_text(event)
    if text:
      payload.insert(
          0,
          {
              'conversational': {
                  'role': self._conversational_role(event),
                  'content': {'text': text},
              }
          },
      )
    timestamp = datetime.fromtimestamp(event.timestamp, tz=timezone.utc)
    await self._invoke(
        'create_event',
        memoryId=self.config.memory_id,
        actorId=self._actor_id(session.app_name, session.user_id),
        sessionId=session.id,
        eventTimestamp=timestamp,
        payload=payload,
    )
    session.last_update_time = event.timestamp
    return event

  def _extract_text(self, event: Event) -> Optional[str]:
    if not event.content or not event.content.parts:
      return None
    texts = [p.text for p in event.content.parts if p.text]
    return '\n'.join(texts) if texts else None

  def _conversational_role(self, event: Event) -> str:
    # AgentCore only accepts ASSISTANT | USER | TOOL | OTHER (uppercase).
    parts = event.content.parts if event.content else None
    if parts and any(getattr(p, 'function_response', None) for p in parts):
      return 'TOOL'
    if event.author == 'user':
      return 'USER'
    return 'ASSISTANT'

  def _parse_payload(
      self, raw: dict[str, Any]
  ) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[Event]]:
    """Returns (bootstrap kind, bootstrap state, ADK event)."""
    for item in raw.get('payload') or []:
      blob = item.get('blob')
      if not isinstance(blob, str):
        continue
      try:
        data = json.loads(blob)
      except json.JSONDecodeError:
        continue
      if isinstance(data, dict) and data.get('adk_kind') == _BOOTSTRAP_KIND:
        state = data.get('state')
        return (
            _BOOTSTRAP_KIND,
            state if isinstance(state, dict) else {},
            None,
        )
      try:
        return None, None, Event.model_validate_json(blob)
      except Exception:  # pylint: disable=broad-except
        logger.debug('Skipping unreadable AgentCore event blob.', exc_info=True)
        continue
    return None, None, None

  async def _list_all_events(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      include_payloads: bool = True,
  ) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    next_token: Optional[str] = None
    actor_id = self._actor_id(app_name, user_id)
    while True:
      params: dict[str, Any] = {
          'memoryId': self.config.memory_id,
          'actorId': actor_id,
          'sessionId': session_id,
          'includePayloads': include_payloads,
          'maxResults': _PAGE_SIZE,
      }
      if next_token is not None:
        params['nextToken'] = next_token
      response = await self._invoke('list_events', **params)
      events.extend(response.get('events') or [])
      next_token = response.get('nextToken')
      if not next_token:
        break
    return events

  async def _list_session_summaries(
      self, actor_id: str
  ) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    next_token: Optional[str] = None
    while True:
      params: dict[str, Any] = {
          'memoryId': self.config.memory_id,
          'actorId': actor_id,
          'maxResults': _PAGE_SIZE,
      }
      if next_token is not None:
        params['nextToken'] = next_token
      response = await self._invoke('list_sessions', **params)
      summaries.extend(response.get('sessionSummaries') or [])
      next_token = response.get('nextToken')
      if not next_token:
        break
    return summaries

  async def _list_actor_ids(self, app_name: str) -> list[str]:
    actor_ids: list[str] = []
    next_token: Optional[str] = None
    prefix = f'{app_name}{_ACTOR_SEPARATOR}'
    while True:
      params: dict[str, Any] = {
          'memoryId': self.config.memory_id,
          'maxResults': _PAGE_SIZE,
      }
      if next_token is not None:
        params['nextToken'] = next_token
      response = await self._invoke('list_actors', **params)
      for item in response.get('actorSummaries') or []:
        actor_id = item.get('actorId')
        if isinstance(actor_id, str) and actor_id.startswith(prefix):
          actor_ids.append(actor_id)
      next_token = response.get('nextToken')
      if not next_token:
        break
    return actor_ids
