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

from collections.abc import Mapping
from collections.abc import Sequence
import hashlib
import json
import re
from typing import Any
from typing import TYPE_CHECKING
from urllib.parse import quote

from google.genai import types
from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session

_UNKNOWN_SESSION_ID = '__unknown_session_id__'


def _key_part(value: str) -> str:
  return quote(value, safe='')


def _decode(value: Any) -> str:
  if isinstance(value, bytes):
    return value.decode('utf-8')
  return str(value)


def _extract_words_lower(text: str) -> set[str]:
  """Extracts words from a string and converts them to lowercase."""
  return set([word.lower() for word in re.findall(r'[A-Za-z]+', text)])


def _content_to_text(content: types.Content) -> str:
  return ' '.join([part.text for part in content.parts or [] if part.text])


def _content_from_payload(payload: dict[str, Any]) -> types.Content:
  return types.Content.model_validate(payload)


def _event_id(event: Event) -> str:
  if event.id:
    return event.id
  content_text = _content_to_text(event.content) if event.content else ''
  digest = hashlib.sha256(
      f'{event.author}:{event.timestamp}:{content_text}'.encode('utf-8')
  ).hexdigest()
  return f'generated-{digest}'


def _event_to_payload(
    event: Event,
    *,
    session_id: str,
    custom_metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
  metadata = dict(custom_metadata or {})
  metadata.setdefault('session_id', session_id)
  return {
      'id': _event_id(event),
      'author': event.author,
      'timestamp': _utils.format_timestamp(event.timestamp),
      'content': event.content.model_dump(
          mode='json', by_alias=True, exclude_none=True
      ),
      'custom_metadata': metadata,
  }


class RedisMemoryService(BaseMemoryService):
  """A Redis-backed memory service.

  This service mirrors InMemoryMemoryService's keyword search behavior while
  keeping memory entries in Redis so they survive process restarts.
  """

  def __init__(
      self,
      redis_url: str | None = None,
      *,
      key_prefix: str = 'adk:memory:',
      client: Any | None = None,
      **redis_kwargs: Any,
  ):
    """Initializes the Redis memory service.

    Args:
      redis_url: URL passed to redis.asyncio.from_url. Required when client is
        not supplied.
      key_prefix: Prefix for all Redis keys written by this service.
      client: Optional async Redis-compatible client, mainly for tests.
      **redis_kwargs: Extra keyword arguments forwarded to from_url.
    """
    if client is None:
      if redis_url is None:
        raise ValueError('redis_url is required when client is not supplied.')
      try:
        from redis import asyncio as redis_asyncio
      except ImportError as e:
        from ..utils._dependency import missing_extra

        raise missing_extra('redis', 'redis') from e
      client = redis_asyncio.from_url(redis_url, **redis_kwargs)

    self._client = client
    self._key_prefix = key_prefix

  def _scope_prefix(self, app_name: str, user_id: str) -> str:
    return (
        f'{self._key_prefix}{_key_part(app_name)}:{_key_part(user_id)}'
    )

  def _sessions_key(self, app_name: str, user_id: str) -> str:
    return f'{self._scope_prefix(app_name, user_id)}:sessions'

  def _session_keys(
      self, app_name: str, user_id: str, session_id: str
  ) -> tuple[str, str]:
    session_prefix = (
        f'{self._scope_prefix(app_name, user_id)}:{_key_part(session_id)}'
    )
    return f'{session_prefix}:order', f'{session_prefix}:entries'

  async def _append_events(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      events: Sequence[Event],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    events_to_add = [
        event for event in events if event.content and event.content.parts
    ]
    await self._client.sadd(self._sessions_key(app_name, user_id), session_id)
    order_key, entries_key = self._session_keys(app_name, user_id, session_id)

    for event in events_to_add:
      event_id = _event_id(event)
      payload = _event_to_payload(
          event, session_id=session_id, custom_metadata=custom_metadata
      )
      was_added = await self._client.hsetnx(
          entries_key, event_id, json.dumps(payload)
      )
      if was_added:
        await self._client.rpush(order_key, event_id)

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    session_id = session.id or _UNKNOWN_SESSION_ID
    order_key, entries_key = self._session_keys(
        session.app_name, session.user_id, session_id
    )
    await self._client.delete(order_key, entries_key)
    await self._append_events(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session_id,
        events=session.events,
    )

  @override
  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: str | None = None,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    await self._append_events(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id or _UNKNOWN_SESSION_ID,
        events=events,
        custom_metadata=custom_metadata,
    )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    sessions_key = self._sessions_key(app_name, user_id)
    session_ids = sorted([
        _decode(value) for value in await self._client.smembers(sessions_key)
    ])
    words_in_query = _extract_words_lower(query)
    response = SearchMemoryResponse()

    for session_id in session_ids:
      order_key, entries_key = self._session_keys(app_name, user_id, session_id)
      event_ids = [
          _decode(value)
          for value in await self._client.lrange(order_key, 0, -1)
      ]
      for event_id in event_ids:
        raw_payload = await self._client.hget(entries_key, event_id)
        if raw_payload is None:
          continue
        payload = json.loads(_decode(raw_payload))
        content = _content_from_payload(payload['content'])
        words_in_memory = _extract_words_lower(_content_to_text(content))
        if not words_in_memory:
          continue
        if any(query_word in words_in_memory for query_word in words_in_query):
          response.memories.append(
              MemoryEntry(
                  id=payload['id'],
                  content=content,
                  author=payload.get('author'),
                  timestamp=payload.get('timestamp'),
                  custom_metadata=payload.get('custom_metadata') or {},
              )
          )

    return response

  async def close(self) -> None:
    """Closes the Redis client if it exposes a close method."""
    close = getattr(self._client, 'aclose', None)
    if close is None:
      close = getattr(self._client, 'close', None)
    if close is not None:
      result = close()
      if hasattr(result, '__await__'):
        await result

  async def __aenter__(self) -> RedisMemoryService:
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self.close()
