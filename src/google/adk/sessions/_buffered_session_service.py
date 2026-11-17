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

"""A session service decorator that persists writes in the background."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
import copy
import datetime
import math
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from typing_extensions import override

from ..errors._stale_session_error import StaleSessionError
from ..errors.session_not_found_error import SessionNotFoundError
from ..events.event import Event
from ..features import experimental
from ..features import FeatureName
from ..platform import _random as platform_random
from ..workflow._retry_config import RetryConfig
from ._session_write_buffer import SessionKey
from ._session_write_buffer import SessionWriteBuffer
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State

# Another attempt with the same store session gets the same error again.
_NOT_RETRIED = (StaleSessionError, SessionNotFoundError)

_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_INITIAL_DELAY = 1.0
_DEFAULT_MAX_DELAY = 60.0
_DEFAULT_BACKOFF_FACTOR = 2.0
_DEFAULT_JITTER = 1.0


def _stores_batches(service: BaseSessionService) -> bool:
  """Returns whether the service's class overrides `append_events`."""
  # Checked on the class, since a mock answers every attribute.
  append_events = getattr(service.__class__, 'append_events', None)
  return (
      callable(append_events)
      and append_events is not BaseSessionService.append_events
  )


class BufferedSessionServiceConfig(BaseModel):
  """Configuration for `BufferedSessionService`."""

  model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

  retry_config: RetryConfig = Field(
      default_factory=lambda: RetryConfig(max_attempts=1)
  )
  """How a failed write is retried. Off by default, since a retry can store an
  event twice if its first attempt committed. Stale and not-found errors are
  never retried."""

  max_write_delay: datetime.timedelta = datetime.timedelta(0)
  """How long a session's first queued write waits for more events to join it.

  Applies when the wrapped service overrides `append_events`.
  """

  max_batch_events: int = Field(default=50, ge=1)
  """The most events sent in one `append_events` call."""

  @field_validator('max_write_delay')
  @classmethod
  def _validate_max_write_delay(
      cls, value: datetime.timedelta
  ) -> datetime.timedelta:
    if value < datetime.timedelta(0):
      raise ValueError('max_write_delay must not be negative.')
    return value


@experimental(FeatureName.BUFFERED_SESSION_SERVICE)
class BufferedSessionService(BaseSessionService):
  """A session service that writes events to another one in the background.

  `append_event` updates the session and returns at once. A writer per session
  sends the events to the wrapped service in order, and reads wait for the
  writes they would see. When a write fails, the session's next
  `append_event`, `flush_session` or `delete_session` raises
  `SessionPersistenceError`.

  Writers run on the event loop of the first append; use one instance per
  loop. The service is not thread-safe.
  """

  def __init__(
      self,
      delegate: BaseSessionService,
      *,
      config: BufferedSessionServiceConfig | None = None,
  ) -> None:
    """Initializes the service.

    Args:
      delegate: The service that stores the events. Closing this service
        leaves it open.
      config: How failed writes are retried, and how writes are batched.
    """
    self._delegate = delegate
    self._config = config or BufferedSessionServiceConfig()
    self._batch_writes = _stores_batches(delegate)
    # A delay only helps writes that share a call.
    delay = (
        self._config.max_write_delay.total_seconds()
        if self._batch_writes
        else 0.0
    )
    self._buffer = SessionWriteBuffer(
        self._persist,
        max_write_delay=delay,
        max_batch_events=(
            self._config.max_batch_events if self._batch_writes else 1
        ),
    )

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: dict[str, Any] | None = None,
      session_id: str | None = None,
      **kwargs: Any,
  ) -> Session:
    """Waits for buffered `app:` and `user:` changes, then creates it."""
    self._buffer.check_loop()
    await self._buffer.await_writes_seen_by(
        SessionKey(app_name, user_id, session_id or '')
    )
    return await self._delegate.create_session(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
        **kwargs,
    )

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: GetSessionConfig | None = None,
  ) -> Session | None:
    """Waits for the buffered writes the session would see, then reads it."""
    self._buffer.check_loop()
    await self._buffer.await_writes_seen_by(
        SessionKey(app_name, user_id, session_id)
    )
    return await self._delegate.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        config=config,
    )

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: str | None = None
  ) -> ListSessionsResponse:
    """Waits for the buffered writes the listing would see, then lists."""
    self._buffer.check_loop()
    await self._buffer.await_writes_listed_for(app_name, user_id)
    return await self._delegate.list_sessions(
        app_name=app_name, user_id=user_id
    )

  @override
  async def get_user_state(
      self, *, app_name: str, user_id: str
  ) -> dict[str, Any]:
    """Waits for buffered `user:` changes, then reads the user's state."""
    self._buffer.check_loop()
    await self._buffer.await_user_state_writes(app_name, user_id)
    return await self._delegate.get_user_state(
        app_name=app_name, user_id=user_id
    )

  @override
  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Waits for the session's buffered writes, then deletes it.

    Raises:
      SessionPersistenceError: A write of the session failed. The session is
        not deleted.
    """
    self._buffer.check_loop()
    await self._buffer.flush(
        SessionKey(app_name, user_id, session_id), _do_nothing
    )
    await self._delegate.delete_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    """Applies the event to the session and queues its write.

    Raises:
      SessionPersistenceError: An earlier write of the session failed, or
        `session` holds an event whose write failed.
    """
    if event.partial:
      return event
    if not session.id:
      raise ValueError('BufferedSessionService requires a session id.')
    # Copied first, so an event that cannot be copied changes nothing.
    snapshot = _snapshot(event)
    return self._buffer.append(
        session, snapshot, lambda: self._apply_to_live_session(session, event)
    )

  def _apply_to_live_session(self, session: Session, event: Event) -> Event:
    """Applies the event to the session as a store's `append_event` does."""
    self._apply_temp_state(session, event)
    event = self._trim_temp_delta_state(event)
    appended = self._commit_event_to_session(session, event)
    session.last_update_time = appended.timestamp
    return appended

  @override
  async def flush_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Waits until one session's buffered writes are stored.

    Raises:
      SessionPersistenceError: A write of the session failed.
    """
    self._buffer.check_loop()
    if not session_id.strip():
      raise ValueError(
          'BufferedSessionService.flush_session requires a session id.'
      )
    await self._buffer.flush(
        SessionKey(app_name, user_id, session_id),
        lambda: self._delegate.flush_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        ),
    )

  @override
  async def flush(self) -> None:
    """Waits until every session's buffered writes are stored.

    Raises:
      SessionPersistenceError: A write failed. Reports each failure once.
    """
    self._buffer.check_loop()
    await self._buffer.flush_all(self._delegate.flush)

  async def aclose(self) -> None:
    """Flushes every session and stops the writers; leaves the store open."""
    self._buffer.check_loop()
    await self._buffer.aclose(self._delegate.flush)

  async def close(self) -> None:
    """Same as `aclose`."""
    await self.aclose()

  async def __aenter__(self) -> BufferedSessionService:
    return self

  async def __aexit__(self, *exc_info: object) -> None:
    await self.aclose()

  async def _persist(
      self, store_session: Session, events: Sequence[Event]
  ) -> BaseException | None:
    """Writes `events` with retries and returns the final error, if any."""

    async def attempt() -> None:
      # A failed attempt may have left events on the store session.
      store_session.events.clear()
      if self._batch_writes:
        await self._delegate.append_events(store_session, list(events))
      else:
        await self._delegate.append_event(store_session, events[0])

    return await self._with_retries(attempt)

  async def _with_retries(
      self, call: Callable[[], Awaitable[object]]
  ) -> BaseException | None:
    """Runs `call` until it succeeds or retries run out; returns the error."""
    retry = self._config.retry_config
    failed_attempts = 0
    while True:
      try:
        await call()
        return None
      except asyncio.CancelledError as e:
        task = asyncio.current_task()
        if task is None or task.cancelling():
          raise
        # Raised by the store, not a cancellation of this task.
        failure: BaseException = e
      except _NOT_RETRIED as e:
        return e
      except Exception as e:  # pylint: disable=broad-exception-caught
        failure = e
      failed_attempts += 1
      if not _should_retry(retry, failure, failed_attempts):
        return failure
      await asyncio.sleep(_retry_delay(retry, failed_attempts))


def _snapshot(event: Event) -> Event:
  """Returns a deep copy of the event without `temp:` state."""
  delta = event.actions.state_delta if event.actions else None
  if delta and any(k.startswith(State.TEMP_PREFIX) for k in delta):
    actions = event.actions.model_copy(
        update={
            'state_delta': {
                k: v
                for k, v in delta.items()
                if not k.startswith(State.TEMP_PREFIX)
            }
        }
    )
    event = event.model_copy(update={'actions': actions})
  return copy.deepcopy(event)


def _should_retry(
    retry: RetryConfig, failure: BaseException, failed_attempts: int
) -> bool:
  max_attempts = (
      retry.max_attempts
      if retry.max_attempts is not None
      else _DEFAULT_MAX_ATTEMPTS
  )
  if failed_attempts >= max_attempts:
    return False
  if retry.exceptions is None:
    return True
  # A configured exception also covers its subclasses, as in workflow retries.
  names = {cls.__name__ for cls in type(failure).__mro__ if cls is not object}
  return not names.isdisjoint(retry.exceptions)


def _retry_delay(retry: RetryConfig, failed_attempts: int) -> float:
  """Returns the seconds to wait after `failed_attempts` failed attempts."""
  initial_delay = _or_default(retry.initial_delay, _DEFAULT_INITIAL_DELAY)
  max_delay = _or_default(retry.max_delay, _DEFAULT_MAX_DELAY)
  backoff_factor = _or_default(retry.backoff_factor, _DEFAULT_BACKOFF_FACTOR)
  jitter = _or_default(retry.jitter, _DEFAULT_JITTER)
  if initial_delay <= 0:
    return 0.0
  try:
    delay = initial_delay * backoff_factor ** (failed_attempts - 1)
  except OverflowError:
    delay = math.inf
  if jitter > 0:
    delay = min(delay, max_delay / (1.0 + jitter))
    delay += platform_random.get_random().uniform(
        -jitter * delay, jitter * delay
    )
  return min(max(0.0, delay), max_delay)


def _or_default(value: float | None, default: float) -> float:
  return default if value is None else value


async def _do_nothing() -> None:
  pass
