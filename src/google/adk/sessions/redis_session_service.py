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

from __future__ import annotations

import json
import logging
from typing import Any, Optional
import uuid
from datetime import datetime

import redis.asyncio as redis
from pydantic import ValidationError
from typing_extensions import override

from ..errors.already_exists_error import AlreadyExistsError
from ..errors.session_service_error import SessionServiceError
from ..events.event import Event
from . import _session_util
from .base_session_service import BaseSessionService, GetSessionConfig, ListSessionsResponse
from .session import Session
from .state import State

logger = logging.getLogger('google_adk.' + __name__)


class RedisSessionService(BaseSessionService):
    """A Redis-backed implementation of the session service."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        redis_client: Optional[redis.Redis] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """Initialize the Redis session service.

        Args:
            redis_url: Redis connection URL
            redis_client: Optional existing Redis client
            ttl_seconds: Optional TTL for session keys
        """
        self.redis = redis_client or redis.from_url(redis_url)
        self.ttl_seconds = ttl_seconds

    @override
    async def create_session(
        self,
        *, 
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """Create a new session in Redis."""
        session_id = session_id or str(uuid.uuid4())
        session_key = f"session:{app_name}:{user_id}:{session_id}"

        # Check if session already exists
        if await self.redis.exists(session_key):
            raise AlreadyExistsError(f"Session {session_id} already exists")

        # Extract state deltas
        state = state or {}
        deltas = _session_util.extract_state_delta(state)

        # Create session data
        now = datetime.utcnow().timestamp()
        session_data = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "last_update_time": now,
        }

        # Store session metadata and state
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                # Store session metadata
                await pipe.hset(session_key, mapping=session_data)

                # Store session-specific state
                if deltas["session"]:
                    await pipe.hset(f"{session_key}:state", mapping=deltas["session"])

                # Store app-level state
                if deltas["app"]:
                    app_state_key = f"app:{app_name}:state"
                    await pipe.hset(app_state_key, mapping=deltas["app"])

                # Store user-level state
                if deltas["user"]:
                    user_state_key = f"user:{user_id}:state"
                    await pipe.hset(user_state_key, mapping=deltas["user"])

                # Set TTL if specified
                if self.ttl_seconds:
                    await pipe.expire(session_key, self.ttl_seconds)
                    await pipe.expire(f"{session_key}:state", self.ttl_seconds)
                    await pipe.expire(f"{session_key}:events", self.ttl_seconds)

                await pipe.execute()
            except redis.RedisError as e:
                logger.error(f"Redis error creating session: {e}")
                raise SessionServiceError(f"Failed to create session: {e}") from e

        # Create and return session object
        session = Session(
            app_name=app_name,
            user_id=user_id,
            id=session_id,
            state=state,
            last_update_time=now,
        )
        return self._merge_state(app_name, user_id, session)

    @override
    async def get_session(
        self,
        *, 
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """Retrieve a session from Redis with optional event filtering."""
        session_key = f"session:{app_name}:{user_id}:{session_id}"

        # Get session metadata
        session_data = await self.redis.hgetall(session_key)
        if not session_data:
            return None

        # Get session state
        session_state_key = f"{session_key}:state"
        session_state = {k.decode(): v.decode() for k, v in await self.redis.hgetall(session_state_key).items()}

        # Get app state
        app_state_key = f"app:{app_name}:state"
        app_state_raw = await self.redis.hgetall(app_state_key)
        app_state = {f"{State.APP_PREFIX}{k.decode()}": v.decode() for k, v in app_state_raw.items()}

        # Get user state
        user_state_key = f"user:{user_id}:state"
        user_state_raw = await self.redis.hgetall(user_state_key)
        user_state = {f"{State.USER_PREFIX}{k.decode()}": v.decode() for k, v in user_state_raw.items()}

        # Merge all states
        merged_state = {**app_state, **user_state, **session_state}

        # Get events with filtering
        events = await self._get_events(session_key, config)

        # Create session object
        session = Session(
            app_name=app_name,
            user_id=user_id,
            id=session_id,
            state=merged_state,
            events=events,
            last_update_time=float(session_data[b"last_update_time"]),
        )

        return session

    @override
    async def list_sessions(
        self, *, app_name: str, user_id: str, config: ListSessionsConfig
    ) -> ListSessionsResponse:
        """List sessions for a given app and user with pagination and error handling."""
        try:
            # Find all sessions matching the app and user
            session_pattern = f"session:{app_name}:{user_id}:*"
            session_keys = await self.redis.keys(session_pattern)

            sessions = []
            for key in session_keys:
                parts = key.decode().split(":")
                if len(parts) < 4:
                    logger.warning(f"Invalid session key format: {key}")
                    continue
                session_id = parts[3]

                session_data = await self.redis.hgetall(key)
                if not session_data:
                    continue

                try:
                    last_update_time = float(session_data[b"last_update_time"])
                except (KeyError, ValueError) as e:
                    logger.error(f"Invalid session data for {session_id}: {e}")
                    continue

                session = Session(
                    app_name=app_name,
                    user_id=user_id,
                    id=session_id,
                    state={},
                    events=[],
                    last_update_time=last_update_time,
                )
                sessions.append(session)

            # Sort sessions by last_update_time descending
            sessions.sort(key=lambda s: s.last_update_time, reverse=True)

            # Apply pagination
            start = config.page * config.page_size
            end = start + config.page_size
            paginated_sessions = sessions[start:end]

            return ListSessionsResponse(
                sessions=paginated_sessions,
                total=len(sessions),
                page=config.page,
                page_size=config.page_size,
            )
        except redis.RedisError as e:
            logger.error(f"Redis error listing sessions: {e}")
            raise SessionServiceError(f"Failed to list sessions: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error listing sessions: {e}")
            raise

    @override
    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        """Delete a session and its associated data."""
        session_key = f"session:{app_name}:{user_id}:{session_id}"

        # Get session state to identify app and user state keys to potentially clean up
        session_state_key = f"{session_key}:state"
        session_state = await self.redis.hgetall(session_state_key)

        async with self.redis.pipeline(transaction=True) as pipe:
            # Delete session-specific data
            await pipe.delete(session_key)
            await pipe.delete(session_state_key)
            await pipe.delete(f"{session_key}:events")

            # Clean up app state if it's only used by this session
            app_state_key = f"app:{app_name}:state"
            if await self.redis.exists(app_state_key):
                # Count sessions using this app state
                app_sessions = await self.redis.keys(f"session:{app_name}:*:*")
                if len(app_sessions) <= 1:
                    await pipe.delete(app_state_key)

            # Clean up user state if it's only used by this session
            user_state_key = f"user:{user_id}:state"
            if await self.redis.exists(user_state_key):
                # Count sessions using this user state
                user_sessions = await self.redis.keys(f"session:*:{user_id}:*")
                if len(user_sessions) <= 1:
                    await pipe.delete(user_state_key)

            await pipe.execute()

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        """Append an event to the session's event list."""
        session_key = f"session:{session.app_name}:{session.user_id}:{session.id}"
        events_key = f"{session_key}:events"

        # Serialize event with proper handling of nested structures
        try:
            event_data = event.model_dump_json(exclude_unset=True)
        except Exception as e:
            logger.error(f"Failed to serialize event: {str(e)}")
            raise EventSerializationError(f"Could not serialize event: {str(e)}") from e
        timestamp = event.timestamp or datetime.utcnow().timestamp()

        # Add event to sorted set (score = timestamp)
        try:
            await self.redis.zadd(events_key, {event_data: timestamp})
        except redis.RedisError as e:
            logger.error(f"Redis error while appending event: {str(e)}")
            raise RedisOperationError(f"Could not append event: {str(e)}") from e

        # Update last update time
        now = datetime.utcnow().timestamp()
        await self.redis.hset(session_key, "last_update_time", now)
        session.last_update_time = now

        # Refresh TTL if specified
        if self.ttl_seconds:
            await self.redis.expire(session_key, self.ttl_seconds)
            await self.redis.expire(events_key, self.ttl_seconds)
            await self.redis.expire(f"{session_key}:state", self.ttl_seconds)

        return event

    async def _get_events(
        self, session_key: str, config: Optional[GetSessionConfig] = None
    ) -> list[Event]:
        """Get filtered events for a session."""
        events_key = f"{session_key}:events"
        config = config or GetSessionConfig()

        # Base query parameters
        min_score = config.after_timestamp or -float('inf')
        max_score = float('inf')
        start = 0
        num_events = config.num_recent_events

        # Calculate pagination parameters
        if num_events and num_events > 0:
            # For positive num_events, get the most recent ones
            # Use zrevrangebyscore to get descending order then slice
            events_data = await self.redis.zrevrangebyscore(
                events_key,
                max_score,
                min_score,
                start=0,
                num=num_events,
                withscores=True,
            )
            # Reverse to maintain ascending order
            events_data = list(reversed(events_data))
        else:
            # Use standard zrangebyscore for ascending order
            events_data = await self.redis.zrangebyscore(
                events_key,
                min_score,
                max_score,
                start=start,
                num=num_events,
                withscores=True,
            )

        # Get events from Redis sorted set
        try:
            # Get events from Redis sorted set
            event_data = await self.redis.zrangebyscore(
                events_key,
                min=min_score,
                max=max_score,
                start=start,
                num=num_events,
                withscores=False,
                score_cast_func=float,
            )
        except redis.RedisError as e:
            logger.error(f"Redis error while fetching events: {str(e)}")
            raise RedisOperationError(f"Could not retrieve events: {str(e)}") from e

        # Deserialize events
        events = []
        for data in event_data:
            try:
                event_dict = json.loads(data)
                # Handle nested state_delta if present
                if 'actions' in event_dict:
                    for action in event_dict['actions']:
                        if 'state_delta' in action and action['state_delta'] is not None:
                            action['state_delta'] = StateDelta(**action['state_delta'])
                event = Event(**event_dict)
                events.append(event)
            except Exception as e:
                logger.error(f"Failed to deserialize event: {str(e)}")
                continue

        return events

    def _merge_state(self, app_name: str, user_id: str, session: Session) -> Session:
        """Merge app and user state into session state using state delta extraction."""
        from google.adk.sessions._session_util import extract_state_delta

        # Extract state deltas
        state_delta = extract_state_delta(session.state)

        # Merge states using the extracted deltas
        merged_state = dict(session.state)
        merged_state.update(state_delta.app_state)
        merged_state.update(state_delta.user_state)

        return session.copy(update={"state": merged_state})