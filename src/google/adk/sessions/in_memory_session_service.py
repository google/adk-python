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

"""
Enhanced In-memory Session Service with Global State Cache

This module provides an implementation of the session service that maintains
persistent state across agent transitions within sequential pipelines. The primary 
enhancements include:

1. Global State Cache: A shared dictionary (_GLOBAL_STATE_CACHE) accessible to all
   sessions and agents within an application. This ensures critical state values 
   persist even when session objects are copied or recreated.

2. EnhancedStateDict: A dictionary implementation for session state that automatically
   syncs with the global cache. It provides complete dictionary interface compatibility
   while ensuring any state changes are persisted globally.

3. Debug Support: Comprehensive logging of state operations to help diagnose issues
   with state persistence.

This implementation solves the common problem of state loss between agent transitions
in sequential pipelines, especially when using tools that need to share data across
multiple agents.

Usage Notes:
- All session states will automatically use EnhancedStateDict
- The LlmAgent and SequentialAgent implementations have been updated to ensure proper 
  state persistence across transitions
- No additional configuration is needed - persistence works automatically

Contributors to this solution:
- PhantomRecon team - April 2025
"""

import copy
import time
from typing import Any, Dict, Optional, Set, Iterator
import uuid
import os
import logging

from typing_extensions import override

from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListEventsResponse
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State

logger = logging.getLogger(__name__)

# Outside the class, add caching helpers for easier access
_GLOBAL_STATE_CACHE: Dict[str, Any] = {}

def _get_from_global_cache(key: str, default: Any = None) -> Any:
    """Helper to access global state cache."""
    return _GLOBAL_STATE_CACHE.get(key, default)

def _set_in_global_cache(key: str, value: Any) -> None:
    """Helper to update global state cache."""
    _GLOBAL_STATE_CACHE[key] = value
    
def _print_global_cache() -> None:
    """Print global cache for debugging."""
    logger.debug(f"Global state cache contents: {_GLOBAL_STATE_CACHE}")

class EnhancedStateDict(Dict[str, Any]):
    """
    Enhanced dictionary implementation for session state that syncs with global cache.
    This ensures state persistence across different agent runs and sessions.
    """
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        """Initialize with optional initial data."""
        super().__init__()
        if initial_data:
            self.update(initial_data)
        
        # Sync with global cache upon initialization
        for key, value in _GLOBAL_STATE_CACHE.items():
            if key not in self:
                self[key] = value
    
    def __getitem__(self, key: str) -> Any:
        """Get item with fallback to global cache."""
        # First try local state
        if key in super().__dict__:
            value = super().__getitem__(key)
            # Ensure consistency with global cache
            if key not in _GLOBAL_STATE_CACHE or _GLOBAL_STATE_CACHE[key] != value:
                _set_in_global_cache(key, value)
            return value
        
        # Try global cache
        if key in _GLOBAL_STATE_CACHE:
            value = _get_from_global_cache(key)
            # Update local state
            super().__setitem__(key, value)
            return value
        
        # Not found anywhere
        raise KeyError(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set item in both local state and global cache."""
        super().__setitem__(key, value)
        _set_in_global_cache(key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get with fallback to global cache and default."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def update(self, other: Dict[str, Any], **kwargs) -> None:
        """Update from dict and kwargs, syncing with global cache."""
        if other:
            for key, value in other.items():
                self[key] = value
        if kwargs:
            for key, value in kwargs.items():
                self[key] = value
    
    def items(self) -> Iterator:
        """Return all items, including those from global cache."""
        # Create a combined view of local and global keys
        combined = dict(_GLOBAL_STATE_CACHE)
        local_dict = dict(super().items())
        combined.update(local_dict)
        return combined.items()
    
    def keys(self) -> Set[str]:
        """Return all keys, including those from global cache."""
        return set(super().keys()).union(_GLOBAL_STATE_CACHE.keys())
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in either local state or global cache."""
        return super().__contains__(key) or key in _GLOBAL_STATE_CACHE

class InMemorySessionService(BaseSessionService):
  """An in-memory implementation of the session service."""

  def __init__(self, debug_mode: bool = False):
    # A map from app name to a map from user ID to a map from session ID to session.
    self.sessions: dict[str, dict[str, dict[str, Session]]] = {}
    # A map from app name to a map from user ID to a map from key to the value.
    self.user_state: dict[str, dict[str, dict[str, Any]]] = {}
    # A map from app name to a map from key to the value.
    self.app_state: dict[str, dict[str, Any]] = {}
    # A single shared global state cache for all sessions
    self.global_state_cache: dict[str, Any] = _GLOBAL_STATE_CACHE
    # Enable verbose logging
    self.debug_mode = debug_mode or os.environ.get('DEBUG', '0') == '1'
    if self.debug_mode:
      logger.setLevel(logging.DEBUG)
      logger.debug("InMemorySessionService initialized with DEBUG mode")

  @override
  def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    session_id = (
        session_id.strip()
        if session_id and session_id.strip()
        else str(uuid.uuid4())
    )
    
    if self.debug_mode:
      logger.debug(f"Creating new session {session_id} for app {app_name} and user {user_id}")
      if state:
        logger.debug(f"Initial state keys: {list(state.keys())}")
    
    # Use our enhanced state dictionary to ensure persistence
    initial_state = EnhancedStateDict(state or {})
    # Add any existing global state to initial state
    for key, value in self.global_state_cache.items():
      if key not in initial_state:
        initial_state[key] = value
        if self.debug_mode:
          logger.debug(f"Added key {key} from global cache to initial state")
    
    session = Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=initial_state,
        last_update_time=time.time(),
    )

    if app_name not in self.sessions:
      self.sessions[app_name] = {}
    if user_id not in self.sessions[app_name]:
      self.sessions[app_name][user_id] = {}
    self.sessions[app_name][user_id][session_id] = session

    copied_session = copy.deepcopy(session)
    return self._merge_state(app_name, user_id, copied_session)

  @override
  def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Session:
    if app_name not in self.sessions:
      if self.debug_mode:
        logger.debug(f"get_session: App {app_name} not found in sessions")
      return None
    if user_id not in self.sessions[app_name]:
      if self.debug_mode:
        logger.debug(f"get_session: User {user_id} not found in app {app_name}")
      return None
    if session_id not in self.sessions[app_name][user_id]:
      if self.debug_mode:
        logger.debug(f"get_session: Session {session_id} not found for user {user_id} in app {app_name}")
      return None

    session = self.sessions[app_name][user_id].get(session_id)
    
    if self.debug_mode:
      logger.debug(f"get_session: Retrieved session {session_id} with state keys: {list(session.state.keys())}")
      
    # Ensure session state is an EnhancedStateDict
    if not isinstance(session.state, EnhancedStateDict):
      session.state = EnhancedStateDict(session.state)
      if self.debug_mode:
        logger.debug(f"get_session: Upgraded session state to EnhancedStateDict")
    
    # Ensure session has all global cache entries
    for key, value in self.global_state_cache.items():
      if key not in session.state:
        session.state[key] = value
        if self.debug_mode:
          logger.debug(f"get_session: Added missing key {key} from global cache to session {session_id}")
    
    copied_session = copy.deepcopy(session)

    if config:
      if config.num_recent_events:
        copied_session.events = copied_session.events[
            -config.num_recent_events :
        ]
      elif config.after_timestamp:
        i = len(session.events) - 1
        while i >= 0:
          if copied_session.events[i].timestamp < config.after_timestamp:
            break
          i -= 1
        if i >= 0:
          copied_session.events = copied_session.events[i:]

    return self._merge_state(app_name, user_id, copied_session)

  def _merge_state(self, app_name: str, user_id: str, copied_session: Session):
    # Ensure session state is an EnhancedStateDict
    if not isinstance(copied_session.state, EnhancedStateDict):
      copied_session.state = EnhancedStateDict(copied_session.state)
      if self.debug_mode:
        logger.debug(f"_merge_state: Upgraded copied session state to EnhancedStateDict")

    # Merge app state
    if app_name in self.app_state:
      for key in self.app_state[app_name].keys():
        copied_session.state[State.APP_PREFIX + key] = self.app_state[app_name][
            key
        ]

    # Merge global state cache
    for key, value in self.global_state_cache.items():
      if key not in copied_session.state:
        copied_session.state[key] = value
        if self.debug_mode:
          logger.debug(f"_merge_state: Added key {key} from global cache to session")

    if (
        app_name not in self.user_state
        or user_id not in self.user_state[app_name]
    ):
      return copied_session

    # Merge session state with user state.
    for key in self.user_state[app_name][user_id].keys():
      copied_session.state[State.USER_PREFIX + key] = self.user_state[app_name][
          user_id
      ][key]
    return copied_session

  @override
  def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    empty_response = ListSessionsResponse()
    if app_name not in self.sessions:
      return empty_response
    if user_id not in self.sessions[app_name]:
      return empty_response

    sessions_without_events = []
    for session in self.sessions[app_name][user_id].values():
      copied_session = copy.deepcopy(session)
      copied_session.events = []
      copied_session.state = {}
      sessions_without_events.append(copied_session)
    return ListSessionsResponse(sessions=sessions_without_events)

  @override
  def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    if (
        self.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        is None
    ):
      return None

    self.sessions[app_name][user_id].pop(session_id)

  @override
  def append_event(self, session: Session, event: Event) -> Event:
    # Update the in-memory session.
    super().append_event(session=session, event=event)
    session.last_update_time = event.timestamp

    # Update the storage session
    app_name = session.app_name
    user_id = session.user_id
    session_id = session.id
    if app_name not in self.sessions:
      return event
    if user_id not in self.sessions[app_name]:
      return event
    if session_id not in self.sessions[app_name][user_id]:
      return event

    if event.actions and event.actions.state_delta:
      for key in event.actions.state_delta:
        # Also store ALL state changes in the global cache for persistence
        value = event.actions.state_delta[key]
        self.global_state_cache[key] = value
        
        if self.debug_mode:
          logger.debug(f"append_event: Added/updated {key} in global cache")
        
        if key.startswith(State.APP_PREFIX):
          self.app_state.setdefault(app_name, {})[
              key.removeprefix(State.APP_PREFIX)
          ] = value

        if key.startswith(State.USER_PREFIX):
          self.user_state.setdefault(app_name, {}).setdefault(user_id, {})[
              key.removeprefix(State.USER_PREFIX)
          ] = value

    storage_session = self.sessions[app_name][user_id].get(session_id)
    
    # Ensure storage_session.state is an EnhancedStateDict
    if not isinstance(storage_session.state, EnhancedStateDict):
      storage_session.state = EnhancedStateDict(storage_session.state)
      if self.debug_mode:
        logger.debug(f"append_event: Upgraded storage session state to EnhancedStateDict")
        
    super().append_event(session=storage_session, event=event)

    storage_session.last_update_time = event.timestamp
    
    if self.debug_mode:
      logger.debug(f"append_event: State delta added with keys: {list(event.actions.state_delta.keys()) if event.actions and event.actions.state_delta else 'None'}")
      logger.debug(f"append_event: Session {session_id} now has state keys: {list(storage_session.state.keys())}")
      logger.debug(f"append_event: Global cache keys: {list(self.global_state_cache.keys())}")

    return event

  @override
  def list_events(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
  ) -> ListEventsResponse:
    raise NotImplementedError()

  def _get_session(self, app_name: str, user_id: str,
                session_id: str) -> Session:
    """Gets or creates a session for the given app name, user ID, and session ID.

    Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      session_id: The ID of the session.

    Returns:
      The session.
    """
    if app_name not in self.sessions:
      self.sessions[app_name] = {}
    if user_id not in self.sessions[app_name]:
      self.sessions[app_name][user_id] = {}
    if session_id not in self.sessions[app_name][user_id]:
      # Initialize with global state cache to maintain persistent state
      init_state = EnhancedStateDict()
      # Initialize from global cache if available
      for key, value in self.global_state_cache.items():
          init_state[key] = value
      
      self.sessions[app_name][user_id][session_id] = Session(
          app_name=app_name,
          user_id=user_id,
          id=session_id,
          state=init_state,
          last_update_time=time.time(),
      )
      if self.debug_mode:
        logger.debug(f"Created new session: {app_name}/{user_id}/{session_id}")
    return self.sessions[app_name][user_id][session_id]

  def get_state(self, app_name: str, user_id: str, session_id: str,
                key: str) -> Any:
    """Gets a value from the session state.

    Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      session_id: The ID of the session.
      key: The state key.

    Returns:
      The state value, or None if the key doesn't exist.
    """
    session = self._get_session(app_name, user_id, session_id)
    
    # First check session state
    if key in session.state:
        value = session.state.get(key)
        if self.debug_mode:
            logger.debug(f"Get state from session {app_name}/{user_id}/{session_id}: {key} = {value}")
        return value
    
    # Then check global cache
    if key in self.global_state_cache:
        value = self.global_state_cache[key]
        # Update session state for future access
        session.state[key] = value
        if self.debug_mode:
            logger.debug(f"Get state from global cache {app_name}/{user_id}/{session_id}: {key} = {value}")
        return value
    
    # Not found
    if self.debug_mode:
        logger.debug(f"Key not found in either session or global cache: {key}")
    return None

  def update_state(self, app_name: str, user_id: str, session_id: str, key: str,
                  value: Any) -> None:
    """Updates a value in the session state.

    Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      session_id: The ID of the session.
      key: The state key.
      value: The state value.
    """
    session = self._get_session(app_name, user_id, session_id)
    
    # Update session state
    session.state[key] = value
    
    # Also update global cache for persistence
    self.global_state_cache[key] = value
    
    if self.debug_mode:
      logger.debug(f"Updated state {app_name}/{user_id}/{session_id}: {key} = {value}")
      
    # Special handling for APP and USER prefixed keys
    if key.startswith(State.APP_PREFIX):
        app_key = key.removeprefix(State.APP_PREFIX)
        self.app_state.setdefault(app_name, {})[app_key] = value
        if self.debug_mode:
            logger.debug(f"Updated app state: {app_name}.{app_key} = {value}")
            
    if key.startswith(State.USER_PREFIX):
        user_key = key.removeprefix(State.USER_PREFIX)
        self.user_state.setdefault(app_name, {}).setdefault(user_id, {})[user_key] = value
        if self.debug_mode:
            logger.debug(f"Updated user state: {app_name}.{user_id}.{user_key} = {value}")

  # Convenience methods for global state cache
  def get_from_global_cache(self, key: str) -> Any:
    """Gets a value from the global state cache.
    
    Args:
      key: The state key.
      
    Returns:
      The state value, or None if the key doesn't exist.
    """
    return self.global_state_cache.get(key)
    
  def set_in_global_cache(self, key: str, value: Any) -> None:
    """Sets a value in the global state cache.
    
    Args:
      key: The state key.
      value: The state value.
    """
    self.global_state_cache[key] = value
    if self.debug_mode:
        logger.debug(f"Set in global cache: {key} = {value}")
    
  def print_global_cache(self) -> None:
    """Prints the contents of the global state cache for debugging."""
    keys = list(self.global_state_cache.keys())
    count = len(keys)
    if self.debug_mode:
        logger.debug(f"Global state cache has {count} keys: {keys}")
    else:
        logger.info(f"Global state cache has {count} keys")
