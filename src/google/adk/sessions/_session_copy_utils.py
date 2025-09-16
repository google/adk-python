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

"""Utilities for safely copying session objects that may contain non-serializable objects."""

from __future__ import annotations

import copy
import inspect
import logging
from typing import Any

logger = logging.getLogger('google_adk.' + __name__)


def _is_async_generator(obj: Any) -> bool:
    """Check if an object is an async generator."""
    return inspect.isasyncgen(obj)


def _filter_non_serializable_objects(obj: Any, path: str = "root") -> Any:
    """Recursively filter out non-serializable objects from a data structure.
    
    Args:
        obj: The object to filter
        path: The current path in the object tree (for logging)
        
    Returns:
        A copy of the object with non-serializable objects removed
    """
    if _is_async_generator(obj):
        logger.warning(
            f"Removing async generator from session state at {path}. "
            "Async generators cannot be persisted in session state."
        )
        return None
    
    if isinstance(obj, dict):
        filtered_dict = {}
        for key, value in obj.items():
            filtered_value = _filter_non_serializable_objects(value, f"{path}.{key}")
            if filtered_value is not None:
                filtered_dict[key] = filtered_value
        return filtered_dict
    
    elif isinstance(obj, (list, tuple)):
        filtered_items = []
        for i, item in enumerate(obj):
            filtered_item = _filter_non_serializable_objects(item, f"{path}[{i}]")
            if filtered_item is not None:
                filtered_items.append(filtered_item)
        return type(obj)(filtered_items)
    
    # For other types, assume they're serializable
    return obj


def safe_deepcopy_session(session):
    """Safely deepcopy a session object, filtering out non-serializable objects.
    
    This function creates a deep copy of a session while filtering out objects
    that cannot be pickled, such as async generators.
    
    Args:
        session: The session object to copy
        
    Returns:
        A deep copy of the session with non-serializable objects filtered out
    """
    # Create a shallow copy first
    session_copy = copy.copy(session)
    
    # Deep copy the state while filtering non-serializable objects
    if hasattr(session_copy, 'state') and session_copy.state:
        session_copy.state = _filter_non_serializable_objects(session_copy.state, "state")
        # Now we can safely deepcopy the filtered state
        session_copy.state = copy.deepcopy(session_copy.state)
    
    # Deep copy other attributes that should be safe
    if hasattr(session_copy, 'events'):
        session_copy.events = copy.deepcopy(session.events)
    
    return session_copy