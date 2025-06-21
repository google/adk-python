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
LangGraph Utilities for Context Management

This module provides utilities for integrating ADK's context management with LangGraph.
It focuses on making it easy to use efficient context reference storage with LangGraph state.
"""

from typing import Dict, Any, Optional, TypeVar, List, Callable

from google.adk.sessions.context_reference_store import ContextReferenceStore


StateType = TypeVar("StateType", bound=Dict[str, Any])


class LangGraphContextManager:
    """
    Context manager for LangGraph applications.

    Provides methods to integrate ADK's context reference store with LangGraph state.
    """

    def __init__(self, context_store: Optional[ContextReferenceStore] = None):
        """

        Args:
            context_store: Context reference store to use
        """
        self._context_store = context_store or ContextReferenceStore()

    def add_to_state(
        self,
        state: StateType,
        content: Any,
        ref_key: str = "context_ref",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateType:
        """
        Add content to context store and reference it in state.

        Args:
            state: LangGraph state dict
            content: The content to store
            ref_key: Key to store the reference under
            metadata: Optional metadata about the content

        Returns:
            Updated state dict with reference added
        """
        # Create a copy of the state to avoid modifying the original
        new_state = state.copy()

        # Store content and get a reference ID
        context_id = self._context_store.store(content, metadata)

        new_state[ref_key] = context_id  # store the reference to the state

        return new_state

    def retrieve_from_state(
        self, state: StateType, ref_key: str = "context_ref"
    ) -> Any:
        """
        Retrieve content from a reference in the state.

        Args:
            state: LangGraph state dict
            ref_key: Key where the reference is stored

        Returns:
            The retrieved content
        """
        if ref_key not in state:
            raise KeyError(f"Context reference key '{ref_key}' not found in state")

        context_id = state[ref_key]
        return self._context_store.retrieve(context_id)


def create_reference_aware_merge(
    context_store: Optional[ContextReferenceStore] = None,
) -> Callable[[StateType, StateType], StateType]:
    """
    Create a merge function for LangGraph that's aware of context references.

    This merge function handles special merging of reference keys, ensuring that
    the reference itself is passed rather than trying to merge the content.

    Args:
        context_store: Context reference store to use

    Returns:
        A merge function that can be used with LangGraph's StateGraph
    """
    store = context_store or ContextReferenceStore()

    def reference_aware_merge(left: StateType, right: StateType) -> StateType:
        """
        Merge two state dicts with awareness of context references.

        Args:
            left: First state dict
            right: Second state dict

        Returns:
            Merged state dict
        """
        # Start with a copy of the left dict
        result = left.copy()

        # Process all keys in the right dict
        for key, value in right.items():

            if key.endswith("_ref") and key in left:
                result[key] = value
            else:
                result[key] = value

        return result

    return reference_aware_merge
