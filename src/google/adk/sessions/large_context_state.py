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
Enhanced State class for handling large context windows efficiently.

This module extends the ADK State class to provide efficient handling of large context
windows (1M-2M tokens) using a reference-based approach.
"""

import json
from typing import Dict, Any, Optional, List

from google.adk.sessions.state import State
from google.adk.sessions.context_reference_store import ContextReferenceStore


class LargeContextState(State):
    """
    Enhanced State class for efficient handling of large contexts with Gemini.

    This class extends ADK's State to handle large contexts efficiently by:
    - Storing references to contexts instead of the contexts themselves
    - Providing methods to resolve references when needed
    - Supporting Gemini's context caching feature for cost optimization
    - Handling both text and structured contexts
    """

    def __init__(
        self,
        value: Dict[str, Any] = None,
        delta: Dict[str, Any] = None,
        context_store: Optional[ContextReferenceStore] = None,
    ):
        """

        Args:
            value: The current value of the state dict
            delta: The delta change to the current value that hasn't been committed
            context_store: Context reference store to use
        """
        super().__init__(value=value or {}, delta=delta or {})
        self._context_store = context_store or ContextReferenceStore()

    def add_large_context(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        key: str = "context_ref",
    ) -> str:
        """
        Add large context to the state using reference-based storage.

        Args:
            content: The context content to store (string or structured data)
            metadata: Optional metadata about the context
            key: The key to store the reference under in the state

        Returns:
            The reference ID for the stored context
        """
        context_id = self._context_store.store(content, metadata)
        self[key] = context_id
        return context_id

    def get_context(self, ref_key: str = "context_ref") -> Any:
        """
        Retrieve context from a reference stored in the state.

        Args:
            ref_key: The key where the context reference is stored

        Returns:
            The context content
        """
        if ref_key not in self:
            raise KeyError(f"Context reference key '{ref_key}' not found in state")

        context_id = self[ref_key]
        return self._context_store.retrieve(context_id)

    def with_cache_hint(self, ref_key: str = "context_ref") -> Dict[str, Any]:
        """
        Get a cache hint object for Gemini API calls.

        This allows the Gemini API to cache the context for reuse.
        According to Gemini API docs, context caching can significantly
        reduce costs when reusing the same context multiple times.

        Args:
            ref_key: The key where the context reference is stored

        Returns:
            A cache hint object suitable for passing to Gemini API
        """
        if ref_key not in self:
            raise KeyError(f"Context reference key '{ref_key}' not found in state")

        context_id = self[ref_key]
        return self._context_store.get_cache_hint(context_id)

    def store_structured_context(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        key: str = "structured_context_ref",
    ) -> str:
        """
        Store structured data (JSON/dict) in the context store.

        Args:
            data: The structured data to store
            metadata: Optional metadata about the context
            key: The key to store the reference under in the state

        Returns:
            The reference ID for the stored context
        """
        if metadata is None:
            metadata = {}

        # Ensure we mark this as structured data if not already specified
        if "content_type" not in metadata:
            metadata["content_type"] = "application/json"

        return self.add_large_context(data, metadata, key)
