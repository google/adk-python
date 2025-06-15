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
Context Reference Store for Efficient Management of Large Context Windows

This module implements a solution for efficiently managing large context windows (1M-2M tokens)
by using a reference-based approach rather than direct context passing.
"""

import time
import json
import uuid
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ContextMetadata:
    """Metadata for stored context."""

    content_type: str = "text/plain"
    token_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    cache_id: Optional[str] = None
    cached_until: Optional[float] = None  # Timestamp when cache expires
    is_structured: bool = False  # Whether this is JSON or not

    def update_access_stats(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class ContextReferenceStore:
    """
    A store for large contexts that provides reference-based access.

    This class allows large contexts to be stored once and referenced by ID,
    preventing unnecessary duplication and serialization of large data.
    """

    def __init__(self, cache_size: int = 50):
        """
        Args:
            cache_size: Maximum number of contexts to keep in memory
        """
        self._contexts: Dict[str, str] = {}
        self._metadata: Dict[str, ContextMetadata] = {}
        self._lru_cache_size = cache_size

    def store(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store context and return a reference ID.

        Args:
            content: The context content to store (string or structured data)
            metadata: Optional metadata about the context

        Returns:
            A reference ID for the stored context
        """
        # Handle both string and structured data (like JSON objects)
        is_structured = not isinstance(content, str)

        # Convert structured data to string for storage
        if is_structured:
            content_str = json.dumps(content)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
        else:
            content_str = content
            content_hash = hashlib.md5(content.encode()).hexdigest()

        # Check if we already have this content
        for context_id, existing_content in self._contexts.items():
            existing_hash = hashlib.md5(existing_content.encode()).hexdigest()
            if (
                existing_hash == content_hash
                and self._metadata[context_id].is_structured == is_structured
            ):
                # Update access stats
                self._metadata[context_id].update_access_stats()
                return context_id

        # Generate a new ID if not found
        context_id = str(uuid.uuid4())

        self._contexts[context_id] = content_str

        # Set content type based on input type
        if is_structured:
            content_type = "application/json"
        else:
            content_type = (
                metadata.get("content_type", "text/plain") if metadata else "text/plain"
            )

        # Create and store metadata
        meta = ContextMetadata(
            content_type=content_type,
            token_count=len(content_str) // 4,  # This is a rough approximation
            tags=metadata.get("tags", []) if metadata else [],
            is_structured=is_structured,
        )

        # Generate a cache ID for Gemini caching
        if metadata and "cache_id" in metadata:
            meta.cache_id = metadata["cache_id"]
        else:
            meta.cache_id = f"context_{content_hash[:16]}"

        # Set cache expiration if provided
        if metadata and "cache_ttl" in metadata:
            ttl_seconds = metadata["cache_ttl"]
            meta.cached_until = time.time() + ttl_seconds

        self._metadata[context_id] = meta

        self._manage_cache()

        return context_id

    def retrieve(self, context_id: str) -> Any:
        """
        Retrieve context by its reference ID.

        Args:
            context_id: The reference ID for the context

        Returns:
            The context content (string or structured data depending on how it was stored)
        """
        if context_id not in self._contexts:
            raise KeyError(f"Context ID {context_id} not found")

        # Update access stats
        self._metadata[context_id].update_access_stats()

        # Get the content and metadata
        content = self._contexts[context_id]
        metadata = self._metadata[context_id]

        # If the content is structured (JSON), parse it back
        if metadata.is_structured:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback to returning as string if JSON parsing fails
                return content

        return content

    def get_metadata(self, context_id: str) -> ContextMetadata:
        """Get metadata for a context."""
        if context_id not in self._metadata:
            raise KeyError(f"Context ID {context_id} not found")
        return self._metadata[context_id]

    def _manage_cache(self):
        """Manage the cache size by removing least recently used contexts."""
        if len(self._contexts) <= self._lru_cache_size:
            return

        # Sort by last accessed time
        sorted_contexts = sorted(
            self._metadata.items(), key=lambda x: x[1].last_accessed
        )

        # Remove oldest contexts until we're under the limit
        contexts_to_remove = len(self._contexts) - self._lru_cache_size
        for i in range(contexts_to_remove):
            context_id = sorted_contexts[i][0]
            del self._contexts[context_id]
            del self._metadata[context_id]

    def get_cache_hint(self, context_id: str) -> Dict[str, Any]:
        """
        Get a cache hint object for Gemini API calls.

        This allows the Gemini API to cache the context for reuse.
        According to Gemini API docs, context caching can significantly
        reduce costs when reusing the same context multiple times.
        """
        if context_id not in self._metadata:
            raise KeyError(f"Context ID {context_id} not found")

        metadata = self._metadata[context_id]

        # Create cache hint with recommended parameters
        cache_hint = {
            "cache_id": metadata.cache_id,
            "cache_level": "HIGH",  # Strong caching for this context
        }

        # If we have a cached_until timestamp, add it
        if metadata.cached_until:
            now = time.time()
            if metadata.cached_until > now:
                # Still valid, calculate remaining TTL in seconds
                cache_hint["ttl_seconds"] = int(metadata.cached_until - now)

        return cache_hint
