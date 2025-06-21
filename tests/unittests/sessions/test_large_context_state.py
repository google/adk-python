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

"""Tests for large context state and context reference store."""

import json
import time
import pytest
from typing import Dict, Any

# Force the importing of conftest.py which sets up our mock modules
from . import conftest

# Now import the modules we want to test
from google.adk.sessions.context_reference_store import (
    ContextReferenceStore,
    ContextMetadata,
)
from google.adk.sessions.large_context_state import LargeContextState


class TestContextReferenceStore:
    """Tests for ContextReferenceStore."""

    def test_store_and_retrieve_text(self):
        """Test storing and retrieving text content."""
        store = ContextReferenceStore()
        content = (
            "This is a large text content" * 100
        )  # This is not large, but sufficient for testing
        context_id = store.store(content)

        # Verify content can be retrieved
        retrieved = store.retrieve(context_id)
        assert content == retrieved

        # Verify metadata is created
        metadata = store.get_metadata(context_id)
        assert isinstance(metadata, ContextMetadata)
        assert metadata.content_type == "text/plain"
        assert not metadata.is_structured

    def test_store_and_retrieve_structured(self):
        """Test storing and retrieving structured content."""
        store = ContextReferenceStore()
        content = {
            "title": "Test Document",
            "sections": [
                {"id": "section-1", "title": "Section 1", "content": "Content 1"},
                {"id": "section-2", "title": "Section 2", "content": "Content 2"},
            ],
        }
        context_id = store.store(content)

        # Verify content can be retrieved
        retrieved = store.retrieve(context_id)
        assert content == retrieved

        # Verify metadata is created
        metadata = store.get_metadata(context_id)
        assert isinstance(metadata, ContextMetadata)
        assert metadata.content_type == "application/json"
        assert metadata.is_structured

    def test_duplicate_content_deduplication(self):
        """Test that storing the same content twice returns the same ID."""
        store = ContextReferenceStore()
        content = "This is a duplicate content"

        # Store the same content twice
        id1 = store.store(content)
        id2 = store.store(content)

        # Verify both IDs are the same
        assert id1 == id2

    def test_cache_hint(self):
        """Test getting cache hints for stored content."""
        store = ContextReferenceStore()

        # Store content with a cache TTL
        content = "Content with cache TTL"
        metadata = {"cache_ttl": 3600}  # 1 hour cache
        context_id = store.store(content, metadata)

        # Get cache hint
        cache_hint = store.get_cache_hint(context_id)

        # Verify cache hint has expected fields
        assert "cache_id" in cache_hint
        assert cache_hint["cache_level"] == "HIGH"
        assert "ttl_seconds" in cache_hint
        assert cache_hint["ttl_seconds"] <= 3600

    def test_cache_management(self):
        """Test cache size management."""
        # Create a store with small cache size
        store = ContextReferenceStore(cache_size=2)

        # Store 3 different items, which should evict the first one
        id1 = store.store("Content 1")
        time.sleep(0.01)  # Ensure different access times
        id2 = store.store("Content 2")
        time.sleep(0.01)
        id3 = store.store("Content 3")

        # First content should be evicted
        with pytest.raises(KeyError):
            store.retrieve(id1)

        # The other two should still be accessible
        assert "Content 2" == store.retrieve(id2)
        assert "Content 3" == store.retrieve(id3)


class TestLargeContextState:
    """Tests for LargeContextState."""

    def test_add_and_get_context(self):
        """Test adding and retrieving context."""
        state = LargeContextState()
        content = "This is a test context"

        # Add context
        ref_id = state.add_large_context(content)

        # Verify reference is stored in state
        assert "context_ref" in state
        assert state["context_ref"] == ref_id

        # Retrieve context
        retrieved = state.get_context()
        assert content == retrieved

    def test_add_and_get_structured_context(self):
        """Test adding and retrieving structured context."""
        state = LargeContextState()
        content = {"key": "value", "nested": {"subkey": "subvalue"}}

        # Add structured context
        ref_id = state.store_structured_context(content)

        # Verify reference is stored in state
        assert "structured_context_ref" in state
        assert state["structured_context_ref"] == ref_id

        # Retrieve context
        retrieved = state.get_context("structured_context_ref")
        assert content == retrieved

    def test_with_cache_hint(self):
        """Test getting cache hints from state."""
        state = LargeContextState()
        content = "Content for caching"
        metadata = {"cache_ttl": 1800}  # 30 minutes

        # Add context with cache metadata
        state.add_large_context(content, metadata)

        # Get cache hint
        cache_hint = state.with_cache_hint()

        # Verify cache hint
        assert "cache_id" in cache_hint
        assert cache_hint["cache_level"] == "HIGH"
        assert "ttl_seconds" in cache_hint
        assert cache_hint["ttl_seconds"] <= 1800

    def test_context_not_found(self):
        """Test error handling when context is not found."""
        state = LargeContextState()

        # Attempt to retrieve non-existent context
        with pytest.raises(KeyError):
            state.get_context("nonexistent_ref")
