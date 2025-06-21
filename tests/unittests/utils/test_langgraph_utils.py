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

"""Tests for LangGraph context management utilities."""

import pytest
from typing import Dict, Any, List

# Force the importing of conftest.py which sets up our mock modules
from . import conftest

# Now import the modules we want to test
from google.adk.sessions.context_reference_store import ContextReferenceStore
from google.adk.utils.langgraph_utils import (
    LangGraphContextManager,
    create_reference_aware_merge,
)


class TestLangGraphContextManager:
    """Tests for LangGraphContextManager."""

    def test_add_to_state(self):
        """Test adding content to state."""
        manager = LangGraphContextManager()
        original_state = {"counter": 1, "messages": ["Hello"]}

        # Add content to state
        content = "This is test content"
        new_state = manager.add_to_state(original_state, content)

        # Verify state is updated with reference
        assert "context_ref" in new_state
        assert new_state["counter"] == 1  # Original data preserved
        assert new_state["messages"] == ["Hello"]  # Original data preserved

        # Verify original state is not modified
        assert "context_ref" not in original_state

    def test_retrieve_from_state(self):
        """Test retrieving content from state."""
        manager = LangGraphContextManager()
        content = "This is retrievable content"

        # Add content to state
        state = manager.add_to_state({}, content)

        # Retrieve content
        retrieved = manager.retrieve_from_state(state)
        assert content == retrieved

        # Test with custom key
        custom_state = manager.add_to_state(
            {}, "Custom key content", ref_key="custom_ref"
        )
        custom_retrieved = manager.retrieve_from_state(custom_state, "custom_ref")
        assert "Custom key content" == custom_retrieved

    def test_reference_aware_merge(self):
        """Test reference-aware merge function."""
        # Create merge function
        merge_fn = create_reference_aware_merge()

        # Create states with references
        context_store = ContextReferenceStore()
        ref1 = context_store.store("Content 1")
        ref2 = context_store.store("Content 2")

        # Create states to merge
        left: Dict[str, Any] = {
            "context_ref": ref1,
            "counter": 1,
            "messages": ["First message"],
        }

        right: Dict[str, Any] = {
            "context_ref": ref2,  # New reference that should replace the old one
            "counter": 2,
            "messages": ["Second message"],
        }

        # Merge states
        merged = merge_fn(left, right)

        # Verify merge results
        assert merged["context_ref"] == ref2  # Right reference preferred
        assert merged["counter"] == 2  # Right value preferred
        assert merged["messages"] == ["Second message"]  # Right value preferred

    def test_reference_aware_merge_partial_update(self):
        """Test reference-aware merge with partial updates."""
        # Create merge function
        merge_fn = create_reference_aware_merge()

        # Create states with references
        context_store = ContextReferenceStore()
        ref1 = context_store.store("Content 1")

        # Create states to merge
        left: Dict[str, Any] = {
            "context_ref": ref1,
            "other_ref": "other-ref-1",
            "counter": 1,
            "messages": ["First message"],
        }

        right: Dict[str, Any] = {
            # Note: no context_ref in right state
            "counter": 2,
            # Note: no messages in right state
        }

        # Merge states
        merged = merge_fn(left, right)

        # Verify merge results
        assert merged["context_ref"] == ref1  # Left reference preserved
        assert merged["other_ref"] == "other-ref-1"  # Left reference preserved
        assert merged["counter"] == 2  # Right value used
        assert merged["messages"] == ["First message"]  # Left value preserved
