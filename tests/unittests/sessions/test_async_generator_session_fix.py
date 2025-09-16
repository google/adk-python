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

"""Tests for async generator handling in session services.

This module tests the fix for issue #1862 where async generators in session
state would cause pickle errors during deepcopy operations.
"""

import asyncio
import pytest
from typing import AsyncGenerator

from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions._session_copy_utils import (
    _filter_non_serializable_objects,
    _is_async_generator,
    safe_deepcopy_session,
)


async def test_async_generator() -> AsyncGenerator[str, None]:
    """A test async generator function."""
    yield "test_message_1"
    yield "test_message_2"


class TestAsyncGeneratorSessionHandling:
    """Test class for async generator handling in sessions."""

    def test_is_async_generator_detection(self):
        """Test that async generators are correctly detected."""
        async def regular_async_func():
            return "not a generator"
        
        def regular_func():
            return "regular function"
        
        # Test with actual async generator
        async_gen = test_async_generator()
        assert _is_async_generator(async_gen) is True
        
        # Test with non-generators
        assert _is_async_generator(regular_func) is False
        assert _is_async_generator("string") is False
        assert _is_async_generator(123) is False
        assert _is_async_generator([1, 2, 3]) is False
        assert _is_async_generator({"key": "value"}) is False
        
        # Clean up
        asyncio.run(async_gen.aclose())

    def test_filter_non_serializable_objects(self):
        """Test filtering of async generators from nested data structures."""
        async_gen = test_async_generator()
        
        # Test simple case
        state = {"async_tool": async_gen, "normal_data": "test_value"}
        filtered = _filter_non_serializable_objects(state)
        
        assert "normal_data" in filtered
        assert filtered["normal_data"] == "test_value"
        assert "async_tool" not in filtered
        
        # Test nested structure
        nested_state = {
            "level1": {
                "level2": {
                    "async_gen": async_gen,
                    "normal": "value"
                },
                "other": "data"
            },
            "top_level": "value"
        }
        
        filtered_nested = _filter_non_serializable_objects(nested_state)
        assert filtered_nested["level1"]["level2"]["normal"] == "value"
        assert "async_gen" not in filtered_nested["level1"]["level2"]
        assert filtered_nested["level1"]["other"] == "data"
        assert filtered_nested["top_level"] == "value"
        
        # Test list with async generator
        list_state = {"tools": [async_gen, "normal_tool"]}
        filtered_list = _filter_non_serializable_objects(list_state)
        assert len(filtered_list["tools"]) == 1
        assert filtered_list["tools"][0] == "normal_tool"
        
        # Clean up
        asyncio.run(async_gen.aclose())

    @pytest.mark.asyncio
    async def test_session_creation_with_async_generator(self):
        """Test that session creation works with async generators in state."""
        session_service = InMemorySessionService()
        async_gen = test_async_generator()
        
        # This should not raise an exception
        session = await session_service.create_session(
            app_name="test_app",
            user_id="test_user",
            state={
                "streaming_tool": async_gen,
                "normal_data": "test_value"
            }
        )
        
        # The async generator should be filtered out
        assert "streaming_tool" not in session.state
        assert "normal_data" in session.state
        assert session.state["normal_data"] == "test_value"
        
        # Clean up
        await async_gen.aclose()

    @pytest.mark.asyncio
    async def test_session_operations_with_filtered_state(self):
        """Test that all session operations work after filtering."""
        session_service = InMemorySessionService()
        
        # Create session with normal state
        session = await session_service.create_session(
            app_name="test_app",
            user_id="test_user",
            state={"normal_data": "test_value"}
        )
        
        # Test get_session
        retrieved_session = await session_service.get_session(
            app_name="test_app",
            user_id="test_user",
            session_id=session.id
        )
        assert retrieved_session is not None
        assert retrieved_session.state["normal_data"] == "test_value"
        
        # Test list_sessions
        sessions_response = await session_service.list_sessions(
            app_name="test_app",
            user_id="test_user"
        )
        assert len(sessions_response.sessions) == 1
        assert sessions_response.sessions[0].id == session.id

    def test_safe_deepcopy_session(self):
        """Test the safe_deepcopy_session function."""
        # This test would require creating a mock session object
        # For now, we test that the function exists and can be imported
        assert callable(safe_deepcopy_session)