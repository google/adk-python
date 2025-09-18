"""
Contract tests for GET /api/v1/chat/stream endpoint
Tests the streaming chat functionality that will be implemented.
These tests should FAIL initially as part of TDD approach.
"""

import pytest
import httpx
import json
from typing import AsyncGenerator, Dict, Any


class TestChatStreamEndpoint:
    """Test the chat stream endpoint contract"""

    BASE_URL = "http://localhost:8000"
    ENDPOINT = "/api/v1/chat/stream"

    @pytest.mark.asyncio
    async def test_chat_stream_basic_query(self):
        """Test basic streaming chat functionality"""
        params = {
            "message": "What security policies are active?",
            "session_id": "test_session_123"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "GET",
                f"{self.BASE_URL}{self.ENDPOINT}",
                params=params
            ) as response:
                # This should fail initially - endpoint doesn't exist yet
                assert response.status_code == 200
                assert response.headers.get("content-type") == "text/event-stream"

                chunks = []
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        chunks.append(chunk.strip())

                # Should receive at least one chunk
                assert len(chunks) > 0

                # Check for SSE format
                for chunk in chunks:
                    if chunk.startswith("data:"):
                        data_part = chunk[5:].strip()
                        if data_part and data_part != "[DONE]":
                            # Should be valid JSON
                            json.loads(data_part)

    @pytest.mark.asyncio
    async def test_chat_stream_with_tools(self):
        """Test streaming chat with tool usage"""
        params = {
            "message": "Query the database for IAM policies",
            "session_id": "test_session_tools",
            "use_tools": "true"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "GET",
                f"{self.BASE_URL}{self.ENDPOINT}",
                params=params
            ) as response:
                assert response.status_code == 200

                tool_usage_detected = False
                chunks = []

                async for chunk in response.aiter_text():
                    if chunk.strip():
                        chunks.append(chunk.strip())

                        if "data:" in chunk:
                            try:
                                data = json.loads(chunk.split("data:", 1)[1].strip())
                                if "tool_use" in data or "tool_call" in data:
                                    tool_usage_detected = True
                            except (json.JSONDecodeError, IndexError):
                                continue

                # Should detect tool usage for database queries
                assert tool_usage_detected, "No tool usage detected in stream"

    @pytest.mark.asyncio
    async def test_chat_stream_missing_message(self):
        """Test streaming chat without required message parameter"""
        params = {
            "session_id": "test_session_no_msg"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}{self.ENDPOINT}",
                params=params
            )

        # Should return 422 for missing required parameter
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_stream_empty_message(self):
        """Test streaming chat with empty message"""
        params = {
            "message": "",
            "session_id": "test_session_empty"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}{self.ENDPOINT}",
                params=params
            )

        # Should return 400 for empty message
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_stream_long_conversation(self):
        """Test streaming chat in long conversation context"""
        session_id = "test_session_long_conv"

        # Simulate multiple messages in same session
        messages = [
            "List all GCP projects",
            "What are the IAM policies for the first project?",
            "Are there any security recommendations?"
        ]

        for i, message in enumerate(messages):
            params = {
                "message": message,
                "session_id": session_id
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "GET",
                    f"{self.BASE_URL}{self.ENDPOINT}",
                    params=params
                ) as response:
                    assert response.status_code == 200

                    chunks_received = 0
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            chunks_received += 1

                    # Should receive response chunks
                    assert chunks_received > 0, f"No chunks received for message {i+1}"

    @pytest.mark.asyncio
    async def test_chat_stream_session_persistence(self):
        """Test that session context is maintained across stream requests"""
        session_id = "test_session_persistence"

        # First message
        params1 = {
            "message": "Remember that my project is called 'test-project'",
            "session_id": session_id
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "GET",
                f"{self.BASE_URL}{self.ENDPOINT}",
                params=params1
            ) as response:
                assert response.status_code == 200

                # Consume the stream
                async for chunk in response.aiter_text():
                    pass

        # Second message referencing previous context
        params2 = {
            "message": "What is the name of my project?",
            "session_id": session_id
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "GET",
                f"{self.BASE_URL}{self.ENDPOINT}",
                params=params2
            ) as response:
                assert response.status_code == 200

                full_response = ""
                async for chunk in response.aiter_text():
                    if chunk.strip() and chunk.startswith("data:"):
                        try:
                            data = json.loads(chunk[5:].strip())
                            if "content" in data:
                                full_response += data["content"]
                        except json.JSONDecodeError:
                            continue

                # Should reference the project name from context
                assert "test-project" in full_response.lower()

    @pytest.mark.asyncio
    async def test_chat_stream_error_handling(self):
        """Test error handling in streaming responses"""
        params = {
            "message": "FORCE_ERROR_FOR_TESTING",
            "session_id": "test_session_error"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "GET",
                f"{self.BASE_URL}{self.ENDPOINT}",
                params=params
            ) as response:
                # Even with errors, should maintain streaming format
                if response.status_code == 200:
                    error_chunk_found = False
                    async for chunk in response.aiter_text():
                        if chunk.strip() and "error" in chunk.lower():
                            error_chunk_found = True
                            break

                    # Should handle errors gracefully in stream
                    assert error_chunk_found or response.status_code >= 400

    @pytest.mark.asyncio
    async def test_chat_stream_response_format(self):
        """Test SSE response format compliance"""
        params = {
            "message": "Simple test message",
            "session_id": "test_session_format"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "GET",
                f"{self.BASE_URL}{self.ENDPOINT}",
                params=params
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers.get("content-type", "")

                sse_chunks = []
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        sse_chunks.append(chunk.strip())

                # Validate SSE format
                for chunk in sse_chunks:
                    if chunk.startswith("data:"):
                        data_content = chunk[5:].strip()
                        if data_content and data_content != "[DONE]":
                            # Should be valid JSON
                            try:
                                parsed = json.loads(data_content)
                                assert isinstance(parsed, dict)
                            except json.JSONDecodeError:
                                pytest.fail(f"Invalid JSON in SSE chunk: {data_content}")

    @pytest.mark.asyncio
    async def test_chat_stream_timeout_handling(self):
        """Test streaming timeout handling"""
        params = {
            "message": "This is a test for timeout handling",
            "session_id": "test_session_timeout"
        }

        # Use a very short timeout to test handling
        async with httpx.AsyncClient(timeout=1.0) as client:
            try:
                async with client.stream(
                    "GET",
                    f"{self.BASE_URL}{self.ENDPOINT}",
                    params=params
                ) as response:
                    async for chunk in response.aiter_text():
                        pass
            except httpx.TimeoutException:
                # Timeout is acceptable for this test
                pass