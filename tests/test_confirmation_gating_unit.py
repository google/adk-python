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
Unit tests for tool confirmation gating functionality.

Tests the fix for Issue #3018: When a tool requires confirmation,
other tools should be hidden from the model to prevent bypassing confirmation.
"""

import pytest
from google.adk.agents.invocation_context import InvocationContext


def test_invocation_context_pending_confirmation():
    """Test InvocationContext pending confirmation state management."""

    # Create a mock invocation context with minimal required fields
    from google.adk.sessions.session import Session
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    session_service = InMemorySessionService()
    session = Session(
        id="test-session",
        app_name="test-app",
        user_id="test-user"
    )

    # Mock agent
    class MockAgent(BaseAgent):
        pass

    agent = MockAgent(name="test_agent")

    inv_ctx = InvocationContext(
        invocation_id="test-invocation",
        session_service=session_service,
        session=session,
        agent=agent
    )

    # Test initial state
    assert inv_ctx.has_pending_confirmation is False
    assert inv_ctx.pending_confirmation_tool is None

    # Test setting pending confirmation
    inv_ctx.set_pending_confirmation("my_tool")
    assert inv_ctx.has_pending_confirmation is True
    assert inv_ctx.pending_confirmation_tool == "my_tool"

    # Test clearing pending confirmation
    inv_ctx.clear_pending_confirmation()
    assert inv_ctx.has_pending_confirmation is False
    assert inv_ctx.pending_confirmation_tool is None

    print("✅ InvocationContext confirmation state management works correctly")


@pytest.mark.asyncio
async def test_canonical_tools_filters_when_confirmation_pending():
    """Test that canonical_tools() filters tools when confirmation is pending."""

    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.tools.function_tool import FunctionTool
    from google.adk.sessions.session import Session
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.agents.readonly_context import ReadonlyContext

    # Define test tools
    def tool_a(x: int) -> str:
        """Tool A that requires confirmation."""
        return f"A: {x}"

    def tool_b(y: int) -> str:
        """Tool B that should be gated."""
        return f"B: {y}"

    def tool_c(z: int) -> str:
        """Tool C that should also be gated."""
        return f"C: {z}"

    # Create agent with multiple tools
    agent = LlmAgent(
        model='gemini-2.5-flash',
        name='test_agent',
        tools=[
            FunctionTool(tool_a),
            FunctionTool(tool_b),
            FunctionTool(tool_c)
        ]
    )

    # Create invocation context
    session_service = InMemorySessionService()
    session = Session(
        id="test-session",
        app_name="test-app",
        user_id="test-user"
    )

    inv_ctx = InvocationContext(
        invocation_id="test-invocation",
        session_service=session_service,
        session=session,
        agent=agent
    )

    readonly_ctx = ReadonlyContext(invocation_context=inv_ctx)

    # Test 1: All tools available when no confirmation pending
    all_tools = await agent.canonical_tools(readonly_ctx)
    assert len(all_tools) == 3, f"Expected 3 tools, got {len(all_tools)}"
    tool_names = {t.name for t in all_tools}
    assert tool_names == {"tool_a", "tool_b", "tool_c"}
    print("✅ All tools available when no confirmation pending")

    # Test 2: Only pending tool available when confirmation pending
    inv_ctx.set_pending_confirmation("tool_a")
    filtered_tools = await agent.canonical_tools(readonly_ctx)
    assert len(filtered_tools) == 1, f"Expected 1 tool, got {len(filtered_tools)}"
    assert filtered_tools[0].name == "tool_a"
    print("✅ Only tool_a available when tool_a confirmation pending")

    # Test 3: All tools available again after clearing
    inv_ctx.clear_pending_confirmation()
    all_tools_again = await agent.canonical_tools(readonly_ctx)
    assert len(all_tools_again) == 3
    print("✅ All tools available again after clearing confirmation")


if __name__ == "__main__":
    import asyncio

    print("Running Confirmation Gating Unit Tests")
    print("=" * 50)

    test_invocation_context_pending_confirmation()
    print()

    asyncio.run(test_canonical_tools_filters_when_confirmation_pending())
    print()

    print("All unit tests passed! ✅")
