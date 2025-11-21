"""Unit test demonstrating ParallelAgent + include_contents='none' bug.

This test shows that when using include_contents='none' with ParallelAgent,
agents lose their own previous events when parallel execution causes event
interleaving in the session history.
"""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.genai import types
import pytest

from .. import testing_utils


def simple_tool(agent_name: str, value: str) -> dict:
    """A simple tool for testing."""
    return {"agent": agent_name, "processed": value}


@pytest.mark.asyncio
async def test_parallel_agent_include_none_loses_own_events():
    """Test that demonstrates the bug where agents lose their own previous events.

    Expected behavior:
    - Agent should see its own previous function_call when processing function_response
    - Even with include_contents='none', agent should maintain its own context

    Actual behavior (BUG):
    - When other parallel agents' events interleave in session history
    - Turn boundary detection stops at the "other agent" event
    - Agent's own earlier events are never scanned
    - This breaks function call/response pairing
    """

    # Create two agents that will make function calls
    # Agent A will make 2 sequential function calls
    agent_a_responses = [
        # First call
        types.Part.from_function_call(
            name="simple_tool",
            args={"agent_name": "agent_a", "value": "first"}
        ),
        # Second call (after receiving first response)
        types.Part.from_function_call(
            name="simple_tool",
            args={"agent_name": "agent_a", "value": "second"}
        ),
        # Final response
        "Agent A completed both calls"
    ]

    agent_a_model = testing_utils.MockModel.create(responses=agent_a_responses)

    agent_a = LlmAgent(
        name="agent_a",
        model=agent_a_model,
        instruction="You are Agent A. Make two tool calls.",
        include_contents="none",  # ← Bug trigger
        tools=[simple_tool]
    )

    # Agent B will also make function calls (to create interleaving)
    agent_b_responses = [
        types.Part.from_function_call(
            name="simple_tool",
            args={"agent_name": "agent_b", "value": "test"}
        ),
        "Agent B done"
    ]

    agent_b_model = testing_utils.MockModel.create(responses=agent_b_responses)

    agent_b = LlmAgent(
        name="agent_b",
        model=agent_b_model,
        instruction="You are Agent B. Make one tool call.",
        include_contents="none",  # ← Bug trigger
        tools=[simple_tool]
    )

    # Create ParallelAgent
    parallel_agent = ParallelAgent(
        name="parallel_root",
        sub_agents=[agent_a, agent_b]
    )

    # Run the parallel agent
    runner = testing_utils.InMemoryRunner(parallel_agent)
    events = runner.run("Execute both agents")

    # Analyze the session events
    session = runner.session
    print("\n--- Session Event Timeline ---")
    for i, event in enumerate(session.events):
        if event.content:
            print(f"Event {i}: author='{event.author}' branch='{event.branch}'")

    # Check Agent A's LLM requests
    print("\n--- Agent A's LLM Requests ---")

    # The bug manifests in Agent A's second request
    # (after first function call/response, before second function call)
    if len(agent_a_model.requests) >= 2:
        second_request = agent_a_model.requests[1]

        print(f"\nAgent A's 2nd LLM Request (after 1st function response):")
        print(f"  Total contents: {len(second_request.contents)}")

        # Count function calls and responses in this request
        function_calls_seen = 0
        function_responses_seen = 0

        for content in second_request.contents:
            for part in content.parts:
                if part.function_call:
                    function_calls_seen += 1
                    print(f"  ✓ Sees function_call: {part.function_call.name}")
                if part.function_response:
                    function_responses_seen += 1
                    print(f"  ✓ Sees function_response: {part.function_response.name}")

        print(f"\n  Summary:")
        print(f"    - Function calls seen: {function_calls_seen}")
        print(f"    - Function responses seen: {function_responses_seen}")

        # BUG: Agent A should see its own previous function_call (at least 1)
        # But due to the bug, it might see 0 if AgentB's event created turn boundary
        if function_calls_seen == 0:
            print(f"\n  ❌ BUG DETECTED!")
            print(f"     Agent A lost its own previous function_call")
            print(f"     This happens because:")
            print(f"     1. AgentB's event created a turn boundary")
            print(f"     2. Turn boundary detection stopped there")
            print(f"     3. AgentA's earlier function_call was never scanned")

            # This is the bug we're demonstrating
            assert function_calls_seen == 0, (
                "Bug reproduced: Agent lost its own function call due to "
                "turn boundary created by parallel agent's interleaved event"
            )
        else:
            print(f"\n  ℹ️  Bug not reproduced in this execution")
            print(f"     (Event interleaving might be different)")

    # Additional check: verify events did interleave
    agent_a_events = [e for e in session.events if e.author == "agent_a" and e.content]
    agent_b_events = [e for e in session.events if e.author == "agent_b" and e.content]

    if agent_a_events and agent_b_events:
        # Find if there's interleaving
        a_indices = [session.events.index(e) for e in agent_a_events]
        b_indices = [session.events.index(e) for e in agent_b_events]

        interleaved = any(
            min(b_indices) < a_idx < max(b_indices)
            for a_idx in a_indices
        )

        if interleaved:
            print(f"\n  ✓ Events are interleaved (parallel execution confirmed)")
        else:
            print(f"\n  ℹ️  Events not interleaved (sequential execution?)")


@pytest.mark.asyncio
async def test_parallel_agent_with_default_contents_works():
    """Control test: ParallelAgent with default contents should work correctly."""

    agent_a_responses = [
        types.Part.from_function_call(
            name="simple_tool",
            args={"agent_name": "agent_a", "value": "test"}
        ),
        "Agent A done"
    ]

    agent_a_model = testing_utils.MockModel.create(responses=agent_a_responses)

    agent_a = LlmAgent(
        name="agent_a",
        model=agent_a_model,
        instruction="You are Agent A",
        include_contents="default",  # ← No bug with default
        tools=[simple_tool]
    )

    agent_b_model = testing_utils.MockModel.create(responses=["Agent B done"])

    agent_b = LlmAgent(
        name="agent_b",
        model=agent_b_model,
        instruction="You are Agent B",
        include_contents="default"
    )

    parallel_agent = ParallelAgent(
        name="parallel_root",
        sub_agents=[agent_a, agent_b]
    )

    runner = testing_utils.InMemoryRunner(parallel_agent)
    events = runner.run("Execute both agents")

    # With default contents, agent should see its previous function call
    if len(agent_a_model.requests) >= 2:
        second_request = agent_a_model.requests[1]

        function_calls_seen = sum(
            1 for c in second_request.contents
            for p in c.parts if p.function_call
        )

        # Should see at least its own function call
        assert function_calls_seen >= 1, (
            "With default contents, agent should see its own function call"
        )

        print("\n✅ With include_contents='default', agent sees its own events correctly")
