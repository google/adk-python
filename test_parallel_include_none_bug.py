"""Test to reproduce ParallelAgent + include_contents='none' bug.

This test demonstrates that when using include_contents='none' in ParallelAgent,
agents lose their own previous events when other parallel agents' events
interleave in the session history.
"""

import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.runners import InMemoryRunner
from google.genai import types


def counter_tool(agent_name: str, step: int) -> dict:
    """Simple tool to track execution steps."""
    print(f"[TOOL CALL] {agent_name} - Step {step}")
    return {"agent": agent_name, "step": step, "message": f"{agent_name} completed step {step}"}


def create_test_agent(name: str, use_include_none: bool = False):
    """Create a test agent with multiple response steps."""

    # Agent will make 2 tool calls in sequence
    responses = [
        # First LLM call: make first tool call
        types.Part.from_function_call(
            name="counter_tool",
            args={"agent_name": name, "step": 1}
        ),
        # Second LLM call: after receiving first response, make second tool call
        types.Part.from_function_call(
            name="counter_tool",
            args={"agent_name": name, "step": 2}
        ),
        # Third LLM call: final response referencing both steps
        f"I am {name}. I completed step 1 and step 2. Both tools returned successfully."
    ]

    from tests.unittests import testing_utils
    mock_model = testing_utils.MockModel.create(responses=responses)

    return LlmAgent(
        name=name,
        model=mock_model,
        instruction=f"You are {name}. Execute your steps in sequence.",
        include_contents="none" if use_include_none else "default",
        tools=[counter_tool]
    )


def test_parallel_with_default_contents():
    """Control test: ParallelAgent with default contents should work fine."""
    print("\n" + "="*80)
    print("TEST 1: ParallelAgent with include_contents='default' (CONTROL)")
    print("="*80)

    agent_a = create_test_agent("AgentA", use_include_none=False)
    agent_b = create_test_agent("AgentB", use_include_none=False)

    parallel_agent = ParallelAgent(
        name="parallel_test",
        sub_agents=[agent_a, agent_b]
    )

    runner = InMemoryRunner(parallel_agent)
    events = list(runner.run("Start parallel execution"))

    print(f"\nTotal events: {len(events)}")

    # Check Agent A's requests
    print("\n--- Agent A's LLM Requests ---")
    for i, req in enumerate(agent_a.model.requests):
        print(f"\nRequest {i+1}:")
        print(f"  Number of contents: {len(req.contents)}")
        for j, content in enumerate(req.contents):
            role = content.role
            parts_summary = []
            for part in content.parts:
                if part.text:
                    parts_summary.append(f"text: {part.text[:50]}...")
                elif part.function_call:
                    parts_summary.append(f"function_call: {part.function_call.name}")
                elif part.function_response:
                    parts_summary.append(f"function_response: {part.function_response.name}")
            print(f"  Content {j+1} ({role}): {parts_summary}")

    print("\n✅ With default contents, Agent A can see all its previous events")
    return True


def test_parallel_with_none_contents():
    """Bug test: ParallelAgent with include_contents='none' loses context."""
    print("\n" + "="*80)
    print("TEST 2: ParallelAgent with include_contents='none' (BUG)")
    print("="*80)

    agent_a = create_test_agent("AgentA", use_include_none=True)
    agent_b = create_test_agent("AgentB", use_include_none=True)

    parallel_agent = ParallelAgent(
        name="parallel_test",
        sub_agents=[agent_a, agent_b]
    )

    runner = InMemoryRunner(parallel_agent)

    print("\nExecuting parallel agents with include_contents='none'...")
    events = list(runner.run("Start parallel execution"))

    print(f"\nTotal events: {len(events)}")
    print("\n--- Session Event Timeline ---")
    for i, event in enumerate(events):
        if event.content:
            parts_summary = []
            for part in event.content.parts:
                if part.text:
                    parts_summary.append(f"text: '{part.text[:40]}...'")
                elif part.function_call:
                    parts_summary.append(f"fn_call: {part.function_call.name}")
                elif part.function_response:
                    parts_summary.append(f"fn_resp: {part.function_response.name}")
            print(f"Event {i}: author='{event.author}' branch='{event.branch}' - {parts_summary}")

    # Check Agent A's requests - this is where the bug manifests
    print("\n--- Agent A's LLM Requests ---")
    bug_detected = False

    for i, req in enumerate(agent_a.model.requests):
        print(f"\nRequest {i+1}:")
        print(f"  Number of contents: {len(req.contents)}")

        # Count own events (function calls/responses from AgentA)
        own_function_calls = 0
        own_function_responses = 0

        for j, content in enumerate(req.contents):
            role = content.role
            parts_summary = []
            for part in content.parts:
                if part.text:
                    text_preview = part.text[:50].replace('\n', ' ')
                    parts_summary.append(f"text: {text_preview}...")
                elif part.function_call:
                    parts_summary.append(f"function_call: {part.function_call.name}")
                    own_function_calls += 1
                elif part.function_response:
                    parts_summary.append(f"function_response: {part.function_response.name}")
                    own_function_responses += 1
            print(f"  Content {j+1} ({role}): {parts_summary}")

        # BUG: In request 2 (after first function call/response),
        # Agent A should see its own previous function_call and function_response
        # But with include_contents='none' in ParallelAgent, it loses them!
        if i == 1:  # Second request (after first tool call)
            print(f"\n  📊 Analysis:")
            print(f"     - Own function_calls seen: {own_function_calls}")
            print(f"     - Own function_responses seen: {own_function_responses}")

            if own_function_calls == 0 and own_function_responses == 0:
                print(f"  ❌ BUG DETECTED: Agent A lost its previous function call/response!")
                print(f"     This happens because AgentB's events created a turn boundary,")
                print(f"     and branch filtering excluded AgentA's earlier events.")
                bug_detected = True
            elif own_function_calls >= 1 and own_function_responses >= 1:
                print(f"  ✅ Agent A can see its previous context")

    return bug_detected


def test_sequential_with_none_contents():
    """Control test: SequentialAgent with include_contents='none' should work."""
    print("\n" + "="*80)
    print("TEST 3: SequentialAgent with include_contents='none' (CONTROL)")
    print("="*80)

    from google.adk.agents.sequential_agent import SequentialAgent

    agent_a = create_test_agent("AgentA", use_include_none=True)
    agent_b = create_test_agent("AgentB", use_include_none=True)

    sequential_agent = SequentialAgent(
        name="sequential_test",
        sub_agents=[agent_a, agent_b]
    )

    runner = InMemoryRunner(sequential_agent)
    events = list(runner.run("Start sequential execution"))

    print(f"\nTotal events: {len(events)}")

    # Check Agent A's requests
    print("\n--- Agent A's LLM Requests ---")
    for i, req in enumerate(agent_a.model.requests):
        print(f"\nRequest {i+1}: {len(req.contents)} contents")

        own_function_calls = sum(
            1 for c in req.contents for p in c.parts if p.function_call
        )
        own_function_responses = sum(
            1 for c in req.contents for p in c.parts if p.function_response
        )

        if i == 1:
            if own_function_calls >= 1 and own_function_responses >= 1:
                print(f"  ✅ Agent A sees its previous function call/response")
                print(f"     (Sequential agents don't interleave, so no bug)")
            else:
                print(f"  ⚠️  Unexpected: Agent A lost context even in Sequential")

    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BUG REPRODUCTION: ParallelAgent + include_contents='none'")
    print("="*80)
    print("\nThis test demonstrates that include_contents='none' breaks in ParallelAgent")
    print("because interleaved events from parallel agents create incorrect turn boundaries,")
    print("causing agents to lose their own previous context.\n")

    # Run tests
    test_parallel_with_default_contents()
    bug_found = test_parallel_with_none_contents()
    test_sequential_with_none_contents()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if bug_found:
        print("\n❌ BUG CONFIRMED!")
        print("\nRoot cause:")
        print("1. ParallelAgent interleaves events from multiple agents")
        print("2. _get_current_turn_contents() scans backward for turn boundary")
        print("3. Finds OTHER agent's event first (e.g., AgentB)")
        print("4. Stops scanning, missing OWN earlier events (AgentA's previous calls)")
        print("5. Branch filtering then removes the other agent's events")
        print("6. Result: Agent loses its own previous function calls/responses!")

        print("\nRecommendation:")
        print("- Don't use include_contents='none' with ParallelAgent")
        print("- Or fix _get_current_turn_contents to be branch-aware")
    else:
        print("\n⚠️  Bug might not reproduce in this environment")
        print("Try running with actual parallel execution and event interleaving")
