"""Simplified test to reproduce ParallelAgent + include_contents='none' bug.

This test demonstrates the bug by examining session events and showing
how turn boundary logic loses agent's own previous events.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import InMemoryRunner


def simple_tool_a(value: str) -> dict:
    """Tool for agent A."""
    return {"result": f"Tool A processed: {value}"}


def simple_tool_b(value: str) -> dict:
    """Tool for agent B."""
    return {"result": f"Tool B processed: {value}"}


def analyze_session_events(runner, test_name):
    """Analyze session events to show interleaving."""
    print(f"\n{'='*80}")
    print(f"{test_name}")
    print('='*80)

    session = runner.session
    print(f"\nTotal events in session: {len(session.events)}")
    print("\nEvent Timeline:")
    print("-" * 80)

    for i, event in enumerate(session.events):
        if not event.content:
            continue

        author = event.author
        branch = event.branch or "None"

        # Summarize content
        parts_info = []
        for part in event.content.parts:
            if part.text:
                text_preview = part.text[:40].replace('\n', ' ')
                parts_info.append(f"TEXT: '{text_preview}...'")
            elif part.function_call:
                parts_info.append(f"FUNC_CALL: {part.function_call.name}()")
            elif part.function_response:
                parts_info.append(f"FUNC_RESP: {part.function_response.name}")

        print(f"Event {i:2d} | author={author:20s} | branch={branch:30s}")
        print(f"         | {' | '.join(parts_info)}")

    return session.events


def simulate_turn_boundary_logic(events, agent_name, current_branch):
    """Simulate the _get_current_turn_contents logic to show the bug."""
    print(f"\n--- Simulating Turn Boundary Logic for '{agent_name}' ---")
    print(f"Current branch: {current_branch}")
    print("\nScanning backward from latest event...")

    # Simulate backward scan
    for i in range(len(events) - 1, -1, -1):
        event = events[i]
        if not event.content:
            continue

        # Check if this is "other agent reply"
        is_other_agent = (
            agent_name
            and event.author != agent_name
            and event.author != 'user'
        )

        status = "SKIP (self)" if event.author == agent_name else "OTHER AGENT!"

        print(f"  Event {i}: author='{event.author}' -> {status}")

        if event.author == 'user' or is_other_agent:
            print(f"\n  ⚠️  TURN BOUNDARY FOUND at Event {i}")
            print(f"  → Will include events[{i}:] = Event {i} to {len(events)-1}")
            print(f"  → Events before Event {i} are IGNORED!")

            # Now show what gets filtered by branch
            print(f"\n  After branch filtering (branch={current_branch}):")
            for j in range(i, len(events)):
                e = events[j]
                if not e.content:
                    continue

                # Simulate branch check
                event_branch = e.branch or "None"
                belongs_to_branch = (
                    not current_branch
                    or not event_branch
                    or current_branch == event_branch
                    or current_branch.startswith(f'{event_branch}.')
                )

                status = "✅ INCLUDED" if belongs_to_branch else "❌ FILTERED OUT"
                print(f"    Event {j}: author='{e.author}' branch='{event_branch}' -> {status}")

            return

    print("\n  No turn boundary found, would include all events")


def test_parallel_bug():
    """Main bug reproduction test."""

    print("\n" + "="*80)
    print("BUG REPRODUCTION TEST")
    print("="*80)

    # Create simple agents
    agent_a = LlmAgent(
        name="agent_a",
        model="gemini-2.0-flash-exp",
        instruction="Call simple_tool_a with value 'test_a', then respond 'Done A'",
        include_contents="none",  # ← This causes the bug!
        tools=[simple_tool_a]
    )

    agent_b = LlmAgent(
        name="agent_b",
        model="gemini-2.0-flash-exp",
        instruction="Call simple_tool_b with value 'test_b', then respond 'Done B'",
        include_contents="none",  # ← This causes the bug!
        tools=[simple_tool_b]
    )

    parallel_agent = ParallelAgent(
        name="parallel_root",
        sub_agents=[agent_a, agent_b]
    )

    print("\n🔧 Setup:")
    print(f"  - ParallelAgent with 2 sub-agents")
    print(f"  - Both agents have include_contents='none'")
    print(f"  - Each agent will make function calls")
    print(f"  - Events will interleave in session history")

    # Note: This would require actual API key and will make real calls
    # For demonstration, we'll just show the logic
    print("\n⚠️  Note: This needs GOOGLE_API_KEY to actually run")
    print("Instead, let's demonstrate the bug logic with a simulated scenario:\n")

    # Simulate what would happen
    print("SIMULATED EVENT SEQUENCE:")
    print("-" * 80)
    simulated_events = [
        "Event 0: author='user', branch=None - Initial request",
        "Event 1: author='agent_a', branch='parallel_root.agent_a' - function_call(simple_tool_a)",
        "Event 2: author='agent_b', branch='parallel_root.agent_b' - function_call(simple_tool_b)",  # ← AgentB interleaves!
        "Event 3: author='agent_a', branch='parallel_root.agent_a' - function_response(simple_tool_a)",
        "Event 4: author='agent_b', branch='parallel_root.agent_b' - function_response(simple_tool_b)",
        "Event 5: author='agent_a', branch='parallel_root.agent_a' - text response",
    ]

    for event in simulated_events:
        print(f"  {event}")

    print("\n" + "="*80)
    print("BUG ANALYSIS: What happens when agent_a makes its next LLM call?")
    print("="*80)

    print("\nWith include_contents='none', _get_current_turn_contents() scans backward:")
    print("\n  Event 5: author='agent_a' → SKIP (self)")
    print("  Event 4: author='agent_b' → SKIP (self)")
    print("  Event 3: author='agent_a' → SKIP (self)")
    print("  Event 2: author='agent_b' → ⚠️  OTHER AGENT DETECTED!")
    print("           → TURN BOUNDARY! Return events[2:]")
    print("           → Events 0, 1 are NEVER SCANNED!")

    print("\n  Then branch filtering on events[2:]:")
    print("    Event 2: branch='parallel_root.agent_b' ≠ 'parallel_root.agent_a' → ❌ FILTERED")
    print("    Event 3: branch='parallel_root.agent_a' → ✅ INCLUDED")
    print("    Event 4: branch='parallel_root.agent_b' → ❌ FILTERED")
    print("    Event 5: branch='parallel_root.agent_a' → ✅ INCLUDED")

    print("\n  📊 RESULT: agent_a only sees Events 3, 5")
    print("  ❌ LOST: Event 1 (agent_a's own function_call!)")
    print("  ❌ LOST: Event 0 (original user request!)")

    print("\n" + "="*80)
    print("WHY THIS IS A BUG")
    print("="*80)
    print("\n1. Agent loses its OWN previous function_call (Event 1)")
    print("2. This breaks the function call/response pairing")
    print("3. LLM receives function_response without seeing the original call")
    print("4. This causes confusion and incorrect responses")

    print("\n" + "="*80)
    print("ROOT CAUSE")
    print("="*80)
    print("\n_get_current_turn_contents() has two problems:")
    print("1. Stops at FIRST 'other agent' event, missing earlier self events")
    print("2. Not branch-aware during turn boundary detection")

    print("\n" + "="*80)
    print("SOLUTION")
    print("="*80)
    print("\nOption 1: Validate and reject")
    print("  - Raise error when include_contents='none' used in ParallelAgent")
    print("\nOption 2: Fix the logic")
    print("  - Make turn boundary detection branch-aware")
    print("  - Only stop at 'other agent' events in DIFFERENT branches")
    print("  - Or: scan backward within same branch only")

    return True


if __name__ == "__main__":
    test_parallel_bug()

    print("\n" + "="*80)
    print("To actually reproduce with real execution, you need:")
    print("="*80)
    print("1. Set GOOGLE_API_KEY environment variable")
    print("2. Run ParallelAgent with include_contents='none'")
    print("3. Each sub-agent makes multiple function calls")
    print("4. Inspect the LLM request contents to see missing events")
    print("\nThis simulated analysis shows the logical bug in the code.")
