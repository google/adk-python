#!/usr/bin/env python3
"""Executable demonstration of ParallelAgent + include_contents='none' bug."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from google.adk.flows.llm_flows.contents import _get_current_turn_contents, _is_other_agent_reply
from google.adk.events.event import Event
from google.genai import types


def demo_bug_with_real_logic():
    """Demonstrate the bug using actual ADK code logic."""

    print("="*80)
    print("DEMONSTRATION: ParallelAgent + include_contents='none' Bug")
    print("="*80)

    # Simulate a realistic ParallelAgent scenario
    print("\n📋 Scenario:")
    print("  - ParallelAgent with 2 sub-agents (agent_a, agent_b)")
    print("  - Both use include_contents='none'")
    print("  - Events interleave due to parallel execution")

    # Create simulated events as they would appear in session.events
    events = []

    # Event 0: User input
    events.append(Event(
        author="user",
        content=types.Content(role="user", parts=[types.Part(text="Start parallel work")]),
        branch=None
    ))

    # Event 1: Agent A's first function call
    events.append(Event(
        author="agent_a",
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(
                name="tool_a",
                args={"value": "test"}
            ))]
        ),
        branch="parallel_root.agent_a"
    ))

    # Event 2: Agent B's function call (INTERLEAVING!)
    events.append(Event(
        author="agent_b",
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(
                name="tool_b",
                args={"value": "test"}
            ))]
        ),
        branch="parallel_root.agent_b"
    ))

    # Event 3: Agent A's function response
    events.append(Event(
        author="agent_a",
        content=types.Content(
            role="user",
            parts=[types.Part(function_response=types.FunctionResponse(
                name="tool_a",
                response={"result": "done"}
            ))]
        ),
        branch="parallel_root.agent_a"
    ))

    # Event 4: Agent B's function response
    events.append(Event(
        author="agent_b",
        content=types.Content(
            role="user",
            parts=[types.Part(function_response=types.FunctionResponse(
                name="tool_b",
                response={"result": "done"}
            ))]
        ),
        branch="parallel_root.agent_b"
    ))

    # Event 5: Agent A's text response
    events.append(Event(
        author="agent_a",
        content=types.Content(
            role="model",
            parts=[types.Part(text="Agent A finished")]
        ),
        branch="parallel_root.agent_a"
    ))

    print("\n📊 Session Events (in time order):")
    print("-" * 80)
    for i, event in enumerate(events):
        author = event.author.ljust(15)
        branch = (event.branch or "None").ljust(30)

        content_desc = ""
        if event.content and event.content.parts:
            part = event.content.parts[0]
            if part.text:
                content_desc = f"text: '{part.text[:30]}...'"
            elif part.function_call:
                content_desc = f"function_call: {part.function_call.name}"
            elif part.function_response:
                content_desc = f"function_response: {part.function_response.name}"

        print(f"  Event {i} | author={author} | branch={branch} | {content_desc}")

    # Now simulate what _get_current_turn_contents does for agent_a
    print("\n" + "="*80)
    print("🐛 BUG DEMONSTRATION: Call _get_current_turn_contents() for agent_a")
    print("="*80)

    agent_name = "agent_a"
    current_branch = "parallel_root.agent_a"

    print(f"\nAgent: {agent_name}")
    print(f"Branch: {current_branch}")
    print("\nScanning backward to find turn boundary...")

    # Simulate the backward scan from _get_current_turn_contents
    turn_boundary_index = None

    for i in range(len(events) - 1, -1, -1):
        event = events[i]
        if not event.content:
            continue

        is_other_agent = _is_other_agent_reply(agent_name, event)
        is_user = event.author == 'user'

        status_icon = "🔄" if event.author == agent_name else ("👤" if is_user else "🚫")
        status_text = "SELF" if event.author == agent_name else ("USER" if is_user else "OTHER AGENT")

        print(f"  {status_icon} Event {i}: author='{event.author}' → {status_text}")

        if is_user or is_other_agent:
            turn_boundary_index = i
            print(f"\n  ⚠️  TURN BOUNDARY at Event {i}!")
            print(f"  ⚠️  Will return events[{i}:] (Events {i} to {len(events)-1})")
            print(f"  ❌ Events 0 to {i-1} are NEVER SCANNED!")
            break

    if turn_boundary_index is None:
        print("\n  ✓ No turn boundary, would include all events")
        return

    # Show what gets returned
    print(f"\n" + "="*80)
    print(f"📦 Result: _get_current_turn_contents returns events[{turn_boundary_index}:]")
    print("="*80)

    print(f"\nEvents included BEFORE branch filtering:")
    for i in range(turn_boundary_index, len(events)):
        event = events[i]
        if event.content:
            print(f"  Event {i}: author='{event.author}' branch='{event.branch}'")

    # Now apply branch filtering (as _get_contents does)
    print(f"\n" + "="*80)
    print(f"🔍 Branch filtering (current_branch='{current_branch}')")
    print("="*80)

    final_events = []
    for i in range(turn_boundary_index, len(events)):
        event = events[i]
        if not event.content:
            continue

        # Simulate _is_event_belongs_to_branch logic
        event_branch = event.branch or ""
        belongs = (
            not current_branch
            or not event_branch
            or current_branch == event_branch
            or current_branch.startswith(f'{event_branch}.')
        )

        status = "✅ KEPT" if belongs else "❌ FILTERED OUT"
        print(f"  Event {i}: branch='{event_branch}' → {status}")

        if belongs:
            final_events.append((i, event))

    # Show final result
    print(f"\n" + "="*80)
    print(f"📊 FINAL RESULT: What agent_a actually receives")
    print("="*80)

    print(f"\nEvents that agent_a will see:")
    for idx, event in final_events:
        content_desc = ""
        if event.content and event.content.parts:
            part = event.content.parts[0]
            if part.text:
                content_desc = f"text: '{part.text}'"
            elif part.function_call:
                content_desc = f"function_call: {part.function_call.name}"
            elif part.function_response:
                content_desc = f"function_response: {part.function_response.name}"

        print(f"  ✓ Event {idx}: {content_desc}")

    # Analyze what was lost
    print(f"\n" + "="*80)
    print(f"❌ WHAT WAS LOST")
    print("="*80)

    lost_events = []
    for i in range(0, turn_boundary_index):
        event = events[i]
        if event.author == agent_name and event.content:
            lost_events.append((i, event))

    if lost_events:
        print(f"\nagent_a's OWN events that were lost:")
        for idx, event in lost_events:
            content_desc = ""
            if event.content and event.content.parts:
                part = event.content.parts[0]
                if part.text:
                    content_desc = f"text: '{part.text}'"
                elif part.function_call:
                    content_desc = f"function_call: {part.function_call.name}"
                elif part.function_response:
                    content_desc = f"function_response: {part.function_response.name}"

            print(f"  ❌ Event {idx}: {content_desc}")

        print(f"\n⚠️  This is the BUG!")
        print(f"   Agent lost its own previous events because:")
        print(f"   1. Event {turn_boundary_index} (agent_b) created turn boundary")
        print(f"   2. Backward scan stopped there")
        print(f"   3. Earlier agent_a events were never scanned")
    else:
        print("\n✓ No agent_a events were lost")

    # Summary
    print(f"\n" + "="*80)
    print(f"💡 SUMMARY")
    print("="*80)

    print("""
This demonstrates the bug in ParallelAgent + include_contents='none':

1. ❌ Agent A loses its own Event 1 (function_call)
2. ❌ Agent A only sees Event 3 (function_response) and Event 5 (text)
3. ❌ Function call/response pairing is broken
4. ❌ LLM receives response without seeing the original call

Root Cause:
- _get_current_turn_contents() scans backward
- Stops at first "other agent" event (agent_b at Event 2)
- Never scans Events 0-1 (including agent_a's own function_call!)
- Branch filtering then removes agent_b's events
- Result: Agent loses its own context!

Fix Options:
1. Validate: Reject include_contents='none' in ParallelAgent
2. Fix logic: Make turn boundary detection branch-aware
""")


if __name__ == "__main__":
    demo_bug_with_real_logic()
