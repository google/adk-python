"""Minimal reproduction of ParallelAgent + include_contents='none' bug."""

from google.adk.flows.llm_flows.contents import _is_other_agent_reply
from google.adk.events.event import Event
from google.genai import types


# Simulate typical ParallelAgent scenario
events = [
    Event(author="user", content=types.Content(role="user", parts=[types.Part(text="Start")])),
    Event(author="agent_a", content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="tool_a"))]), branch="parallel.agent_a"),
    Event(author="agent_b", content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="tool_b"))]), branch="parallel.agent_b"),
    Event(author="agent_a", content=types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(name="tool_a", response={}))]), branch="parallel.agent_a"),
]

print("Session events:")
for i, e in enumerate(events):
    print(f"  [{i}] author={e.author:10s} branch={e.branch or 'None'}")

# Simulate _get_current_turn_contents backward scan
agent_name = "agent_a"
print(f"\nBackward scan for agent_name='{agent_name}':")

for i in range(len(events) - 1, -1, -1):
    event = events[i]
    if not event.content:
        continue

    is_other = _is_other_agent_reply(agent_name, event)
    is_user = event.author == 'user'

    print(f"  [{i}] author={event.author:10s} -> is_other={is_other}, is_user={is_user}")

    if is_user or is_other:
        print(f"\n  Turn boundary at [{i}]")
        print(f"  Would return events[{i}:]")
        print(f"  Lost events: [0:{i}]")

        lost = [j for j in range(i) if events[j].author == agent_name]
        if lost:
            print(f"\n  ❌ BUG: agent_a's own events {lost} are lost!")
        break
