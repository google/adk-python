"""Bug: ParallelAgent + include_contents='none' loses agent's own context."""

from google.adk.flows.llm_flows.contents import _is_other_agent_reply
from google.adk.events.event import Event
from google.genai import types

# Simulate ParallelAgent with interleaved events
events = [
    Event(author="user", content=types.Content(role="user", parts=[types.Part(text="Start")])),
    Event(author="agent_a", content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="tool"))]), branch="parallel.a"),
    Event(author="agent_b", content=types.Content(role="model", parts=[types.Part(text="B working")]), branch="parallel.b"),
    Event(author="agent_a", content=types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(name="tool", response={}))]), branch="parallel.a"),
]

# Simulate _get_current_turn_contents backward scan
agent_name = "agent_a"
for i in range(len(events) - 1, -1, -1):
    event = events[i]
    if event.author == 'user' or _is_other_agent_reply(agent_name, event):
        print(f"Turn boundary at event {i} (author={event.author})")
        print(f"Returns events[{i}:]")

        lost_own_events = [j for j in range(i) if events[j].author == agent_name]
        if lost_own_events:
            print(f"\n❌ BUG: agent_a loses its own event {lost_own_events[0]}")
            print(f"   - Event {lost_own_events[0]}: function_call")
            print(f"   - Event {i}: function_response")
            print(f"   → Agent receives response without seeing the call!")
        break
