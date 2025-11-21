"""Reproduces ParallelAgent + include_contents='none' bug where agents lose their own context."""

from google.adk.flows.llm_flows.contents import _get_current_turn_contents
from google.adk.events.event import Event
from google.genai import types


def create_event(author, branch, content_type, **kwargs):
    """Helper to create test events."""
    if content_type == "text":
        parts = [types.Part(text=kwargs["text"])]
    elif content_type == "function_call":
        parts = [types.Part(function_call=types.FunctionCall(
            name=kwargs["name"], args=kwargs.get("args", {})
        ))]
    elif content_type == "function_response":
        parts = [types.Part(function_response=types.FunctionResponse(
            name=kwargs["name"], response=kwargs.get("response", {})
        ))]

    return Event(
        author=author,
        content=types.Content(role="model" if content_type == "function_call" else "user", parts=parts),
        branch=branch
    )


# Simulate ParallelAgent event interleaving
events = [
    create_event("user", None, "text", text="Start"),
    create_event("agent_a", "parallel.agent_a", "function_call", name="tool_a", args={"x": 1}),
    create_event("agent_b", "parallel.agent_b", "function_call", name="tool_b", args={"x": 2}),
    create_event("agent_a", "parallel.agent_a", "function_response", name="tool_a", response={"result": 1}),
    create_event("agent_b", "parallel.agent_b", "function_response", name="tool_b", response={"result": 2}),
]

print("Events:")
for i, e in enumerate(events):
    print(f"  {i}: author={e.author:10s} branch={e.branch or 'None':20s}")

print(f"\nCalling _get_current_turn_contents(agent_name='agent_a', branch='parallel.agent_a'):")

result = _get_current_turn_contents(
    current_branch="parallel.agent_a",
    events=events,
    agent_name="agent_a"
)

print(f"\nResult: {len(result)} contents")
for i, content in enumerate(result):
    part_type = "text" if content.parts[0].text else \
                 "function_call" if content.parts[0].function_call else \
                 "function_response"
    print(f"  {i}: {part_type}")

# Expected: agent_a should see its own function_call (event 1) and function_response (event 3)
# Actual: agent_a loses event 1 because turn boundary stops at event 2 (agent_b)

has_function_call = any(c.parts[0].function_call for c in result)
has_function_response = any(c.parts[0].function_response for c in result)

print(f"\nBug check:")
print(f"  Has function_call: {has_function_call}")
print(f"  Has function_response: {has_function_response}")

if has_function_response and not has_function_call:
    print(f"\n❌ BUG: Agent received function_response without seeing its own function_call!")
    print(f"   Cause: Turn boundary at event 2 (agent_b) stopped backward scan at event 2")
    print(f"   Result: Event 1 (agent_a's function_call) was never scanned")
