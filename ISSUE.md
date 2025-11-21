# Bug: ParallelAgent + `include_contents='none'` causes agents to lose their own context

## Describe the bug

When using `include_contents='none'` with `ParallelAgent`, agents lose their own previous events when parallel execution causes event interleaving in the session history.

The `_get_current_turn_contents()` function in `src/google/adk/flows/llm_flows/contents.py` scans backward to find a turn boundary but stops at the first "other agent" event. When events from parallel agents interleave, this causes an agent to lose its own earlier events (including function calls), breaking the function call/response pairing.

**Impact:** Agents receive function responses without seeing the original function calls, causing confusion and incorrect behavior.

## To Reproduce

**Minimal reproduction:**

```python
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
            print(f"   - Event {i+1}: function_response")
            print(f"   → Agent receives response without seeing the call!")
        break
```

**Steps to reproduce:**
1. Install ADK: `pip install google-adk`
2. Create the reproduction script above
3. Run: `python bug_repro.py`
4. Observe output showing agent_a loses event 1 (its own function_call)

**Output:**
```
Turn boundary at event 2 (author=agent_b)
Returns events[2:]

❌ BUG: agent_a loses its own event 1
   - Event 1: function_call
   - Event 3: function_response
   → Agent receives response without seeing the call!
```

## Expected behavior

When `agent_a` calls `_get_current_turn_contents()`, it should see **all of its own events** from the current turn, including:
- Event 1: agent_a's function_call
- Event 3: agent_a's function_response

This maintains the function call/response pairing and provides proper context to the LLM.

## Actual behavior

`agent_a` only sees:
- Event 3: function_response

Event 1 (function_call) is lost because:
1. Backward scan finds event 2 (agent_b) as turn boundary
2. Scan stops at event 2, never checking events 0-1
3. `_get_contents(events[2:])` is called
4. Event 2 is filtered out by branch filtering
5. Result: Only event 3 remains

## Root cause

`_get_current_turn_contents()` in `src/google/adk/flows/llm_flows/contents.py:441-447`:

```python
for i in range(len(events) - 1, -1, -1):
    event = events[i]
    if not event.content:
        continue
    if event.author == 'user' or _is_other_agent_reply(agent_name, event):
        return _get_contents(current_branch, events[i:], agent_name)  # ← Stops here
```

The logic is not branch-aware. It stops at the first "other agent" event regardless of branch, causing agents to lose their own earlier events in different branches.

## Proposed solution

Make turn boundary detection branch-aware:

```python
for i in range(len(events) - 1, -1, -1):
    event = events[i]
    if not event.content:
        continue

    # Skip events from different branches during turn boundary detection
    if not _is_event_belongs_to_branch(current_branch, event):
        continue

    # Only check turn boundary within the same branch
    if event.author == 'user' or _is_other_agent_reply(agent_name, event):
        return _get_contents(current_branch, events[i:], agent_name)
```

This ensures agents only consider events in their own branch when finding turn boundaries, preventing loss of their own context.

**Alternative solution:** Add validation in `ParallelAgent` to reject `include_contents='none'` on sub-agents until the root cause is fixed.

## Environment

- **OS:** macOS 15.0.1
- **Python version:** 3.12.7
- **ADK version:** 1.18.0

## Model Information

- **Using LiteLLM:** No
- **Model:** gemini-2.0-flash-exp

## Additional context

This bug specifically affects `ParallelAgent` because parallel execution causes events from different agents to interleave in the session timeline. `SequentialAgent` with `include_contents='none'` works correctly because events don't interleave.

The issue occurs in any ParallelAgent workflow where sub-agents make function calls. When one agent's events appear between another agent's function_call and function_response, the turn boundary detection breaks the call/response pairing.
