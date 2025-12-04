# Phase 2: Request Flow Tracing

**Duration:** Days 2-3
**Outcome:** Understand exactly how a request flows through ADK

---

## Overview

The best way to understand ADK is to trace a real request from start to finish. This phase walks you through the complete execution path.

## The Complete Request Journey

```
User Input
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              RUNNER                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Load Session from SessionService                                │  │
│  │ 2. Create InvocationContext                                        │  │
│  │ 3. Append user message as Event                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│                              ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                         AGENT.run_async()                          │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ 4. Run before_agent_callback (if any)                        │  │  │
│  │  │ 5. Execute _run_async_impl() [THE REASON-ACT LOOP]           │  │  │
│  │  │ 6. Run after_agent_callback (if any)                         │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│                              ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 7. Plugin callbacks (on_event, after_run)                          │  │
│  │ 8. Event compaction (if configured)                                │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Events yielded to caller
```

---

## Step-by-Step Breakdown

### Step 1: Runner.run_async() Entry Point

**File:** `src/google/adk/runners.py:364`

```python
async def run_async(
    self,
    *,
    user_id: str,
    session_id: str,
    new_message: Optional[types.Content] = None,
    run_config: Optional[RunConfig] = None,
) -> AsyncGenerator[Event, None]:
```

**What happens:**
1. Validates the `new_message` has a role (defaults to "user")
2. Opens a tracing span for observability
3. Loads the session from SessionService

```python
# runners.py:408
session = await self.session_service.get_session(
    app_name=self.app_name,
    user_id=user_id,
    session_id=session_id
)
```

### Step 2: Create InvocationContext

**File:** `src/google/adk/agents/invocation_context.py`

The InvocationContext bundles everything needed for one execution:

```python
@dataclass
class InvocationContext:
    invocation_id: str           # Unique ID for this run
    agent: BaseAgent             # Current executing agent
    session: Session             # The conversation state
    user_content: Content        # The user's message

    # Services
    artifact_service: BaseArtifactService
    session_service: BaseSessionService
    memory_service: BaseMemoryService

    # Configuration
    run_config: RunConfig

    # Multi-agent state
    branch: Optional[str]        # For branched conversations
    end_invocation: bool         # Signal to stop
    live_request_queue: LiveRequestQueue  # For streaming
```

### Step 3: Append User Message

Before running the agent, the user's message is saved as an Event:

```python
# runners.py:753
async def _append_new_message_to_session(
    self,
    session: Session,
    new_message: types.Content,
    invocation_context: InvocationContext,
):
    event = Event(
        invocation_id=invocation_context.invocation_id,
        author='user',
        content=new_message,
    )
    await self.session_service.append_event(session=session, event=event)
```

### Step 4: Agent.run_async()

**File:** `src/google/adk/agents/base_agent.py:271`

This is the main entry point for agent execution:

```python
@final
async def run_async(
    self,
    parent_context: InvocationContext,
) -> AsyncGenerator[Event, None]:
    with tracer.start_as_current_span(f'invoke_agent {self.name}'):
        ctx = self._create_invocation_context(parent_context)

        # Run before callback
        if event := await self._handle_before_agent_callback(ctx):
            yield event
        if ctx.end_invocation:
            return

        # THE MAIN EXECUTION
        async for event in self._run_async_impl(ctx):
            yield event

        # Run after callback
        if event := await self._handle_after_agent_callback(ctx):
            yield event
```

### Step 5: The Reason-Act Loop (LlmAgent._run_async_impl)

**File:** `src/google/adk/agents/llm_agent.py`

This is where the magic happens. The LlmAgent implements a loop:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        REASON-ACT LOOP                                   │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ REASON: Call LLM with conversation history + tools                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Response is final text? ─────────────────────▶ EXIT (yield event) │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │ No                                        │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Response is function call?                                        │  │
│  │   1. Find matching tool                                           │  │
│  │   2. Execute tool.run_async(args, context)                        │  │
│  │   3. Create FunctionResponse event                                │  │
│  │   4. Loop back to REASON                                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│                     (back to REASON)                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key code flow:**

```python
# Simplified version of the flow
class LlmAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        flow = self._get_llm_flow()  # AutoFlow or SingleFlow

        async for event in flow.run(ctx):
            yield event

            # Check if we should continue
            if event.is_final_response():
                break
```

### Step 6: LLM Flow Execution

**File:** `src/google/adk/flows/llm_flows/`

The flow handles the actual LLM interaction:

```python
# Simplified flow
class BaseLlmFlow:
    async def run(self, ctx: InvocationContext):
        while True:
            # 1. Build the LLM request
            request = await self._build_request(ctx)

            # 2. Run before_model callbacks
            if override := await self._run_before_model_callbacks(ctx, request):
                yield self._create_event(override)
                continue

            # 3. Call the LLM
            async for response in self.llm.generate_content_async(request):
                event = self._create_event(response)
                yield event

                # 4. Handle function calls
                if function_calls := event.get_function_calls():
                    for fc in function_calls:
                        result = await self._execute_tool(fc, ctx)
                        yield self._create_function_response_event(fc, result)

                # 5. Check if done
                if event.is_final_response():
                    return
```

### Step 7: Tool Execution

When the LLM requests a function call:

```python
# flows/llm_flows/functions.py
async def execute_function_call(
    function_call: types.FunctionCall,
    tool_context: ToolContext,
    tools: list[BaseTool],
) -> Any:
    # Find matching tool
    tool = find_matching_tool(function_call.name, tools)

    # Execute the tool
    result = await tool.run_async(
        args=function_call.args,
        tool_context=tool_context,
    )

    return result
```

### Step 8: Event Persistence

Every non-partial event is saved to the session:

```python
# runners.py:736
if event.partial is not True:
    await self.session_service.append_event(
        session=session,
        event=event
    )
```

---

## Tracing Exercise: Debug a Real Request

Create this file and run it to see the complete flow:

```python
# File: exercises/trace_request.py
import asyncio
import logging
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

# Enable debug logging to see the flow
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('google_adk').setLevel(logging.DEBUG)

def get_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

agent = LlmAgent(
    name="tracer",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. Use tools when needed.",
    tools=[
        FunctionTool(func=get_time),
        FunctionTool(func=add_numbers),
    ],
)

runner = Runner(
    app_name="trace_demo",
    agent=agent,
    session_service=InMemorySessionService(),
)

async def main():
    session = await runner.session_service.create_session(
        app_name="trace_demo",
        user_id="tracer_user",
    )

    print("\n" + "="*60)
    print("TRACING REQUEST: 'What is 5 + 3?'")
    print("="*60 + "\n")

    event_count = 0
    async for event in runner.run_async(
        user_id="tracer_user",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What is 5 + 3?")]
        ),
    ):
        event_count += 1
        print(f"\n--- Event {event_count} ---")
        print(f"ID: {event.id[:8]}...")
        print(f"Author: {event.author}")
        print(f"Invocation: {event.invocation_id[:8]}...")

        if fc := event.get_function_calls():
            print(f"Function Calls: {[(f.name, f.args) for f in fc]}")

        if fr := event.get_function_responses():
            print(f"Function Responses: {[(f.name, f.response) for f in fr]}")

        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Text: {part.text}")

        print(f"Is Final: {event.is_final_response()}")

    print(f"\n{'='*60}")
    print(f"Total events: {event_count}")
    print(f"Session now has {len(session.events)} events")
    print("="*60)

asyncio.run(main())
```

**Expected Output:**

```
--- Event 1 ---
ID: abc12345...
Author: tracer
Function Calls: [('add_numbers', {'a': 5, 'b': 3})]
Is Final: False

--- Event 2 ---
ID: def67890...
Author: tracer
Function Responses: [('add_numbers', 8)]
Is Final: False

--- Event 3 ---
ID: ghi11111...
Author: tracer
Text: 5 + 3 equals 8.
Is Final: True

Total events: 3
Session now has 4 events  (includes user message)
```

---

## Key Code Locations Reference

| Step | File | Line/Method |
|------|------|-------------|
| Entry point | `runners.py` | `run_async()` ~L364 |
| Session loading | `runners.py` | `get_session()` ~L408 |
| Context creation | `runners.py` | `_setup_context_for_new_invocation()` |
| Agent execution | `base_agent.py` | `run_async()` ~L271 |
| LLM agent impl | `llm_agent.py` | `_run_async_impl()` |
| Flow execution | `flows/llm_flows/auto_flow.py` | `run()` |
| Tool execution | `flows/llm_flows/functions.py` | `execute_function_call()` |
| Event persistence | `runners.py` | `append_event()` ~L736 |

---

## Understanding the Flow Hierarchy

```
Runner.run_async()
    │
    ├── Plugin: before_run_callback
    │
    ├── Agent.run_async()
    │       │
    │       ├── before_agent_callback
    │       │
    │       ├── _run_async_impl()
    │       │       │
    │       │       └── LlmFlow.run()
    │       │               │
    │       │               ├── before_model_callback
    │       │               ├── LLM.generate_content_async()
    │       │               ├── after_model_callback
    │       │               │
    │       │               └── [If function call]
    │       │                       ├── before_tool_callback
    │       │                       ├── Tool.run_async()
    │       │                       └── after_tool_callback
    │       │
    │       └── after_agent_callback
    │
    ├── Plugin: on_event_callback (for each event)
    │
    └── Plugin: after_run_callback
```

---

## Multi-Agent Flow

When an agent delegates to a sub-agent:

```python
# Agent calls transfer_to_agent tool
async for event in runner.run_async(...):
    # Event from root agent with function call to transfer_to_agent
    # ...
    # Events from sub-agent
    # ...
    # Final event (could be from sub-agent or root)
```

The `branch` field in events tracks the delegation path:
- Root agent: `branch = None` or `branch = ""`
- Sub-agent "researcher": `branch = "researcher"`
- Nested: `branch = "researcher.analyst"`

---

## Practice Exercises

### Exercise 1: Add Logging Points

Add print statements at these locations and trace a request:

1. `runners.py:run_async()` - Entry
2. `base_agent.py:run_async()` - Agent start
3. `llm_agent.py:_run_async_impl()` - Each loop iteration
4. Tool's `run_async()` - Tool execution

### Exercise 2: Event Inspector

Build a complete event inspector:

```python
def deep_inspect_event(event: Event):
    """Print every field of an event."""
    print(f"Event Analysis:")
    print(f"  ID: {event.id}")
    print(f"  Author: {event.author}")
    print(f"  Invocation: {event.invocation_id}")
    print(f"  Branch: {event.branch}")
    print(f"  Timestamp: {event.timestamp}")
    print(f"  Partial: {event.partial}")
    print(f"  Is Final: {event.is_final_response()}")

    if event.actions:
        print(f"  Actions:")
        if event.actions.state_delta:
            print(f"    State Delta: {event.actions.state_delta}")
        if event.actions.artifact_delta:
            print(f"    Artifact Delta: {event.actions.artifact_delta}")

    # ... add more fields
```

### Exercise 3: Time the Flow

Measure how long each phase takes:

```python
import time

async def timed_run():
    start = time.time()

    events = []
    async for event in runner.run_async(...):
        events.append({
            'event': event,
            'elapsed': time.time() - start
        })

    for e in events:
        print(f"{e['elapsed']:.3f}s - {e['event'].author}")
```

---

## Key Takeaways

1. **Entry is Runner.run_async()** - Everything starts here
2. **Session is loaded first** - Context is established before execution
3. **InvocationContext bundles everything** - One object carries all state
4. **Agent.run_async() is the wrapper** - Handles callbacks around implementation
5. **_run_async_impl() is the core** - This is where agent-specific logic lives
6. **Events stream out** - AsyncGenerator pattern enables real-time responses
7. **Every event is persisted** - Full history for debugging and context

---

**Next:** [Phase 3: Mini-Projects](../phase-3-mini-projects/README.md)
