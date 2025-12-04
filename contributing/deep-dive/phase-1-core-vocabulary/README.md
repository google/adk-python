# Phase 1: Core Vocabulary

**Duration:** Days 1-2
**Outcome:** Understand the 5 foundational abstractions that power ADK

---

## Overview

ADK is built on 5 core abstractions. Understanding these is essential - everything else in the framework is an implementation or extension of these concepts.

```
┌─────────────────────────────────────────────────────────────────┐
│                          RUNNER                                  │
│   (Orchestrates the execution, manages the Reason-Act loop)     │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │  AGENT  │───▶│  EVENT  │───▶│ SESSION │◀───│  TOOL   │      │
│  │(Blueprint)   │ (Data)  │    │ (State) │    │(Action) │      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## The 5 Core Abstractions

| Abstraction | What It Is | Key File |
|-------------|------------|----------|
| **Agent** | Blueprint defining identity, instructions, tools | `agents/base_agent.py` |
| **Event** | Unit of data flowing through the system | `events/event.py` |
| **Session** | Conversation state container | `sessions/session.py` |
| **Tool** | Capability an agent can invoke | `tools/base_tool.py` |
| **Runner** | Orchestration engine | `runners.py` |

---

## 1. Agent: The Blueprint

**File:** `src/google/adk/agents/base_agent.py`

An Agent is a **declarative configuration object** - it describes WHAT an agent is, not HOW it runs. Think of it as a blueprint or specification.

### Key Properties

```python
class BaseAgent(BaseModel):
    name: str                    # Unique identifier (must be valid Python identifier)
    description: str = ''        # Model uses this for delegation decisions
    parent_agent: Optional[BaseAgent]  # For hierarchical agent trees
    sub_agents: list[BaseAgent]  # Child agents this agent can delegate to

    # Lifecycle hooks
    before_agent_callback: Optional[BeforeAgentCallback] = None
    after_agent_callback: Optional[AfterAgentCallback] = None
```

### Agent Hierarchy

Agents form a **tree structure**. The root agent handles initial requests, and can delegate to sub-agents:

```python
from google.adk.agents import LlmAgent

# Create a simple agent tree
root_agent = LlmAgent(
    name="coordinator",
    description="Routes requests to specialized agents",
    model="gemini-2.0-flash",
    sub_agents=[
        LlmAgent(
            name="researcher",
            description="Handles research and information gathering",
            model="gemini-2.0-flash",
        ),
        LlmAgent(
            name="coder",
            description="Writes and reviews code",
            model="gemini-2.0-flash",
        ),
    ],
)

# Navigate the tree
print(root_agent.root_agent.name)  # "coordinator"
print(root_agent.find_agent("researcher"))  # Returns researcher agent
```

### Core Methods

| Method | Purpose |
|--------|---------|
| `run_async()` | Entry point for text-based conversations |
| `run_live()` | Entry point for video/audio conversations |
| `_run_async_impl()` | Override this to implement custom agent logic |
| `find_agent(name)` | Find agent by name in the tree |
| `clone()` | Create a copy of the agent |

### Why This Matters

The agent abstraction is intentionally **minimal**. It defines:
- Identity (name, description)
- Structure (parent, sub-agents)
- Hooks (callbacks)

The actual "intelligence" comes from **LlmAgent** which extends BaseAgent with:
- LLM model selection
- System instructions
- Tool bindings
- Output schemas

**Read Next:** `src/google/adk/agents/llm_agent.py` to see how a real agent is implemented.

---

## 2. Event: The Data Flow

**File:** `src/google/adk/events/event.py`

An Event represents **a single unit of activity** in the agent execution. Every action generates an event - user input, LLM response, tool call, tool response.

### Key Properties

```python
class Event(LlmResponse):
    id: str                      # Unique identifier (UUID)
    invocation_id: str           # Groups related events in one run
    author: str                  # 'user' or agent name
    timestamp: float             # When the event occurred

    # Content
    content: Optional[types.Content]  # The actual payload

    # Execution metadata
    actions: EventActions        # State changes, tool calls, etc.
    branch: Optional[str]        # For multi-agent branching

    # For long-running operations
    long_running_tool_ids: Optional[set[str]]
```

### Event Types (By Content)

| Event Type | What It Contains |
|------------|------------------|
| User Input | `content` with user's message |
| LLM Response | `content` with model's text response |
| Function Call | `content.parts` containing `FunctionCall` objects |
| Function Response | `content.parts` containing `FunctionResponse` objects |

### Detecting Event Types

```python
def analyze_event(event: Event):
    # Check if it's the final response
    if event.is_final_response():
        print("This is a final response from the agent")

    # Check for function calls
    function_calls = event.get_function_calls()
    if function_calls:
        for fc in function_calls:
            print(f"Agent wants to call: {fc.name} with args: {fc.args}")

    # Check for function responses
    function_responses = event.get_function_responses()
    if function_responses:
        for fr in function_responses:
            print(f"Tool {fr.name} returned: {fr.response}")
```

### Event Flow Example

```
User: "What's the weather in NYC?"
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Event 1: User Input                             │
│   author: "user"                                │
│   content: "What's the weather in NYC?"         │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Event 2: Function Call                          │
│   author: "weather_agent"                       │
│   content.parts: [FunctionCall(                 │
│     name="get_weather",                         │
│     args={"city": "NYC"}                        │
│   )]                                            │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Event 3: Function Response                      │
│   author: "weather_agent"                       │
│   content.parts: [FunctionResponse(             │
│     name="get_weather",                         │
│     response={"temp": "72F", "condition": "sunny"}│
│   )]                                            │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Event 4: Final Response                         │
│   author: "weather_agent"                       │
│   content: "The weather in NYC is 72F and sunny"│
│   is_final_response(): True                     │
└─────────────────────────────────────────────────┘
```

### Why This Matters

Events are the **universal data model** for ADK:
- Every interaction is traceable
- Sessions are just lists of events
- Debugging = inspecting event sequences
- Streaming = yielding events as they're generated

---

## 3. Session: The State Container

**File:** `src/google/adk/sessions/session.py`

A Session represents **a single conversation** - all the events between a user and agents.

### Key Properties

```python
class Session(BaseModel):
    id: str                      # Unique session identifier
    app_name: str                # Which application this session belongs to
    user_id: str                 # Which user owns this session
    state: dict[str, Any]        # Arbitrary key-value state
    events: list[Event]          # The conversation history
    last_update_time: float      # For cache invalidation
```

### Session Lifecycle

```python
from google.adk.sessions import InMemorySessionService

# Create a session service
session_service = InMemorySessionService()

# Create a new session
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
)

print(f"Session ID: {session.id}")
print(f"Events: {len(session.events)}")  # 0 initially

# After running the agent, events are added
# session.events now contains the conversation history
```

### State Management

Session state allows you to persist arbitrary data across turns:

```python
# In a tool or callback, you can access and modify state
async def my_tool(args: dict, tool_context: ToolContext):
    # Read state
    counter = tool_context.state.get("counter", 0)

    # Write state
    tool_context.state["counter"] = counter + 1

    return f"Counter is now {counter + 1}"
```

### Session Storage Options

| Implementation | Use Case |
|----------------|----------|
| `InMemorySessionService` | Testing, development |
| `SQLiteSessionService` | Local persistence |
| `DatabaseSessionService` | Production (PostgreSQL, etc.) |
| `VertexAiSessionService` | Managed cloud sessions |

### Why This Matters

Sessions solve the **stateful conversation problem**:
- LLMs are stateless - sessions give them memory
- Events provide the context window
- State dict allows custom data persistence
- Multiple backends for different deployment needs

---

## 4. Tool: The Capability

**File:** `src/google/adk/tools/base_tool.py`

A Tool is **an action an agent can perform** - calling an API, querying a database, performing calculations.

### Key Properties

```python
class BaseTool(ABC):
    name: str                    # How the LLM refers to this tool
    description: str             # What the tool does (for LLM)
    is_long_running: bool        # For async operations
    custom_metadata: Optional[dict]  # Extra tool-specific data
```

### Tool Types

| Tool Type | Purpose | File |
|-----------|---------|------|
| `FunctionTool` | Wrap any Python function | `function_tool.py` |
| `AgentTool` | Delegate to another agent | `agent_tool.py` |
| `LongRunningFunctionTool` | Async operations | `long_running_tool.py` |
| `GoogleSearchTool` | Web search | `google_search_tool.py` |
| `OpenAPITool` | REST API calls | `openapi_tool/` |

### Creating Tools

**Method 1: Function Decorator (Simplest)**

```python
from google.adk.tools import FunctionTool

def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A dictionary with temperature and conditions.
    """
    # Your implementation here
    return {"temp": "72F", "condition": "sunny"}

# Automatically creates a tool from the function
weather_tool = FunctionTool(func=get_weather)
```

**Method 2: Custom Tool Class**

```python
from google.adk.tools import BaseTool
from google.adk.tools import ToolContext
from google.genai import types

class DatabaseTool(BaseTool):
    def __init__(self, connection_string: str):
        super().__init__(
            name="query_database",
            description="Execute SQL queries against the database"
        )
        self.conn_string = connection_string

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The SQL query to execute"
                    )
                },
                required=["query"]
            )
        )

    async def run_async(self, *, args: dict, tool_context: ToolContext) -> Any:
        query = args["query"]
        # Execute query and return results
        return {"rows": [...], "count": 10}
```

### Tool Execution Flow

```
LLM decides to call tool
         │
         ▼
┌─────────────────────────────────────────┐
│ Runner receives FunctionCall event      │
│   - name: "get_weather"                 │
│   - args: {"city": "NYC"}               │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Runner finds matching tool              │
│ Calls tool.run_async(args, context)     │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Runner creates FunctionResponse event   │
│ Sends response back to LLM              │
└─────────────────────────────────────────┘
```

### Why This Matters

Tools are how agents **interact with the world**:
- Without tools, agents can only generate text
- Tools make agents useful (search, compute, integrate)
- The function calling protocol is standardized
- Tools can be composed and shared

---

## 5. Runner: The Orchestrator

**File:** `src/google/adk/runners.py`

The Runner is the **execution engine** - it orchestrates the entire agent lifecycle.

### Key Properties

```python
class Runner:
    app_name: str                          # Application identifier
    agent: BaseAgent                       # The root agent
    session_service: BaseSessionService    # Where sessions are stored
    artifact_service: BaseArtifactService  # Where files are stored
    memory_service: BaseMemoryService      # Long-term memory
    credential_service: BaseCredentialService  # OAuth/credentials
    plugin_manager: PluginManager          # Cross-cutting concerns
```

### Runner Creation

```python
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents import LlmAgent

# Create the agent
agent = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

# Create the runner
runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=InMemorySessionService(),
)
```

### Execution Methods

| Method | Use Case |
|--------|----------|
| `run()` | Synchronous, for testing |
| `run_async()` | Asynchronous, for production |
| `run_live()` | Bi-directional streaming (audio/video) |

### The "Reason-Act" Loop

This is the core of ADK's execution model:

```
                    ┌──────────────────────────────┐
                    │                              │
                    ▼                              │
┌─────────────────────────────────────┐           │
│ 1. Build LLM Request                │           │
│    - Gather session history         │           │
│    - Include tool declarations      │           │
│    - Apply system instructions      │           │
└─────────────────────────────────────┘           │
                    │                              │
                    ▼                              │
┌─────────────────────────────────────┐           │
│ 2. Call LLM                         │           │
│    - Send request to model          │           │
│    - Receive response               │           │
└─────────────────────────────────────┘           │
                    │                              │
                    ▼                              │
┌─────────────────────────────────────┐           │
│ 3. Process Response                 │           │
│    - Is it a final response? ───────┼──▶ EXIT   │
│    - Is it a function call?         │           │
└─────────────────────────────────────┘           │
                    │ Yes                          │
                    ▼                              │
┌─────────────────────────────────────┐           │
│ 4. Execute Tool                     │           │
│    - Find matching tool             │           │
│    - Run tool.run_async()           │           │
│    - Create FunctionResponse        │           │
└─────────────────────────────────────┘           │
                    │                              │
                    └──────────────────────────────┘
```

### Running an Agent

```python
from google.genai import types

# Create a session first
session = await runner.session_service.create_session(
    app_name="my_app",
    user_id="user_123",
)

# Run the agent
async for event in runner.run_async(
    user_id="user_123",
    session_id=session.id,
    new_message=types.Content(
        role="user",
        parts=[types.Part(text="Hello, world!")]
    ),
):
    print(f"[{event.author}]: {event.content}")
```

### Why This Matters

The Runner is where **everything comes together**:
- Manages the stateless LLM with stateful sessions
- Coordinates tool execution
- Handles multi-agent delegation
- Applies plugins for cross-cutting concerns
- Streams events for real-time responses

---

## Practice Exercises

### Exercise 1: Read the Code

Open these files and read them in order:
1. `src/google/adk/agents/base_agent.py` (10 min)
2. `src/google/adk/events/event.py` (5 min)
3. `src/google/adk/sessions/session.py` (5 min)
4. `src/google/adk/tools/base_tool.py` (10 min)
5. `src/google/adk/runners.py` lines 1-500 (20 min)

### Exercise 2: Simple Agent

Create a simple agent and run it:

```python
# File: exercises/simple_agent.py
import asyncio
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types

agent = LlmAgent(
    name="greeter",
    model="gemini-2.0-flash",
    instruction="You are a friendly greeter. Keep responses short.",
)

runner = Runner(
    app_name="exercise_1",
    agent=agent,
    session_service=InMemorySessionService(),
)

async def main():
    session = await runner.session_service.create_session(
        app_name="exercise_1",
        user_id="test_user",
    )

    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Hi there!")]
        ),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent: {part.text}")

asyncio.run(main())
```

### Exercise 3: Event Inspector

Write a utility that logs all events in detail:

```python
# File: exercises/event_inspector.py
def inspect_event(event):
    print(f"\n{'='*50}")
    print(f"Event ID: {event.id}")
    print(f"Author: {event.author}")
    print(f"Invocation: {event.invocation_id}")
    print(f"Final Response: {event.is_final_response()}")

    if fc := event.get_function_calls():
        print(f"Function Calls: {[f.name for f in fc]}")

    if fr := event.get_function_responses():
        print(f"Function Responses: {[f.name for f in fr]}")

    if event.content and event.content.parts:
        for i, part in enumerate(event.content.parts):
            if part.text:
                print(f"Part {i} (text): {part.text[:100]}...")
```

---

## Key Takeaways

1. **Agent** = What (configuration/blueprint)
2. **Event** = Data (everything is an event)
3. **Session** = State (conversation + custom data)
4. **Tool** = Action (how agents interact with the world)
5. **Runner** = How (orchestration engine)

Understanding these 5 abstractions lets you predict how any part of ADK works.

---

**Next:** [Phase 2: Request Flow Tracing](../phase-2-request-flow/README.md)
