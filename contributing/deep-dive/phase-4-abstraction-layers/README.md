# Phase 4: Abstraction Layers

**Duration:** Week 2
**Outcome:** Recognize ADK's consistent design patterns and extend any module

---

## Overview

ADK follows a **consistent service abstraction pattern** across all modules. Once you recognize this pattern, you can navigate and extend any part of the codebase.

## The Universal Pattern

Every service in ADK follows this structure:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Base[Service]                             │
│                  (Abstract Base Class)                           │
│                                                                  │
│  @abstractmethod create_X()                                      │
│  @abstractmethod get_X()                                         │
│  @abstractmethod list_X()                                        │
│  @abstractmethod delete_X()                                      │
│  [shared implementation methods]                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  InMemory   │    │  Database   │    │  Cloud      │
    │  [Service]  │    │  [Service]  │    │  [Service]  │
    │             │    │             │    │             │
    │ (Testing)   │    │ (Local)     │    │ (Production)│
    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Pattern Analysis: Sessions

**File:** `src/google/adk/sessions/`

### Base Class

```python
# base_session_service.py
class BaseSessionService(abc.ABC):
    """Base class for session services."""

    @abc.abstractmethod
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """Creates a new session."""

    @abc.abstractmethod
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """Gets a session."""

    @abc.abstractmethod
    async def list_sessions(
        self, *, app_name: str, user_id: Optional[str] = None
    ) -> ListSessionsResponse:
        """Lists all sessions for a user."""

    @abc.abstractmethod
    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        """Deletes a session."""

    # SHARED IMPLEMENTATION - Used by all subclasses
    async def append_event(self, session: Session, event: Event) -> Event:
        """Appends an event to a session object."""
        if event.partial:
            return event
        event = self._trim_temp_delta_state(event)
        self._update_session_state(session, event)
        session.events.append(event)
        return event
```

### Implementations

| Implementation | Backend | Use Case |
|----------------|---------|----------|
| `InMemorySessionService` | Python dict | Testing, development |
| `SQLiteSessionService` | SQLite file | Local persistence |
| `DatabaseSessionService` | SQLAlchemy | PostgreSQL, CloudSQL |
| `VertexAiSessionService` | Vertex AI API | Managed cloud |

### Pattern Recognition

```python
# All implementations share the same interface
session_service: BaseSessionService = InMemorySessionService()
# OR
session_service: BaseSessionService = DatabaseSessionService(db_url="...")
# OR
session_service: BaseSessionService = VertexAiSessionService()

# Code that uses the service doesn't change
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
)
```

---

## Pattern Analysis: Artifacts

**File:** `src/google/adk/artifacts/`

### Base Class

```python
# base_artifact_service.py
class BaseArtifactService(ABC):
    """Base class for artifact services."""

    @abstractmethod
    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: types.Part,
    ) -> int:
        """Saves an artifact."""

    @abstractmethod
    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None,
    ) -> Optional[types.Part]:
        """Loads an artifact."""

    @abstractmethod
    async def list_artifact_keys(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> list[str]:
        """Lists artifact keys."""

    @abstractmethod
    async def delete_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
    ) -> None:
        """Deletes an artifact."""
```

### Implementations

| Implementation | Backend | Use Case |
|----------------|---------|----------|
| `InMemoryArtifactService` | Python dict | Testing |
| `FileArtifactService` | Local filesystem | Development |
| `GcsArtifactService` | Google Cloud Storage | Production |

---

## Pattern Analysis: Memory

**File:** `src/google/adk/memory/`

### Base Class

```python
# base_memory_service.py
class BaseMemoryService(ABC):
    """Base class for memory services."""

    @abstractmethod
    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Adds a memory entry."""

    @abstractmethod
    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[MemoryEntry]:
        """Searches memory entries."""
```

### Implementations

| Implementation | Backend | Use Case |
|----------------|---------|----------|
| `InMemoryMemoryService` | Python list | Testing |
| `VertexAiMemoryBankService` | Memory Bank API | Managed cloud |
| `VertexAiRagMemoryService` | Vertex AI RAG | RAG-based memory |

---

## Pattern Analysis: Models (LLM)

**File:** `src/google/adk/models/`

### Base Class

```python
# base_llm.py
class BaseLlm(ABC):
    """Base class for LLM integrations."""

    model: str

    @abstractmethod
    async def generate_content_async(
        self,
        request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generates content from the model."""

    @abstractmethod
    def supported_models(self) -> list[str]:
        """Returns list of supported model names."""
```

### Implementations

| Implementation | Backend | Models |
|----------------|---------|--------|
| `GoogleLLM` | Google AI / Vertex AI | Gemini |
| `AnthropicLLM` | Anthropic API | Claude |
| `LiteLLM` | LiteLLM library | OpenAI, Azure, local |
| `GemmaLLM` | Ollama | Gemma |

---

## Pattern Analysis: Plugins

**File:** `src/google/adk/plugins/`

The plugin system provides cross-cutting concerns using a callback pattern:

### Base Class

```python
# base_plugin.py
class BasePlugin(ABC):
    """Base class for plugins."""

    def __init__(self, name: str):
        self.name = name

    # Lifecycle callbacks
    async def before_run_callback(self, *, invocation_context) -> Optional[Content]:
        pass

    async def on_event_callback(self, *, invocation_context, event) -> Optional[Event]:
        pass

    async def after_run_callback(self, *, invocation_context) -> None:
        pass

    # Agent callbacks
    async def before_agent_callback(self, *, agent, callback_context) -> Optional[Content]:
        pass

    async def after_agent_callback(self, *, agent, callback_context) -> Optional[Content]:
        pass

    # Model callbacks
    async def before_model_callback(self, *, callback_context, llm_request) -> Optional[LlmResponse]:
        pass

    async def after_model_callback(self, *, callback_context, llm_response) -> Optional[LlmResponse]:
        pass

    async def on_model_error_callback(self, *, callback_context, llm_request, error) -> Optional[LlmResponse]:
        pass

    # Tool callbacks
    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[dict]:
        pass

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> Optional[dict]:
        pass

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error) -> Optional[dict]:
        pass

    # Cleanup
    async def close(self) -> None:
        pass
```

### Built-in Plugins

| Plugin | Purpose |
|--------|---------|
| `LoggingPlugin` | Log events and interactions |
| `BigQueryAgentAnalyticsPlugin` | Log to BigQuery |
| `ReflectRetryToolPlugin` | Auto-retry failed tools |
| `GlobalInstructionPlugin` | Add instructions to all agents |
| `ContextFilterPlugin` | Filter context before LLM |
| `SaveFilesAsArtifactsPlugin` | Auto-save file artifacts |

### Creating a Custom Plugin

```python
# File: my_plugin.py
from google.adk.plugins import BasePlugin
from google.adk.events import Event
import time


class TimingPlugin(BasePlugin):
    """Plugin that tracks execution time."""

    def __init__(self):
        super().__init__(name="timing_plugin")
        self.start_time = None
        self.tool_timings = []

    async def before_run_callback(self, *, invocation_context):
        self.start_time = time.time()
        print(f"[{self.name}] Invocation started")
        return None  # Don't short-circuit

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        tool_context.state["_tool_start"] = time.time()
        print(f"[{self.name}] Tool '{tool.name}' starting")
        return None  # Don't short-circuit

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        start = tool_context.state.get("_tool_start", time.time())
        duration = time.time() - start
        self.tool_timings.append({"tool": tool.name, "duration": duration})
        print(f"[{self.name}] Tool '{tool.name}' took {duration:.2f}s")
        return None  # Don't modify result

    async def after_run_callback(self, *, invocation_context):
        total = time.time() - self.start_time
        print(f"[{self.name}] Invocation completed in {total:.2f}s")
        print(f"[{self.name}] Tool timings: {self.tool_timings}")


# Usage
from google.adk import Runner
from google.adk.apps import App

app = App(
    name="my_app",
    root_agent=my_agent,
    plugins=[TimingPlugin()],
)

runner = Runner(
    app=app,
    session_service=session_service,
)
```

---

## Pattern Analysis: Tools

**File:** `src/google/adk/tools/`

### Base Class

```python
# base_tool.py
class BaseTool(ABC):
    """Base class for all tools."""

    name: str
    description: str
    is_long_running: bool = False

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        """Gets the OpenAPI specification."""
        return None

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        """Runs the tool."""
        raise NotImplementedError()

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
        """Processes the outgoing LLM request."""
        llm_request.append_tools([self])
```

### Tool Implementations

| Tool Type | Purpose |
|-----------|---------|
| `FunctionTool` | Wrap Python functions |
| `AgentTool` | Delegate to sub-agents |
| `LongRunningFunctionTool` | Async operations |
| `GoogleSearchTool` | Web search |
| `OpenAPITool` | REST API calls |
| `MCPToolset` | MCP protocol |

---

## Creating Your Own Service

Follow this pattern to create a custom service:

### Step 1: Define the Interface

```python
# my_service/base_my_service.py
from abc import ABC, abstractmethod
from typing import Optional


class BaseMyService(ABC):
    """Base class for my custom service."""

    @abstractmethod
    async def create_item(
        self, *, name: str, data: dict
    ) -> str:
        """Creates an item. Returns the item ID."""

    @abstractmethod
    async def get_item(
        self, *, item_id: str
    ) -> Optional[dict]:
        """Gets an item by ID."""

    @abstractmethod
    async def list_items(self) -> list[dict]:
        """Lists all items."""

    @abstractmethod
    async def delete_item(self, *, item_id: str) -> None:
        """Deletes an item."""

    # Shared implementation
    def validate_name(self, name: str) -> bool:
        """Validates item name."""
        return bool(name and len(name) <= 100)
```

### Step 2: Create In-Memory Implementation

```python
# my_service/in_memory_my_service.py
from typing import Optional
import uuid
from .base_my_service import BaseMyService


class InMemoryMyService(BaseMyService):
    """In-memory implementation for testing."""

    def __init__(self):
        self._items: dict[str, dict] = {}

    async def create_item(self, *, name: str, data: dict) -> str:
        if not self.validate_name(name):
            raise ValueError("Invalid name")
        item_id = str(uuid.uuid4())
        self._items[item_id] = {"id": item_id, "name": name, "data": data}
        return item_id

    async def get_item(self, *, item_id: str) -> Optional[dict]:
        return self._items.get(item_id)

    async def list_items(self) -> list[dict]:
        return list(self._items.values())

    async def delete_item(self, *, item_id: str) -> None:
        self._items.pop(item_id, None)
```

### Step 3: Create Production Implementation

```python
# my_service/database_my_service.py
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from .base_my_service import BaseMyService


class DatabaseMyService(BaseMyService):
    """Database implementation for production."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    async def create_item(self, *, name: str, data: dict) -> str:
        async with self._session_factory() as session:
            # Insert into database
            ...

    async def get_item(self, *, item_id: str) -> Optional[dict]:
        async with self._session_factory() as session:
            # Query from database
            ...

    async def list_items(self) -> list[dict]:
        async with self._session_factory() as session:
            # Query all from database
            ...

    async def delete_item(self, *, item_id: str) -> None:
        async with self._session_factory() as session:
            # Delete from database
            ...
```

---

## Dependency Injection Pattern

ADK uses constructor injection for flexibility:

```python
# The Runner accepts any implementation of each service
runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=InMemorySessionService(),      # For testing
    artifact_service=FileArtifactService("./data"), # Local storage
    memory_service=InMemoryMemoryService(),         # For testing
)

# For production, swap implementations
runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=DatabaseSessionService(db_url="..."),
    artifact_service=GcsArtifactService(bucket="..."),
    memory_service=VertexAiMemoryBankService(),
)
```

---

## Key Takeaways

1. **Every service has a Base class** - Abstract methods define the interface
2. **In-memory implementations for testing** - Fast, no external dependencies
3. **Cloud implementations for production** - Scalable, managed
4. **Shared logic in base class** - DRY principle
5. **Constructor injection** - Easy to swap implementations
6. **Plugins use callback pattern** - Hook into any execution point

Once you understand these patterns, you can:
- Navigate any module quickly
- Create custom implementations
- Extend ADK for your needs

---

**Next:** [Phase 5: Deep Dive Tracks](../phase-5-deep-dive-tracks/README.md)
