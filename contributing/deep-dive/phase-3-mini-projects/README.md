# Phase 3: Mini-Projects

**Duration:** Week 1 (4-5 days)
**Outcome:** Hands-on experience with 70% of the ADK codebase

---

## Overview

The best way to learn ADK is to build with it. This phase guides you through three progressively complex projects, each teaching different aspects of the framework.

| Project | What You'll Learn | Modules Covered |
|---------|------------------|-----------------|
| 1. Custom Tool Agent | Tool creation, function calling | `tools/`, `agents/llm_agent.py` |
| 2. Multi-Agent Workflow | Agent orchestration, delegation | `SequentialAgent`, `ParallelAgent`, transfers |
| 3. Stateful Chatbot | Persistence, memory, artifacts | `sessions/`, `memory/`, `artifacts/` |

---

## Project 1: Custom Tool Agent

**Goal:** Build an agent with custom tools that interact with external services.

### What You'll Build

A "Research Assistant" agent with these tools:
- `search_wikipedia` - Search Wikipedia articles
- `get_weather` - Get current weather
- `calculate` - Perform calculations
- `save_note` - Save notes to session state

### Step 1: Create the Tools

```python
# File: projects/research_assistant/tools.py
from typing import Any
import httpx
from google.adk.tools import FunctionTool
from google.adk.tools import ToolContext


async def search_wikipedia(query: str) -> dict:
    """Search Wikipedia for information about a topic.

    Args:
        query: The search term to look up on Wikipedia.

    Returns:
        A dictionary containing the search results with title and snippet.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 3,
            },
        )
        data = response.json()
        results = data.get("query", {}).get("search", [])
        return {
            "results": [
                {"title": r["title"], "snippet": r["snippet"]}
                for r in results
            ]
        }


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression like "2 + 2" or "sqrt(16)".

    Returns:
        The result of the calculation.
    """
    import math

    # Safe evaluation with limited scope
    allowed_names = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}


async def save_note(
    note: str,
    category: str,
    tool_context: ToolContext,
) -> dict:
    """Save a note to the session for later reference.

    Args:
        note: The content of the note to save.
        category: A category for organizing the note (e.g., "research", "todo").
        tool_context: The tool context (automatically injected).

    Returns:
        Confirmation of the saved note.
    """
    # Access session state through tool_context
    notes = tool_context.state.get("notes", [])
    new_note = {
        "content": note,
        "category": category,
        "timestamp": __import__("time").time(),
    }
    notes.append(new_note)
    tool_context.state["notes"] = notes

    return {
        "status": "saved",
        "note_count": len(notes),
        "category": category,
    }


async def get_notes(
    category: str | None,
    tool_context: ToolContext,
) -> dict:
    """Retrieve saved notes, optionally filtered by category.

    Args:
        category: Optional category to filter notes. If None, returns all notes.
        tool_context: The tool context (automatically injected).

    Returns:
        List of saved notes.
    """
    notes = tool_context.state.get("notes", [])

    if category:
        notes = [n for n in notes if n["category"] == category]

    return {"notes": notes, "count": len(notes)}


# Create FunctionTool instances
wikipedia_tool = FunctionTool(func=search_wikipedia)
calculate_tool = FunctionTool(func=calculate)
save_note_tool = FunctionTool(func=save_note)
get_notes_tool = FunctionTool(func=get_notes)
```

### Step 2: Create the Agent

```python
# File: projects/research_assistant/agent.py
from google.adk.agents import LlmAgent
from .tools import (
    wikipedia_tool,
    calculate_tool,
    save_note_tool,
    get_notes_tool,
)

INSTRUCTION = """You are a research assistant that helps users find information
and take notes.

You have access to these tools:
- search_wikipedia: Search for information on Wikipedia
- calculate: Perform mathematical calculations
- save_note: Save notes for later reference
- get_notes: Retrieve saved notes

When the user asks a question:
1. Search for relevant information if needed
2. Perform calculations if needed
3. Offer to save important findings as notes
4. Always cite your sources

Be concise but thorough in your responses."""

root_agent = LlmAgent(
    name="research_assistant",
    model="gemini-2.0-flash",
    instruction=INSTRUCTION,
    tools=[
        wikipedia_tool,
        calculate_tool,
        save_note_tool,
        get_notes_tool,
    ],
)
```

### Step 3: Create the Runner

```python
# File: projects/research_assistant/main.py
import asyncio
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from .agent import root_agent


async def main():
    runner = Runner(
        app_name="research_assistant",
        agent=root_agent,
        session_service=InMemorySessionService(),
    )

    session = await runner.session_service.create_session(
        app_name="research_assistant",
        user_id="researcher",
    )

    # Interactive loop
    print("Research Assistant Ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        print("Assistant: ", end="", flush=True)

        async for event in runner.run_async(
            user_id="researcher",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=user_input)],
            ),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text and event.is_final_response():
                        print(part.text)

        print()  # Newline after response


if __name__ == "__main__":
    asyncio.run(main())
```

### What You Learned

- Creating `FunctionTool` from Python functions
- Using `ToolContext` to access session state
- Async tool implementations
- Tool parameter documentation (docstrings → LLM descriptions)

---

## Project 2: Multi-Agent Workflow

**Goal:** Build a system where multiple specialized agents collaborate.

### What You'll Build

A "Content Creation Pipeline" with:
- **Researcher Agent** - Gathers information
- **Writer Agent** - Creates content
- **Editor Agent** - Reviews and improves
- **Coordinator Agent** - Orchestrates the workflow

### Step 1: Define Specialized Agents

```python
# File: projects/content_pipeline/agents.py
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool


# Tool for the researcher
async def web_search(query: str) -> dict:
    """Search the web for information."""
    # Simulated search results
    return {
        "results": [
            f"Found information about: {query}",
            "Key points: ...",
            "Statistics: ...",
        ]
    }


# Researcher Agent
researcher = LlmAgent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="""You are a research specialist. Your job is to:
1. Analyze the topic given to you
2. Use the web_search tool to find relevant information
3. Compile a research brief with key facts and statistics
4. Output a structured research summary

Always be thorough and cite your findings.""",
    tools=[FunctionTool(func=web_search)],
)


# Writer Agent
writer = LlmAgent(
    name="writer",
    model="gemini-2.0-flash",
    instruction="""You are a content writer. Your job is to:
1. Read the research brief provided
2. Create engaging, well-structured content
3. Use clear headings and bullet points
4. Maintain a professional but accessible tone

Output a complete draft article based on the research.""",
)


# Editor Agent
editor = LlmAgent(
    name="editor",
    model="gemini-2.0-flash",
    instruction="""You are an editor. Your job is to:
1. Review the draft content
2. Fix grammar and spelling issues
3. Improve clarity and flow
4. Ensure the content is engaging

Output the final polished version with your edits.""",
)


# Sequential Pipeline: Researcher → Writer → Editor
content_pipeline = SequentialAgent(
    name="content_pipeline",
    description="Creates polished content through research, writing, and editing",
    sub_agents=[researcher, writer, editor],
)


# Coordinator that can choose to use the pipeline
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.0-flash",
    instruction="""You are a content coordinator. When users request content:

1. Understand what type of content they need
2. Delegate to the content_pipeline for creation
3. Present the final result to the user

For simple questions, answer directly without using the pipeline.""",
    sub_agents=[content_pipeline],
)

root_agent = coordinator
```

### Step 2: Alternative - Parallel Execution

For tasks that can run simultaneously:

```python
# File: projects/content_pipeline/parallel_agents.py
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent


# Multiple researchers working in parallel
tech_researcher = LlmAgent(
    name="tech_researcher",
    model="gemini-2.0-flash",
    instruction="Research technical aspects of the topic.",
)

market_researcher = LlmAgent(
    name="market_researcher",
    model="gemini-2.0-flash",
    instruction="Research market trends and business aspects.",
)

social_researcher = LlmAgent(
    name="social_researcher",
    model="gemini-2.0-flash",
    instruction="Research social impact and public perception.",
)

# Run all researchers in parallel
parallel_research = ParallelAgent(
    name="parallel_research",
    description="Conducts multi-faceted research simultaneously",
    sub_agents=[tech_researcher, market_researcher, social_researcher],
)

# Then synthesize results
synthesizer = LlmAgent(
    name="synthesizer",
    model="gemini-2.0-flash",
    instruction="""Combine the research from all three researchers into
a comprehensive summary. Identify common themes and contradictions.""",
)

# Full pipeline: Parallel Research → Synthesis
research_pipeline = SequentialAgent(
    name="research_pipeline",
    sub_agents=[parallel_research, synthesizer],
)

root_agent = research_pipeline
```

### Step 3: Using Transfer Tools

For dynamic agent switching:

```python
# File: projects/content_pipeline/transfer_agents.py
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool


# Specialist agents
coder = LlmAgent(
    name="coder",
    model="gemini-2.0-flash",
    instruction="You are a coding expert. Write and explain code.",
)

analyst = LlmAgent(
    name="analyst",
    model="gemini-2.0-flash",
    instruction="You are a data analyst. Analyze data and create insights.",
)

# Router agent that can transfer to specialists
router = LlmAgent(
    name="router",
    model="gemini-2.0-flash",
    instruction="""You are a helpful assistant. Route requests to specialists:
- For coding questions, transfer to the coder
- For data analysis, transfer to the analyst
- For general questions, answer directly""",
    sub_agents=[coder, analyst],
    # The agent can naturally use transfer_to_agent for sub_agents
)

root_agent = router
```

### What You Learned

- `SequentialAgent` for ordered pipelines
- `ParallelAgent` for concurrent execution
- Agent-to-agent delegation patterns
- Composing complex workflows from simple agents

---

## Project 3: Stateful Chatbot

**Goal:** Build a chatbot with persistent memory and file handling.

### What You'll Build

A "Personal Assistant" that:
- Remembers user preferences across sessions
- Stores and retrieves documents
- Uses long-term memory for context

### Step 1: Set Up Persistent Services

```python
# File: projects/personal_assistant/services.py
from google.adk.sessions import DatabaseSessionService
from google.adk.artifacts import FileArtifactService
from google.adk.memory import InMemoryMemoryService


# SQLite for session persistence
session_service = DatabaseSessionService(
    db_url="sqlite:///./personal_assistant.db"
)

# File system for artifacts
artifact_service = FileArtifactService(
    base_path="./artifacts"
)

# Memory service for long-term recall
memory_service = InMemoryMemoryService()
```

### Step 2: Create Memory-Aware Tools

```python
# File: projects/personal_assistant/tools.py
from google.adk.tools import FunctionTool
from google.adk.tools import ToolContext


async def remember_preference(
    key: str,
    value: str,
    tool_context: ToolContext,
) -> dict:
    """Remember a user preference for future sessions.

    Args:
        key: The preference name (e.g., "favorite_color", "timezone").
        value: The preference value.
        tool_context: Automatically injected context.

    Returns:
        Confirmation of saved preference.
    """
    # Use 'user:' prefix for user-scoped state
    tool_context.state[f"user:preference:{key}"] = value
    return {"status": "remembered", "key": key, "value": value}


async def recall_preference(
    key: str,
    tool_context: ToolContext,
) -> dict:
    """Recall a previously saved user preference.

    Args:
        key: The preference name to recall.
        tool_context: Automatically injected context.

    Returns:
        The preference value if found.
    """
    value = tool_context.state.get(f"user:preference:{key}")
    if value:
        return {"found": True, "key": key, "value": value}
    return {"found": False, "key": key}


async def save_document(
    filename: str,
    content: str,
    tool_context: ToolContext,
) -> dict:
    """Save a document as an artifact.

    Args:
        filename: Name for the document.
        content: The document content.
        tool_context: Automatically injected context.

    Returns:
        Confirmation with artifact info.
    """
    from google.genai import types

    artifact = types.Part(
        inline_data=types.Blob(
            mime_type="text/plain",
            data=content.encode("utf-8"),
        )
    )

    version = await tool_context.save_artifact(
        filename=filename,
        artifact=artifact,
    )

    return {
        "status": "saved",
        "filename": filename,
        "version": version,
    }


async def load_document(
    filename: str,
    tool_context: ToolContext,
) -> dict:
    """Load a previously saved document.

    Args:
        filename: Name of the document to load.
        tool_context: Automatically injected context.

    Returns:
        The document content.
    """
    artifact = await tool_context.load_artifact(filename=filename)

    if artifact and artifact.inline_data:
        content = artifact.inline_data.data.decode("utf-8")
        return {"found": True, "filename": filename, "content": content}
    return {"found": False, "filename": filename}


async def list_documents(tool_context: ToolContext) -> dict:
    """List all saved documents.

    Args:
        tool_context: Automatically injected context.

    Returns:
        List of document filenames.
    """
    filenames = await tool_context.list_artifacts()
    return {"documents": filenames, "count": len(filenames)}


# Create tools
remember_tool = FunctionTool(func=remember_preference)
recall_tool = FunctionTool(func=recall_preference)
save_doc_tool = FunctionTool(func=save_document)
load_doc_tool = FunctionTool(func=load_document)
list_docs_tool = FunctionTool(func=list_documents)
```

### Step 3: Create the Assistant with Memory

```python
# File: projects/personal_assistant/agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import LoadMemoryTool
from .tools import (
    remember_tool,
    recall_tool,
    save_doc_tool,
    load_doc_tool,
    list_docs_tool,
)

INSTRUCTION = """You are a personal assistant with persistent memory.

Key capabilities:
1. Remember user preferences (favorite things, settings, etc.)
2. Store and retrieve documents
3. Access long-term memory from previous conversations

When starting a conversation:
- Check if you have any remembered preferences
- Reference past interactions when relevant

Always be helpful and personalized based on what you know about the user."""

root_agent = LlmAgent(
    name="personal_assistant",
    model="gemini-2.0-flash",
    instruction=INSTRUCTION,
    tools=[
        remember_tool,
        recall_tool,
        save_doc_tool,
        load_doc_tool,
        list_docs_tool,
        LoadMemoryTool(),  # Built-in tool for memory retrieval
    ],
)
```

### Step 4: Wire Everything Together

```python
# File: projects/personal_assistant/main.py
import asyncio
from google.adk import Runner
from google.genai import types
from .agent import root_agent
from .services import session_service, artifact_service, memory_service


async def main():
    runner = Runner(
        app_name="personal_assistant",
        agent=root_agent,
        session_service=session_service,
        artifact_service=artifact_service,
        memory_service=memory_service,
    )

    # Get or create session for this user
    user_id = "user_001"
    sessions = await session_service.list_sessions(
        app_name="personal_assistant",
        user_id=user_id,
    )

    if sessions:
        session = sessions[0]  # Resume existing session
        print(f"Resuming session: {session.id}")
    else:
        session = await session_service.create_session(
            app_name="personal_assistant",
            user_id=user_id,
        )
        print(f"Created new session: {session.id}")

    # Show current state
    print(f"Current preferences: {session.state}")
    print()

    # Interactive loop
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=user_input)],
            ),
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        print(f"Assistant: {part.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

### What You Learned

- `DatabaseSessionService` for persistent sessions
- `FileArtifactService` for file storage
- State management with `user:` prefixes
- Artifact save/load through `ToolContext`
- Memory integration for long-term recall

---

## Summary of Modules Covered

After completing all three projects, you've worked with:

| Module | Project |
|--------|---------|
| `tools/base_tool.py` | 1, 2, 3 |
| `tools/function_tool.py` | 1, 2, 3 |
| `tools/tool_context.py` | 1, 3 |
| `agents/llm_agent.py` | 1, 2, 3 |
| `agents/sequential_agent.py` | 2 |
| `agents/parallel_agent.py` | 2 |
| `sessions/` | 3 |
| `artifacts/` | 3 |
| `memory/` | 3 |
| `runners.py` | 1, 2, 3 |

---

## Next Steps

After completing these projects:

1. **Extend Project 1** - Add more sophisticated tools (databases, APIs)
2. **Extend Project 2** - Add error handling and retry logic
3. **Extend Project 3** - Add image and file upload handling

---

**Next:** [Phase 4: Abstraction Layers](../phase-4-abstraction-layers/README.md)
