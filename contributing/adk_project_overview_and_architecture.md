# ADK Project Overview and Architecture

Google Agent Development Kit (ADK) for Python

## Core Philosophy & Architecture

- Code-First: Everything is defined in Python code for versioning, testing, and IDE support. Avoid GUI-based logic.

- Modularity & Composition: We build complex multi-agent systems by composing multiple, smaller, specialized agents.

- Deployment-Agnostic: The agent's core logic is separate from its deployment environment. The same agent.py can be run locally for testing, served via an API, or deployed to the cloud.

## Foundational Abstractions (Our Vocabulary)

- Agent: The blueprint. It defines an agent's identity, instructions, and tools. It's a declarative configuration object.

- Tool: A capability. A Python function an agent can call to interact with the world (e.g., search, API call).

- Runner: The engine. It orchestrates the "Reason-Act" loop, manages LLM calls, and executes tools.

- Session: The conversation state. It holds the history for a single, continuous dialogue.

- Memory: Long-term recall across different sessions.

- Artifact Service: Manages non-textual data like files.

## Canonical Project Structure

Adhere to this structure for compatibility with ADK tooling.

```
my_adk_project/
└── src/
    └── my_app/
        ├── agents/
        │   ├── my_agent/
        │   │   ├── __init__.py   # Must contain: from. import agent \
        │   │   └── agent.py      # Must contain: root_agent = Agent(...) \
        │   └── another_agent/
        │       ├── __init__.py
        │       └── agent.py\
```

agent.py: Must define the agent and assign it to a variable named root_agent. This is how ADK's tools find it.

`__init__.py`: In each agent directory, it must contain from. import agent to make the agent discoverable.

## Local Development & Debugging

Interactive UI (adk web): This is our primary debugging tool. It's a decoupled system:

Backend: A FastAPI server started with adk api_server.

Frontend: An Angular app that connects to the backend.

Use the "Events" tab to inspect the full execution trace (prompts, tool calls, responses).

CLI (adk run): For quick, stateless functional checks in the terminal.

Programmatic (pytest): For writing automated unit and integration tests.

## The API Layer (FastAPI)

We expose agents as production APIs using FastAPI.

- get_fast_api_app: This is the key helper function from google.adk.cli.fast_api that creates a FastAPI app from our agent directory.

- Standard Endpoints: The generated app includes standard routes like /list-apps and /run_sse for streaming responses. The wire format is camelCase.

- Custom Endpoints: We can add our own routes (e.g., /health) to the app object returned by the helper.

Python

from google.adk.cli.fast_api import get_fast_api_app
app = get_fast_api_app(agent_dir="./agents")

@app.get("/health")
async def health_check():
    return {"status": "ok"}


## Deployment to Production

The adk cli provides the "adk deploy" command to deploy to Google Vertex Agent Engine, Google CloudRun, Google GKE.

## Testing & Evaluation Strategy

Testing is layered, like a pyramid.

### Layer 1: Unit Tests (Base)

What: Test individual Tool functions in isolation.

How: Use pytest in tests/test_tools.py. Verify deterministic logic.

### Layer 2: Integration Tests (Middle)

What: Test the agent's internal logic and interaction with tools.

How: Use pytest in tests/test_agent.py, often with mocked LLMs or services.

### Layer 3: Evaluation Tests (Top)

What: Assess end-to-end performance with a live LLM. This is about quality, not just pass/fail.

How: Use the ADK Evaluation Framework.

Test Cases: Create JSON files with input and a reference (expected tool calls and final response).

Metrics: tool_trajectory_avg_score (does it use tools correctly?) and response_match_score (is the final answer good?).

Run via: adk web (UI), pytest (for CI/CD), or adk eval (CLI).
