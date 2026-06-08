# Task Mode: Structured Delegation

Delegate structured tasks to sub-agents with typed input/output schemas.

## 📋 Agent Verification Checklist (Task Mode)
Use this checklist to verify your Task Mode configuration:
- [ ] **Mode Setting**: Did you explicitly set `mode='task'` or `mode='single_turn'` on the sub-agent?
- [ ] **Description**: Does the sub-agent have a clear `description`? (Crucial for the auto-generated tool's description)
- [ ] **Schemas**: Are `input_schema` and `output_schema` defined as Pydantic models? (If not, defaults are used)
- [ ] **Completion**: Does the sub-agent know it must call `finish_task` to return results to the coordinator?

## 💡 Quick Reference (Generated Tools)
- **`request_task_{agent_name}`**: Generated on the **coordinator** to delegate tasks.
- **`finish_task`**: Generated on the **sub-agent** to return results and complete the task.

## Overview

ADK agents support three delegation modes via the `mode` parameter on `Agent`:

| Mode | Tool Generated | User Interaction | Completion |
|------|---------------|------------------|------------|
| `chat` (default) | `transfer_to_agent` | Full conversational | Agent transfers back |
| `task` | `request_task_{name}` | Multi-turn (can chat with user) | Calls `finish_task` |
| `single_turn` | `request_task_{name}` | None (autonomous) | Calls `finish_task` |

## Imports

```python
from google.adk import Agent
from pydantic import BaseModel
```

**Note**: Task mode uses `Agent` (aliased from `LlmAgent`) from `google.adk`. Both task sub-agents and coordinators use the same `Agent` class — set `mode='task'` or `mode='single_turn'` on sub-agents.

## Task Mode (`mode='task'`)

A task agent receives structured input via `request_task_{name}`, can interact with the user for clarification, and returns structured output via `finish_task`.

### Delegation Lifecycle

1. User asks the coordinator to do something
2. Coordinator calls `request_task_{agent_name}(...)` with structured input
3. Task agent receives the input, works on it (may use tools, may chat with user)
4. Task agent calls `finish_task(...)` with structured output
5. Coordinator receives the result and responds to the user

### Example

```python
from google.adk import Agent
from pydantic import BaseModel

class ResearchInput(BaseModel):
  topic: str
  depth: str = 'standard'

class ResearchOutput(BaseModel):
  summary: str
  key_findings: str
  confidence: str

def search_web(query: str) -> str:
  """Search the web for information."""
  return f'Results for "{query}": ...'

def analyze_sources(sources: str) -> str:
  """Analyze and synthesize source material."""
  return f'Analysis of {len(sources.split())} words complete.'

researcher = Agent(
    name='researcher',
    mode='task',
    input_schema=ResearchInput,
    output_schema=ResearchOutput,
    instruction=(
        'You are a research assistant. When given a topic:\n'
        '1. Use search_web to find information.\n'
        '2. Use analyze_sources to synthesize findings.\n'
        '3. If the user asks for changes, adjust your research.\n'
        '4. Call finish_task with summary, key_findings, and confidence.'
    ),
    description='Researches topics using web search and analysis.',
    tools=[search_web, analyze_sources],
)

root_agent = Agent(
    name='coordinator',
    model='gemini-2.5-flash',
    sub_agents=[researcher],
    instruction=(
        'When the user asks you to research something, delegate to'
        ' the researcher using request_task_researcher. After the'
        ' researcher completes, summarize the results for the user.'
    ),
)
```

## Single-Turn Mode (`mode='single_turn'`)

A single-turn agent completes autonomously with no user interaction. It receives input, does its work, and returns a result.

### Example

```python
class SummaryOutput(BaseModel):
  summary: str
  word_count: int
  key_points: str

def extract_text(url: str) -> str:
  """Extract text from a URL."""
  return f'Extracted content from {url}: ...'

summarizer = Agent(
    name='summarizer',
    mode='single_turn',
    output_schema=SummaryOutput,
    instruction=(
        'Summarize the document:\n'
        '1. Use extract_text to get content.\n'
        '2. Call finish_task with summary, word_count, key_points.\n'
        'Complete autonomously without user interaction.'
    ),
    description='Summarizes documents autonomously.',
    tools=[extract_text],
)

root_agent = Agent(
    name='coordinator',
    model='gemini-2.5-flash',
    sub_agents=[summarizer],
    instruction='Delegate summarization to summarizer via request_task_summarizer.',
)
```

## Input and Output Schemas

### Custom Schemas (Pydantic Models)

Define `input_schema` and/or `output_schema` with Pydantic `BaseModel`:

```python
class TaskInput(BaseModel):
  query: str
  max_results: int = 10
  format: str = 'text'

class TaskOutput(BaseModel):
  results: str
  count: int
  status: str

agent = Agent(
    name='worker',
    mode='task',
    input_schema=TaskInput,    # Validates request_task_worker args
    output_schema=TaskOutput,  # Validates finish_task args
    ...
)
```

### Default Schemas

When no custom schema is provided:

**Default input** (used by `request_task_{name}`):
```python
class _DefaultTaskInput(BaseModel):
  goal: str | None = None
  background: str | None = None
```

**Default output** (used by `finish_task`):
```python
class _DefaultTaskOutput(BaseModel):
  result: str
```

## Auto-Generated Tools

### `request_task_{agent_name}`

Auto-generated on the **coordinator** for each `mode='task'` or `mode='single_turn'` sub-agent. The tool name is `request_task_{agent.name}`.

- Parameters come from `input_schema` (or default: `goal`, `background`)
- Description includes the agent's `description` field
- Validates input against the schema before delegating

### `finish_task`

Auto-generated on the **task agent** itself. Called by the task agent when work is complete.

- Parameters come from `output_schema` (or default: `result`)
- Validates output against the schema before signaling completion
- Sets `tool_context.actions.finish_task` with a `TaskResult`

## Mixed-Mode Patterns

Combine task and single-turn agents under one coordinator:

```python
# Interactive: user can discuss options
flight_searcher = Agent(
    name='flight_searcher',
    mode='task',
    input_schema=FlightSearchInput,
    output_schema=FlightSearchOutput,
    instruction='Search flights, discuss with user, then finish_task.',
    description='Searches and books flights interactively.',
    tools=[search_flights, book_flight],
)

# Autonomous: no user interaction
weather_checker = Agent(
    name='weather_checker',
    mode='single_turn',
    output_schema=WeatherOutput,
    instruction='Check weather and call finish_task. No user interaction.',
    description='Checks weather for a destination.',
    tools=[get_weather],
)

# Autonomous: no user interaction
hotel_finder = Agent(
    name='hotel_finder',
    mode='single_turn',
    output_schema=HotelOutput,
    instruction='Find hotels and call finish_task. No user interaction.',
    description='Finds hotels for a destination.',
    tools=[find_hotels],
)

root_agent = Agent(
    name='travel_planner',
    model='gemini-2.5-flash',
    sub_agents=[flight_searcher, weather_checker, hotel_finder],
    instruction=(
        'Help users plan trips:\n'
        '- request_task_weather_checker: autonomous weather check\n'
        '- request_task_hotel_finder: autonomous hotel search\n'
        '- request_task_flight_searcher: interactive flight booking'
    ),
)
```

## Key Rules

- Both task sub-agents and coordinators use `Agent` from `google.adk`
- Each sub-agent needs a `description` (used in the auto-generated tool description)
- `input_schema` and `output_schema` are optional; defaults are provided
- Sub-agents inherit model from the coordinator if not set
- `finish_task` instructions are auto-injected into the task agent's LLM context
- Single-turn agents receive an extra instruction telling them no user replies will come

## Task Mode vs Chat Mode

| Feature | Chat (`transfer_to_agent`) | Task (`request_task`) |
|---------|---------------------------|----------------------|
| Input | Free-form conversation | Structured (schema-validated) |
| Output | Free-form conversation | Structured (schema-validated) |
| Control flow | Agent decides when to transfer back | Agent calls `finish_task` |
| User interaction | Full chat | `task`: multi-turn; `single_turn`: none |
| Tool name | `transfer_to_agent` | `request_task_{name}` |
| Parallel delegation | Not supported | Supported (multiple `request_task` calls) |

## Source File Locations

| Component | File |
|-----------|------|
| Agent/LlmAgent (mode, schemas) | `src/google/adk/agents/llm_agent.py` |
| BaseLlmFlow (base flow class) | `src/google/adk/flows/llm_flows/base_llm_flow.py` |
| RequestTaskTool | `src/google/adk/agents/llm/task/_request_task_tool.py` |
| FinishTaskTool | `src/google/adk/agents/llm/task/_finish_task_tool.py` |
| TaskRequest, TaskResult | `src/google/adk/agents/llm/task/_task_models.py` |
| Task samples | `contributing/task_samples/` |
