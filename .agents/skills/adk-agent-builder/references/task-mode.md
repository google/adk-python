# Task Delegation (`mode='task'` / `mode='single_turn'`)

Hand a sub-agent a schema-validated job and get a schema-validated answer back,
instead of transferring the whole conversation to it.

## The three modes

`LlmAgent.mode` is `'chat'`, `'task'`, `'single_turn'`, or unset. Unset means
`'chat'` when the agent is a sub-agent, `'single_turn'` when it is a workflow
node.

| Mode | How the parent reaches it | User interaction | How it finishes |
|---|---|---|---|
| `chat` | the `transfer_to_agent` tool | full conversation | transfers back |
| `task` | a tool named after the sub-agent | can ask the user for clarification | calls `finish_task` |
| `single_turn` | a tool named after the sub-agent | none — told no reply is coming | calls `finish_task` |

The delegation tool takes the sub-agent's **`name`** verbatim. An agent called
`researcher` is exposed to the coordinator as a tool called `researcher`, and
its `description` becomes the tool description, so write the description for a
model deciding whether to call it.

## Task mode

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


researcher = Agent(
    name='researcher',
    mode='task',
    input_schema=ResearchInput,
    output_schema=ResearchOutput,
    description='Researches topics using web search and analysis.',
    instruction=(
        'Research the given topic with search_web. If the user asks for'
        ' changes, adjust. When done, call finish_task with summary,'
        ' key_findings, and confidence.'
    ),
    tools=[search_web],
)

root_agent = Agent(
    name='coordinator',
    model='gemini-2.5-flash',
    sub_agents=[researcher],
    instruction=(
        'When the user asks for research, call the researcher tool. Summarize'
        ' its result for the user.'
    ),
)
```

Sequence: the coordinator calls the `researcher` tool with structured input; the
researcher works, possibly talking to the user; the researcher calls
`finish_task` with structured output; the coordinator gets the result.

## Single-turn mode

Same shape, no conversation. The framework appends a nudge to the sub-agent's
input telling it no further user replies will arrive, so it must finish from the
input alone.

```python
class SummaryOutput(BaseModel):
  summary: str
  word_count: int


summarizer = Agent(
    name='summarizer',
    mode='single_turn',
    output_schema=SummaryOutput,
    description='Summarizes documents autonomously.',
    instruction='Summarize the document with extract_text, then finish_task.',
    tools=[extract_text],
)

root_agent = Agent(
    name='coordinator',
    model='gemini-2.5-flash',
    sub_agents=[summarizer],
    instruction='Delegate summarization to the summarizer tool.',
)
```

## Schemas

`input_schema` types the delegation tool's parameters; `output_schema` types
`finish_task`'s parameters. Both are optional.

```python
agent = Agent(
    name='worker',
    mode='task',
    input_schema=TaskInput,    # validates the delegation call
    output_schema=TaskOutput,  # validates the finish_task call
    ...
)
```

Without them the defaults are a single string each:

```python
# delegation tool parameters
{'request': str}   # "Detailed instructions or context for the task sub-agent."

# finish_task parameters
{'result': str}
```

A schema violation is not fatal — `finish_task` returns a validation-error
message and the model gets to retry.

## `finish_task`

`mode='task'` attaches a tool called `finish_task` to the sub-agent
automatically, and injects an instruction telling the model to complete the work
before calling it. Its parameters come from `output_schema`, or `{'result':
str}` when there is none. There is nothing to import or register.

## Mixed modes under one coordinator

```python
flight_searcher = Agent(
    name='flight_searcher',
    mode='task',                # interactive: can discuss options
    input_schema=FlightSearchInput,
    output_schema=FlightSearchOutput,
    description='Searches and books flights interactively.',
    instruction='Search flights, discuss with the user, then finish_task.',
    tools=[search_flights, book_flight],
)

weather_checker = Agent(
    name='weather_checker',
    mode='single_turn',         # autonomous
    output_schema=WeatherOutput,
    description='Checks weather for a destination.',
    instruction='Check the weather and call finish_task.',
    tools=[get_weather],
)

root_agent = Agent(
    name='travel_planner',
    model='gemini-2.5-flash',
    sub_agents=[flight_searcher, weather_checker],
    instruction=(
        'Plan trips. Use weather_checker for weather and flight_searcher for'
        ' booking.'
    ),
)
```

## Rules worth knowing

- Only the coordinator needs `model=`; sub-agents inherit it from the nearest
  `LlmAgent` ancestor.
- Every delegating sub-agent needs a `description` — it is the entire tool
  description the coordinator's model sees.
- The delegation tool is marked as deferring its response and its description
  tells the model **not** to call it in parallel with other tools. Do not build
  a prompt that asks for several delegations in one turn.
- A `mode='chat'` sub-agent gets neither a delegation tool nor `finish_task`; it
  stays a `transfer_to_agent` target.

## Task mode versus chat transfer

| | chat (`transfer_to_agent`) | task / single_turn |
|---|---|---|
| input | free-form conversation | schema-validated |
| output | free-form conversation | schema-validated |
| control returns when | the agent transfers back | the agent calls `finish_task` |
| user interaction | full chat | `task`: multi-turn, `single_turn`: none |

## Where the code lives

| Component | File |
|---|---|
| `mode`, `input_schema`, `output_schema` | `src/google/adk/agents/llm_agent.py` |
| delegation tools, default input schema | `src/google/adk/tools/agent_tool.py` |
| `finish_task` | `src/google/adk/agents/llm/task/_finish_task_tool.py` |
| `TaskRequest`, `TaskResult` | `src/google/adk/agents/llm/task/_task_models.py` |
