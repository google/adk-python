# Getting Started: Creating ADK Agents

Environment, the agent directory convention, a first LLM agent, and the jump to
graph workflows.

## CLI commands

| Command | What it does |
|---|---|
| `adk create {agent_name}` | Scaffolds an agent directory |
| `adk run {agent_dir}` | Runs the agent in the terminal |
| `adk web {agent_dir}` | Dev server on `http://localhost:8000` (development only) |
| `adk api_server {agent_dir}` | HTTP API for the agent |

## 1. Environment

```bash
uv venv --python python3.11 .venv
source .venv/bin/activate
uv pip install google-adk
```

`pip install google-adk` in a `python -m venv` works too. ADK requires Python
3.10 or newer.

## 2. API keys

Put a `.env` file in the **agent directory**, not its parent — the loader looks
beside `agent.py`. Do not commit it.

Google AI Studio (get a key at https://aistudio.google.com/app/apikey):

```bash
GOOGLE_GENAI_USE_ENTERPRISE=FALSE
GOOGLE_API_KEY=YOUR_API_KEY
```

Vertex AI, after `gcloud auth application-default login`:

```bash
GOOGLE_GENAI_USE_ENTERPRISE=TRUE
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

Vertex AI express mode swaps the project/location pair for an API key:

```bash
GOOGLE_GENAI_USE_ENTERPRISE=TRUE
GOOGLE_API_KEY=YOUR_EXPRESS_MODE_KEY
```

`GOOGLE_GENAI_USE_VERTEXAI` is the old name for the same switch. It still works
but emits a `DeprecationWarning`; `GOOGLE_GENAI_USE_ENTERPRISE` wins when both
are set.

## 3. Directory layout

The CLI discovers agents by convention:

```text
my_agent/
├── __init__.py    # from . import agent
├── agent.py       # defines root_agent (and optionally app)
└── .env
```

`__init__.py` must re-export the module, or the agent will not appear in
`adk web`:

```python
from . import agent
```

## 4. A basic LLM agent

`LlmAgent` (aliased as `Agent`) binds a model, an instruction, and tools.

```python
from google.adk import Agent


def get_weather(city: str) -> dict:
  """Returns the current weather for a specified city."""
  return {
      'status': 'success',
      'city': city,
      'weather': 'sunny',
      'temperature': '72F',
  }


root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='An assistant that reports the weather.',
    instruction=(
        'You are a helpful assistant. Use get_weather to look up the'
        ' weather in any city. Be concise.'
    ),
    tools=[get_weather],
)
```

| Field | Purpose |
|---|---|
| `model` | Model id, e.g. `'gemini-2.5-flash'`, `'gemini-2.5-pro'` |
| `instruction` | System prompt; `{var}` placeholders resolve from session state |
| `tools` | Python callables; name, docstring, and type hints become the tool schema |
| `description` | How a parent agent decides to route to this one |
| `output_key` | Session-state key to store the agent's final text under |

A tool function needs a docstring and type hints on every parameter — the LLM
sees only those. Return a `dict` or a `str`.

## 5. Running it programmatically

```python
import asyncio

from google.adk.runners import InMemoryRunner
from google.genai import types

from my_agent import agent


async def main():
  runner = InMemoryRunner(app_name='my_app', agent=agent.root_agent)
  session = await runner.session_service.create_session(
      app_name='my_app', user_id='user1'
  )
  message = types.Content(
      role='user',
      parts=[types.Part.from_text(text="What's the weather in Paris?")],
  )
  async for event in runner.run_async(
      user_id='user1', session_id=session.id, new_message=message
  ):
    if event.content and event.content.parts:
      text = event.content.parts[0].text
      if text:
        print(f'{event.author}: {text}')


asyncio.run(main())
```

## 6. From one agent to a workflow

A `Workflow` replaces "one LLM decides everything" with an explicit graph. The
smallest one has a single edge from `START`:

```python
from google.adk import Workflow


def greet(node_input: str) -> str:
  return f'Hello! You said: {node_input}'


root_agent = Workflow(name='my_workflow', edges=[('START', greet)])
```

### Sequential pipeline of LLM agents

> **Deprecated.** `SequentialAgent` is deprecated in favour of `Workflow`.
> Prefer the explicit edge list above for new code; this form is documented
> because existing agents still use it.

`SequentialAgent` generates `START -> a -> b -> c` for you. Each agent's
`output_key` publishes to session state, and the next agent reads it through an
instruction placeholder.

```python
from google.adk.agents import LlmAgent, SequentialAgent

writer = LlmAgent(
    name='CodeWriterAgent',
    model='gemini-2.5-flash',
    instruction=(
        'Write Python code that fulfills the user request. Output only the'
        ' code block.'
    ),
    description='Writes initial Python code from a specification.',
    output_key='generated_code',
)

reviewer = LlmAgent(
    name='CodeReviewerAgent',
    model='gemini-2.5-flash',
    instruction=(
        'Review this code and reply with a bulleted list of issues, or "No'
        ' major issues found." if it is clean:\n\n{generated_code}'
    ),
    description='Reviews code and provides feedback.',
    output_key='review_comments',
)

refactorer = LlmAgent(
    name='CodeRefactorerAgent',
    model='gemini-2.5-flash',
    instruction=(
        'Improve this code:\n\n{generated_code}\n\nAddressing these'
        ' comments:\n\n{review_comments}\n\nOutput only the final code'
        ' block.'
    ),
    description='Refactors code based on review comments.',
    output_key='refactored_code',
)

root_agent = SequentialAgent(
    name='CodePipelineAgent',
    sub_agents=[writer, reviewer, refactorer],
    description='Writes, reviews, and refactors Python code.',
)
```

### Graph with conditional routing

A node returns `Event(route=...)` and the edge dict picks the branch.

```python
from google.adk import Event, Workflow


def parse_input(node_input: str) -> dict:
  return {'text': node_input, 'word_count': len(node_input.split())}


def classify(node_input: dict):
  route = 'long' if node_input['word_count'] > 10 else 'short'
  return Event(output=node_input, route=route)


def handle_short(node_input: dict) -> str:
  return f"Short ({node_input['word_count']} words): {node_input['text']}"


def handle_long(node_input: dict) -> str:
  return f"Long ({node_input['word_count']} words): {node_input['text'][:50]}..."


root_agent = Workflow(
    name='classifier_workflow',
    input_schema=str,
    edges=[
        ('START', parse_input, classify),
        (classify, {'short': handle_short, 'long': handle_long}),
    ],
)
```

### Parallel list processing

`parallel_worker=True` makes a node run once per item of a list input and
return a list of results.

```python
from google.adk import Workflow
from google.adk.workflow import node


def split_input(node_input: str) -> list:
  return [item.strip() for item in node_input.split(',')]


@node(parallel_worker=True)
def process_item(node_input: str) -> dict:
  return {'item': node_input, 'upper': node_input.upper()}


def format_results(node_input: list) -> str:
  return '\n'.join(f"- {r['item']} -> {r['upper']}" for r in node_input)


root_agent = Workflow(
    name='parallel_processor',
    input_schema=str,
    edges=[('START', split_input, process_item, format_results)],
)
```

### Mixing function nodes and an LLM agent

```python
from google.adk import Workflow
from google.adk.agents import LlmAgent


def get_weather(city: str) -> dict:
  """Get the current weather for a city."""
  return {'city': city, 'temp': '72F', 'condition': 'sunny'}


def extract_city(node_input: str) -> str:
  return node_input.strip()


weather_agent = LlmAgent(
    name='weather_reporter',
    model='gemini-2.5-flash',
    instruction='Use get_weather, then give a natural-language report.',
    tools=[get_weather],
)


def sign_off(node_input: str) -> str:
  return f'{node_input}\n\nHave a great day!'


root_agent = Workflow(
    name='weather_workflow',
    input_schema=str,
    edges=[('START', extract_city, weather_agent, sign_off)],
)
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `No module named 'google.adk'` | Virtual environment not activated, or `google-adk` not installed in it |
| Agent missing from the `adk web` dropdown | `__init__.py` lacks `from . import agent`, or `agent.py` defines no `root_agent` |
| API key errors | `.env` sits in the parent directory instead of the agent directory |
| Model not found | Typo in the model id; non-Google models (Anthropic, LiteLLM) need extra dependencies |
