# Getting Started: Creating ADK Agents

Step-by-step guide covering environment setup, basic LLM agents, and workflow agents.

## 📋 New Agent Checklist
Use this checklist when creating a new agent to ensure it follows convention:
- [ ] **Directory**: Is there a directory for the agent?
- [ ] **__init__.py**: Does it contain `from . import agent`?
- [ ] **agent.py**: Does it define `root_agent` or `app`?
- [ ] **.env**: Is there a `.env` file with the appropriate API keys? (Do not commit to git)

## 💡 Quick Reference (CLI Commands)
- **Create**: `adk create <agent_name>` (Scaffolds a new agent project)
- **Web UI**: `adk web <path_to_agent_dir>` (Starts dev server at localhost:8000)
- **Run CLI**: `adk run <path_to_agent_dir>` (Interactive or query mode)

## 1. Set Up the Environment

Create a virtual environment and install the ADK:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install the ADK package
pip install google-adk
```

Or with `uv`:

```bash
uv venv --python "python3.11" ".venv"
source .venv/bin/activate
uv pip install google-adk
```

## 2. Configure API Keys

### Google AI Studio (recommended for getting started)

Obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Create a `.env` file in the agent directory:

```
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=YOUR_API_KEY
```

### Vertex AI

For production use with Google Cloud:

```
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

Run `gcloud auth application-default login` to authenticate.

### Vertex AI Express Mode

Combines Vertex AI with API key authentication:

```
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_API_KEY=YOUR_EXPRESS_MODE_KEY
```

## 3. Agent Directory Structure

The ADK CLI discovers agents by directory convention. Each agent directory must have:

```
my_agent/
├── __init__.py    # Must import the agent module
├── agent.py       # Must define root_agent
└── .env           # API keys (not committed to git)
```

### __init__.py

```python
from . import agent
```

Or generate the project with the CLI:

```bash
adk create my_agent
```

## 4. Basic LLM Agent with Tools

Before building workflow agents, understand the basic LLM agent pattern. An `LlmAgent` (also aliased as `Agent`) connects an LLM to tools and instructions:

### agent.py

```python
from google.adk.agents.llm_agent import Agent

def get_weather(city: str) -> dict:
  """Returns the current weather for a specified city."""
  # In production, call a real weather API
  return {
      "status": "success",
      "city": city,
      "weather": "sunny",
      "temperature": "72F",
  }

def get_current_time(city: str) -> dict:
  """Returns the current time in a specified city."""
  import datetime
  return {
      "status": "success",
      "city": city,
      "time": datetime.datetime.now().strftime("%I:%M %p"),
  }

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="An assistant that provides weather and time information.",
    instruction="""You are a helpful assistant.
Use the get_weather tool to look up weather and
get_current_time to check the time in any city.
Always be friendly and concise.""",
    tools=[get_weather, get_current_time],
)
```

### Key concepts

- **`model`**: The LLM to use (e.g., `"gemini-2.5-flash"`, `"gemini-2.5-pro"`)
- **`instruction`**: System prompt guiding the agent's behavior
- **`tools`**: Python functions the LLM can call. The function name, docstring, and type hints are sent to the LLM as the tool schema
- **`description`**: Used when this agent is a sub-agent (for transfer routing)
- **`output_key`**: Store the agent's final text output in session state under this key

### Tool function conventions

- Use clear function names and docstrings — the LLM sees these
- Type-hint all parameters — they define the tool's input schema
- Return a `dict` or `str` — the return value becomes the tool response

## 5. Run the Agent

### Web UI (primary debugging tool)

```bash
adk web my_agent/
```

Open `http://localhost:8000`. Select the agent from the dropdown, type a message, and see events in the Events tab.

**Note**: `adk web` is for development only, not production.

### CLI mode

```bash
adk run my_agent/
```

### API server

```bash
adk api_server my_agent/
```

### Programmatic execution

```python
import asyncio
from google.adk.runners import InMemoryRunner
from google.genai import types

async def main():
  from my_agent import agent

  runner = InMemoryRunner(
      app_name="my_app",
      agent=agent.root_agent,
  )

  session = await runner.session_service.create_session(
      app_name="my_app", user_id="user1"
  )

  content = types.Content(
      role="user", parts=[types.Part.from_text(text="What's the weather in Paris?")]
  )

  async for event in runner.run_async(
      user_id="user1",
      session_id=session.id,
      new_message=content,
  ):
    if event.content and event.content.parts:
      if event.content.parts[0].text:
        print(f"{event.author}: {event.content.parts[0].text}")

asyncio.run(main())
```

## 6. From LLM Agent to Workflow Agent

A `Workflow` extends the basic agent pattern with graph-based execution. Instead of a single LLM deciding what to do, define explicit nodes and edges:

### agent.py — Minimal Workflow

```python
from google.adk.workflow import Workflow

def greet(node_input: str) -> str:
  return f"Hello! You said: {node_input}"

root_agent = Workflow(
    name="my_workflow",
    edges=[
        ('START', greet),
    ],
)
```

## 5. Sample: Sequential Pipeline with LLM Agents

A code write-review-refactor pipeline using `SequentialAgent`:

### agent.py

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent

code_writer_agent = LlmAgent(
    name="CodeWriterAgent",
    model="gemini-2.5-flash",
    instruction="""You are a Python Code Generator.
Based *only* on the user's request, write Python code that fulfills the requirement.
Output *only* the complete Python code block.
""",
    description="Writes initial Python code based on a specification.",
    output_key="generated_code",
)

code_reviewer_agent = LlmAgent(
    name="CodeReviewerAgent",
    model="gemini-2.5-flash",
    instruction="""You are an expert Python Code Reviewer.
Review the following code:

```python
{generated_code}
```

Provide feedback as a concise, bulleted list.
If the code is excellent, state: "No major issues found."
""",
    description="Reviews code and provides feedback.",
    output_key="review_comments",
)

code_refactorer_agent = LlmAgent(
    name="CodeRefactorerAgent",
    model="gemini-2.5-flash",
    instruction="""You are a Python Code Refactoring AI.
Improve the code based on the review comments.

**Original Code:**
```python
{generated_code}
```

**Review Comments:**
{review_comments}

If no issues found, return the original code unchanged.
Output *only* the final Python code block.
""",
    description="Refactors code based on review comments.",
    output_key="refactored_code",
)

root_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[code_writer_agent, code_reviewer_agent, code_refactorer_agent],
    description="Executes a sequence of code writing, reviewing, and refactoring.",
)
```

### Key patterns in this sample

- **`output_key`**: Each agent stores its output in session state, making it available to later agents
- **`{generated_code}`**: Instruction placeholders are resolved from session state at runtime
- **`SequentialAgent`**: Convenience wrapper that auto-generates `START -> agent1 -> agent2 -> agent3` edges

## 6. Sample: Graph Workflow with Functions and Routing

A data processing pipeline with conditional routing:

### agent.py

```python
from google.adk.workflow import Workflow
from google.adk.events.event import Event
from google.adk.agents.context import Context

def parse_input(node_input: str) -> dict:
  """Parse the user's input into a structured format."""
  words = node_input.strip().split()
  return {"text": node_input, "word_count": len(words)}

def classify(node_input: dict):
  """Route based on input length."""
  if node_input["word_count"] > 10:
    return Event(output=node_input, route="long")
  return Event(output=node_input, route="short")

def handle_short(node_input: dict) -> str:
  return f"Short input ({node_input['word_count']} words): {node_input['text']}"

def handle_long(node_input: dict) -> str:
  return f"Long input ({node_input['word_count']} words). Summary: {node_input['text'][:50]}..."

root_agent = Workflow(
    name="classifier_workflow",
    input_schema=str,
    edges=[
        ('START', parse_input),
        (parse_input, classify),
        (classify, handle_short, "short"),
        (classify, handle_long, "long"),
    ],
)
```

## 7. Sample: Parallel Processing

Process a list of items concurrently:

### agent.py

```python
from google.adk.workflow import Workflow
from google.adk.workflow import node

def split_input(node_input: str) -> list:
  """Split comma-separated input into a list."""
  return [item.strip() for item in node_input.split(",")]

@node(parallel_worker=True)
def process_item(node_input: str) -> dict:
  """Process a single item (runs in parallel for each list item)."""
  return {"item": node_input, "length": len(node_input), "upper": node_input.upper()}

def format_results(node_input: list) -> str:
  """Format the parallel results into a readable summary."""
  lines = [f"- {r['item']}: {r['length']} chars -> {r['upper']}" for r in node_input]
  return "Results:\n" + "\n".join(lines)

root_agent = Workflow(
    name="parallel_processor",
    input_schema=str,
    edges=[
        ('START', split_input),
        (split_input, process_item),
        (process_item, format_results),
    ],
)
```

## 8. Sample: Workflow with LLM Agent and Tools

Combine function nodes with an LLM agent that has tools:

### agent.py

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow import Workflow
from google.adk.agents.context import Context

def get_weather(city: str) -> dict:
  """Get the current weather for a city."""
  # In production, call a real API
  return {"city": city, "temp": "72F", "condition": "sunny"}

def extract_city(node_input: str) -> str:
  """Extract city name from user input."""
  # Simple extraction; in production, use NLP or LLM
  return node_input.strip()

weather_agent = LlmAgent(
    name="weather_reporter",
    model="gemini-2.5-flash",
    instruction="""You are a friendly weather reporter.
Use the get_weather tool to look up the weather, then give
a natural-language weather report for the city.""",
    tools=[get_weather],
)

def format_output(ctx: Context, node_input: str) -> str:
  """Add a friendly sign-off."""
  return f"{node_input}\n\nHave a great day!"

root_agent = Workflow(
    name="weather_workflow",
    input_schema=str,
    edges=[
        ('START', extract_city),
        (extract_city, weather_agent),
        (weather_agent, format_output),
    ],
)
```

## Troubleshooting

### "No module named 'google.adk'"
Ensure the virtual environment is activated and `google-adk` is installed.

### Agent not showing in `adk web`
Check that `__init__.py` contains `from . import agent` and `agent.py` defines `root_agent`.

### API key errors
Verify `.env` is in the agent directory (not the parent) and contains a valid `GOOGLE_API_KEY`.

### Model not found
Check the model name. Common models: `gemini-2.5-flash`, `gemini-2.5-pro`. The ADK also supports non-Google models (Anthropic, LiteLLM) with extra dependencies.
