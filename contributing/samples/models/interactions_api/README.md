# Interactions API Sample Agent

This sample agent demonstrates the Interactions API integration in ADK. The
Interactions API provides stateful conversation capabilities, allowing chained
interactions using `previous_interaction_id` instead of sending full
conversation history.

## Features Tested

1. **Basic Text Generation** - Simple conversation without tools
1. **Google Search Tool** - Web search using the built-in `GoogleSearchTool`
1. **Multi-Turn Conversations** - Stateful interactions with context retention
   via `previous_interaction_id`
1. **Google Search Tool (Additional)** - A second web search query
1. **Custom Function Tool** - Weather lookup using `get_current_weather`
1. **PDF Summarization** - Summarizing a PDF attached as an inline data part

## Important: Tool Compatibility

This agent keeps `google_search` as a built-in tool next to the custom
`get_current_weather` function tool, so search results come back grounded
rather than as a function call.

If a backend rejects that combination, convert search to a function calling
tool instead:

```python
GoogleSearchTool(bypass_multi_tools_limit=True)
```

With the flag set, ADK wraps the built-in `google_search` in a
`GoogleSearchAgentTool`, and the model calls it as `google_search_agent`.

## How to Run

### Prerequisites

```bash
# From the adk-python root directory
uv sync --all-extras
source .venv/bin/activate

# Set up authentication (choose one):
# Option 1: Using Google Cloud credentials
export GOOGLE_CLOUD_PROJECT=your-project-id

# Option 2: Using API Key
export GOOGLE_API_KEY=your-api-key
```

### Running Tests

```bash
cd contributing/samples

# Run automated tests with Interactions API
python -m models.interactions_api.main
```

The `tests/` directory holds recorded request/response fixtures that can be
replayed offline, without calling a live model:

```bash
adk test models/interactions_api
```

## Key Differences: Interactions API vs Standard API

### Interactions API (`use_interactions_api=True`)

- Uses stateful interactions via `previous_interaction_id`
- Only sends current turn contents when chaining interactions
- Returns `interaction_id` in responses for chaining
- Ideal for long conversations with many turns
- Context caching is not used (state maintained via interaction chaining)

### Standard API (`use_interactions_api=False`)

- Uses stateless `generate_content` calls
- Sends full conversation history with each request
- No interaction IDs in responses
- Context caching can be used

## Code Structure

```
interactions_api/
├── __init__.py                    # Package initialization
├── agent.py                       # Agent definition with Interactions API
├── main.py                        # Test runner
├── tests/                         # Replay fixtures for agent_test_runner
└── README.md                      # This file
```

## Agent Configuration

```python
from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.google_search_tool import GoogleSearchTool

root_agent = Agent(
    model=Gemini(
        model="gemini-3.1-flash-lite",
        use_interactions_api=True,  # Enable Interactions API
    ),
    name="interactions_test_agent",
    tools=[
        GoogleSearchTool(),  # Built-in tool
        get_current_weather,  # Custom function tool
    ],
)
```

## Example Output

```
============================================================
TEST 1: Basic Text Generation
============================================================

>> User: Hello! What can you help me with?
<< Agent: Hello! I can help you with: 1) Search the web...
   [Interaction ID: v1_abc123...]
PASSED: Basic text generation works

============================================================
TEST 2: Function Calling (Google Search Tool)
============================================================

>> User: Search for the capital of France.
<< Agent: The capital of France is Paris.
   [Interaction ID: v1_def456...]
PASSED: Google search tool works

============================================================
TEST 3: Multi-Turn Conversation (Stateful)
============================================================

>> User: My favorite color is blue. Just acknowledge this, don't use any tools.
<< Agent: Got it - your favorite color is blue.
   [Interaction ID: v1_ghi789...]

>> User: What's the weather like in London?
   [Tool Call] get_current_weather({'city': 'London'})
   [Tool Result] get_current_weather: {'city': 'London', 'temperature_f': 59, ...}
<< Agent: London is 59F and Cloudy.
   [Interaction ID: v1_jkl012...]

>> User: What is my favorite color that I mentioned earlier in our conversation?
<< Agent: You said your favorite color is blue.
   [Interaction ID: v1_mno345...]
PASSED: Multi-turn conversation works with context retention

============================================================
TEST 5: Custom Function Tool (get_current_weather)
============================================================

>> User: What's the weather like in Tokyo?
   [Tool Call] get_current_weather({'city': 'Tokyo'})
   [Tool Result] get_current_weather: {'city': 'Tokyo', 'temperature_f': 68, ...}
<< Agent: The weather in Tokyo is 68F and Partly Cloudy.
   [Interaction ID: v1_pqr678...]
PASSED: Custom function tool works alongside the built-in search tool

ALL TESTS PASSED (Interactions API)
```
