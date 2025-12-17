# Interactions API Sample Agent

This sample agent demonstrates the Interactions API integration in ADK. The
Interactions API provides stateful conversation capabilities, allowing chained
interactions using `previous_interaction_id` instead of sending full
conversation history.

## Features Tested

1. **Basic Text Generation** - Simple conversation without tools
2. **Google Search Tool** - Web search using `GoogleSearchTool` with
   `bypass_multi_tools_limit=True`
3. **Multi-Turn Conversations** - Stateful interactions with context retention
   via `previous_interaction_id`
4. **Custom Function Tool** - Weather lookup using `get_current_weather`

## Important: Tool Compatibility

The Interactions API does **NOT** support mixing custom function calling tools
with built-in tools (like `google_search`) in the same agent. To work around
this limitation:

```python
# Use bypass_multi_tools_limit=True to convert google_search to a function tool
GoogleSearchTool(bypass_multi_tools_limit=True)
```

This converts the built-in `google_search` to a function calling tool (via
`GoogleSearchAgentTool`), which allows it to work alongside custom function
tools.

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
python -m interactions_api.main
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
├── test_interactions_curl.sh      # cURL-based API tests
├── test_interactions_direct.py    # Direct API tests
└── README.md                      # This file
```

## Agent Configuration

```python
from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.google_search_tool import GoogleSearchTool

root_agent = Agent(
    model=Gemini(
        model="gemini-2.5-flash",
        use_interactions_api=True,  # Enable Interactions API
    ),
    name="interactions_test_agent",
    tools=[
        GoogleSearchTool(bypass_multi_tools_limit=True),  # Converted to function tool
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
   [Tool Call] google_search_agent({'request': 'capital of France'})
   [Tool Result] google_search_agent: {'result': 'The capital of France is Paris...'}
<< Agent: The capital of France is Paris.
   [Interaction ID: v1_def456...]
PASSED: Google search tool works

============================================================
TEST 3: Multi-Turn Conversation (Stateful)
============================================================

>> User: Remember the number 42.
<< Agent: I'll remember that number - 42.
   [Interaction ID: v1_ghi789...]

>> User: What number did I ask you to remember?
<< Agent: You asked me to remember the number 42.
   [Interaction ID: v1_jkl012...]
PASSED: Multi-turn conversation works with context retention

============================================================
TEST 5: Custom Function Tool (get_current_weather)
============================================================

>> User: What's the weather like in Tokyo?
   [Tool Call] get_current_weather({'city': 'Tokyo'})
   [Tool Result] get_current_weather: {'city': 'Tokyo', 'temperature_f': 68, ...}
<< Agent: The weather in Tokyo is 68F and Partly Cloudy.
   [Interaction ID: v1_mno345...]
PASSED: Custom function tool works with bypass_multi_tools_limit

ALL TESTS PASSED (Interactions API)
```
