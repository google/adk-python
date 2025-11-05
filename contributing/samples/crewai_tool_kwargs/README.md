# CrewAI Tool **kwargs Parameter Handling

This sample demonstrates how `CrewaiTool` correctly handles tools with
`**kwargs` parameters, which is a common pattern in CrewAI tools.

## What This Sample Demonstrates

### Key Feature: **kwargs Parameter Passing

CrewAI tools often accept arbitrary parameters via `**kwargs`:

```python
def _run(self, query: str, **kwargs) -> str:
    # Extra parameters are passed through kwargs
    category = kwargs.get('category')
    date_range = kwargs.get('date_range')
    limit = kwargs.get('limit')
```

The `CrewaiTool` wrapper detects this pattern and passes all parameters through
(except framework-managed ones like `self` and `tool_context`).

### Contrast with Regular Tools

For comparison, tools without `**kwargs` only accept explicitly declared
parameters:

```python
def _run(self, query: str, category: str) -> str:
```

## Prerequisites

### Required: CrewAI Tools (Python 3.10+)

```bash
pip install 'crewai-tools>=0.2.0'
```

### Required: API Key

```bash
export GOOGLE_API_KEY="your-api-key-here"
# OR
export GOOGLE_GENAI_API_KEY="your-api-key-here"
```

## Running the Sample

### Option 1: Run the Happy Path Test

```bash
cd contributing/samples/crewai_tool_kwargs
python main.py
```

**Expected output:**
```
============================================================
CrewAI Tool **kwargs Parameter Test
============================================================

🧪 Test 1: Basic search (no extra parameters)
User: Search for Python tutorials
Agent: [Uses tool and returns results]

🧪 Test 2: Search with filters (**kwargs test)
User: Search for machine learning articles, filtered by...
Agent: [Uses tool with category, date_range, and limit parameters]

============================================================
✅ Happy path test completed successfully!
============================================================
```

## What Gets Tested

✅ **CrewAI tool integration** - Wrapping a CrewAI BaseTool with ADK
✅ **Basic parameters** - Required `query` parameter passes correctly
✅ ****kwargs passing** - Extra parameters (category, date_range, limit) pass
   through
✅ **End-to-end execution** - Tool executes and returns results to agent

## Code Structure

```
crewai_tool_kwargs/
├── __init__.py       # Module initialization
├── agent.py          # Agent with CrewAI tool
├── main.py           # Happy path test
└── README.md         # This file
```

### Key Files

**agent.py:**

- Defines `CustomSearchTool` (CrewAI BaseTool with **kwargs)
- Wraps it with `CrewaiTool`
- Creates agent with the wrapped tool

**main.py:**

- Test 1: Basic search (no extra params)
- Test 2: Search with filters (tests **kwargs)

## How It Works

1. **CrewAI Tool Definition** (`agent.py`):
   ```python
   class CustomSearchTool(BaseTool):
       def _run(self, query: str, **kwargs) -> str:
           # kwargs receives: category, date_range, limit, etc.
   ```

2. **ADK Wrapping** (`agent.py`):
   ```python
   adk_search_tool = CrewaiTool(
       crewai_search_tool,
       name="search_with_filters",
       description="..."
   )
   ```

3. **LLM Function Calling** (`main.py`):
   - LLM sees the tool in function calling format
   - LLM calls with: `{query: "...", category: "...", date_range: "...", limit: 10}`
   - CrewaiTool passes ALL parameters to `**kwargs`

4. **Tool Execution**:
   - `query` → positional parameter
   - `category`, `date_range`, `limit` → collected in `**kwargs`
   - Tool logic uses all parameters

## Troubleshooting

### ImportError: No module named 'crewai'

```bash
pip install 'crewai-tools>=0.2.0'
```

### Python Version Error

CrewAI requires Python 3.10+:

```bash
python --version  # Should be 3.10 or higher
```

### Missing API Key

```bash
export GOOGLE_API_KEY="your-key-here"
```

## Related

- Parent class: `FunctionTool` - Base class for all function-based tools
- Unit tests: `tests/unittests/tools/test_crewai_tool.py`
