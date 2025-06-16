# Tool Agent Example

## What is a Tool Agent?

A Tool Agent extends the basic ADK agent by incorporating tools that allow the agent to perform actions beyond just generating text responses. Tools enable agents to interact with external systems, retrieve information, and perform specific functions to accomplish tasks more effectively.

In this example, we demonstrate how to build an agent that can use built-in tools (like Google Search) and custom function tools to enhance its capabilities.

## Key Components

### 1. Built-in Tools
ADK provides several built-in tools that you can use with your agents:

- **Google Search**: Allows your agent to search the web for information
- **Code Execution**: Enables your agent to run code snippets
- **Vertex AI Search**: Lets your agent search through your own data

**Important Note**: Currently, for each root agent or single agent, only one built-in tool is supported. See the [ADK documentation](https://google.github.io/adk-docs/tools/built-in-tools/#use-built-in-tools-with-other-tools) for more details.

### 2. Custom Function Tools
You can create your own tools by defining Python functions. These custom tools extend your agent's capabilities to perform specific tasks.

#### Best Practices for Custom Function Tools:

- **Parameters**: Define your function parameters using standard JSON-serializable types (string, integer, list, dictionary)
- **No Default Values**: Default values are not currently supported in ADK
- **Return Type**: The preferred return type is a dictionary
  - If you don't return a dictionary, ADK will wrap it into a dictionary `{"result": ...}`
  - Best practice format: `{"status": "success", "error_message": None, "result": "..."}`
- **Docstrings**: The function's docstring serves as the tool's description and is sent to the LLM
  - Focus on clarity so the LLM understands how to use the tool effectively

## Limitations

When working with built-in tools in ADK, there are several important limitations to be aware of:

### Single Built-in Tool Restriction

**Currently, for each root agent or single agent, only one built-in tool is supported.**

For example, this approach using two built-in tools within a single agent is **not** currently supported:

```python
root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    tools=[built_in_code_execution, google_search],  # NOT SUPPORTED
)
```

### Built-in Tools vs. Custom Tools

**You cannot mix built-in tools with custom function tools in the same agent.**

For example, this approach is **not** currently supported:

```python
def get_current_time() -> dict:
    """Get the current time in the format YYYY-MM-DD HH:MM:SS"""
    return {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    tools=[google_search, get_current_time],  # NOT SUPPORTED
)
```

To use both types of tools, you would need to use the Agent Tool approach described in the Multi-Agent example.

## Implementation Example

### Understanding the Code

The agent.py file defines a tool agent that can use Google Search to find information on the web. The agent is configured with:

1. A name and description
2. The Gemini model to use
3. Instructions that tell the agent how to behave and what tools it can use
4. The tools it can access (in this case, google_search)

The file also includes a commented-out example of a custom function tool `get_current_time()` that could be uncommented to explore custom tool functionality.

### Getting Started

This example uses the same virtual environment created in the root directory. Make sure you have:

1. Activated the virtual environment from the root directory:
```bash
# macOS/Linux:
source ../.venv/bin/activate
# Windows CMD:
..\.venv\Scripts\activate.bat
# Windows PowerShell:
..\.venv\Scripts\Activate.ps1
```

2. Set up your API key:
   - Rename `.env.example` to `.env` in the tool_agent folder
   - Add your Google API key to the `GOOGLE_API_KEY` variable in the `.env` file

### Running the Example

To run the tool agent example:

1. Navigate to the 2-tool-agent directory containing your agent folder.

2. Start the interactive web UI:
```bash
adk web
```

3. Access the web UI by opening the URL shown in your terminal (typically http://localhost:8000)

4. Select the "tool_agent" from the dropdown menu in the top-left corner of the UI

5. Start chatting with your agent in the textbox at the bottom of the screen

The ADK CLI tool provides several options:

- **`adk web`**: Launches an interactive web UI for testing your agent with a chat interface
- **`adk run tool_agent`**: Runs your agent directly in the terminal
- **`adk api_server`**: Starts a FastAPI server to test API requests to your agent

### Example Prompts to Try

- "Search for recent news about artificial intelligence"
- "Find information about Google's Agent Development Kit"
- "What are the latest advancements in quantum computing?"

You can exit the conversation or stop the server by pressing `Ctrl+C` in your terminal.

## Additional Resources

- [Types of tools](https://google.github.io/adk-docs/tools/#full-example-tavily-search)
- [ADK Function Tools Documentation](https://google.github.io/adk-docs/tools/function-tools/)
- [ADK Built-in Tools Documentation](https://google.github.io/adk-docs/tools/built-in-tools/)
