# FastMCP Server-Side Sampling with ADK

This project demonstrates how to use server-side sampling with a `fastmcp` server connected to an ADK `MCPToolset`.

## Description

The setup consists of two main components:

1.  **ADK Agent (`agent.py`):** An `LlmAgent` is configured with an `MCPToolset`. This toolset connects to a local `fastmcp` server.
2.  **FastMCP Server (`mcp_server.py`):** A `fastmcp` server that exposes a single tool, `analyze_sentiment`. This server is configured to use its own LLM for sampling, independent of the ADK agent's LLM.

The flow is as follows:
1.  The user provides a text prompt to the ADK agent.
2.  The agent decides to use the `analyze_sentiment` tool from the `MCPToolset`.
3.  The tool call is sent to the `mcp_server.py`.
4.  Inside the `analyze_sentiment` tool, `ctx.sample()` is called. This delegates an LLM call to the `fastmcp` server's own sampling handler.
5.  The `mcp_server`'s LLM processes the prompt from `ctx.sample()` and returns the result to the server.
6.  The server processes the LLM response and returns the final sentiment to the agent.
7.  The agent displays the result to the user.

## Steps to Run

### Prerequisites

- Python 3.10+
- `google-adk` library installed.
- A configured OpenAI API key.

### 1. Set up the Environment

Clone the project and navigate to the directory. Make sure your `OPENAI_API_KEY` is available as an environment variable.

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install fastmcp openai litellm
```

### 3. Run the Example

Navigate to the `samples` directory and choose this ADK agent:

```bash
adk web .
```

The agent will automatically start the FastMCP server in the background.

- **Sample user prompt:** "What is the sentiment of 'I love building things with Python'?"
