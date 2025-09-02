# Example GitHub Agent with Google ADK and MCP
This example project demonstrates how to build a conversational agent capable of interacting with GitHub using the Model Context Protocol (MCP). The agent is built with the Google Agent Development Kit (ADK) and connects to a local Large Language Model (LLM) via `litellm` and Ollama.

## Features

- **GitHub Integration**: Uses ADK's `MCPToolset` to provide the agent with tools to query GitHub APIs.
- **Local LLM**: Connects to an LLM running locally (e.g., Qwen3 via Ollama) using `litellm`.
- **Google ADK**: Leverages core ADK components like `Agent`, `InMemoryRunner`, and `MCPToolset`.
- **Dynamic Tool Usage**: The agent dynamically selects and uses appropriate MCP tools to respond to user questions about GitHub repositories.

## Prerequisites

Before starting, make sure you have the following installed:

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running.
- An LLM model that supports tool use. This example is configured for an improved `Qwen3` model version from [Unsloth](https://unsloth.ai/), which you can obtain with:
```sh
ollama pull hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL
```
- A [GitHub Personal Access Token](https://github.com/settings/tokens) with the `copilot` scope.

## Setup

1. **Install dependencies:**
    Open a terminal in the project's root directory and run:
```sh
pip install -r requirements.txt
```

2. **Configure environment variables:**
    Create a `.env` file in the project's root directory (next to `requirements.txt`) and add your GitHub token:
```env
# filepath: .env
GITHUB_TOKEN="YOUR_GITHUB_TOKEN_HERE"
```

## Running the Agent

1. Make sure your Ollama server is running and the LLM model is available.

2. Run the main script from the project's root:
```sh
python main.py
```

3. When prompted, enter your questions about GitHub. For example:
> "show me the files at the root of the repository google/generative-ai-python"

4. To end the session, type `exit` or press `Ctrl+C`.
