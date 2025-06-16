# Multi-Agent Systems in ADK

This example demonstrates how to create a multi-agent system in ADK, where specialized agents collaborate to handle complex tasks, each focusing on their area of expertise.

## What is a Multi-Agent System?

A Multi-Agent System is an advanced pattern in the Agent Development Kit (ADK) that allows multiple specialized agents to work together to handle complex tasks. Each agent can focus on a specific domain or functionality, and they can collaborate through delegation and communication to solve problems that would be difficult for a single agent.

## Project Structure Requirements

For multi-agent systems to work properly with ADK, your project must follow a specific structure:

```
parent_folder/
├── root_agent_folder/           # Main agent package (e.g., "manager")
│   ├── __init__.py              # Must import agent.py
│   ├── agent.py                 # Must define root_agent
│   ├── .env                     # Environment variables
│   └── sub_agents/              # Directory for all sub-agents
│       ├── __init__.py          # Empty or imports sub-agents
│       ├── agent_1_folder/      # Sub-agent package
│       │   ├── __init__.py      # Must import agent.py
│       │   └── agent.py         # Must define an agent variable
│       ├── agent_2_folder/
│       │   ├── __init__.py
│       │   └── agent.py
│       └── ...
```

### Essential Structure Components:

1. **Root Agent Package**
   - Must have the standard agent structure (like in the basic agent example)
   - The `agent.py` file must define a `root_agent` variable

2. **Sub-agents Directory**
   - Typically organized as a directory called `sub_agents` inside the root agent folder
   - Each sub-agent should be in its own directory following the same structure as regular agents

3. **Importing Sub-agents**
   - Root agent must import sub-agents to use them:
   ```python
   from .sub_agents.funny_nerd.agent import funny_nerd
   from .sub_agents.stock_analyst.agent import stock_analyst
   ```

4. **Command Location**
   - Always run `adk web` from the parent directory (`6-multi-agent`), not from inside any agent directory

This structure ensures that ADK can discover and correctly load all agents in the hierarchy.

## Multi-Agent Architecture Options

ADK offers two primary approaches to building multi-agent systems:

### 1. Sub-Agent Delegation Model

Using the `sub_agents` parameter, the root agent can fully delegate tasks to specialized agents:

```python
root_agent = Agent(
    name="manager",
    model="gemini-2.0-flash",
    description="Manager agent",
    instruction="You are a manager agent that delegates tasks to specialized agents...",
    sub_agents=[stock_analyst, funny_nerd],
)
```

**Characteristics:**
- Complete delegation - sub-agent takes over the entire response
- The sub-agent decision is final and takes control of the conversation
- Root agent acts as a "router" determining which specialist should handle the query

### 2. Agent-as-a-Tool Model

Using the `AgentTool` wrapper, agents can be used as tools by other agents:

```python
from google.adk.tools.agent_tool import AgentTool

root_agent = Agent(
    name="manager",
    model="gemini-2.0-flash",
    description="Manager agent",
    instruction="You are a manager agent that uses specialized agents as tools...",
    tools=[
        AgentTool(news_analyst),
        get_current_time,
    ],
)
```

**Characteristics:**
- Sub-agent returns results to the root agent
- Root agent maintains control and can incorporate the sub-agent's response into its own
- Multiple tool calls can be made to different agent tools in a single response
- Gives the root agent more flexibility in how it uses the results

## Limitations When Using Multi-Agents

### Sub-agent Restrictions

**Built-in tools cannot be used within a sub-agent.**

For example, this approach using built-in tools within sub-agents is **not** currently supported:

```python
search_agent = Agent(
    model='gemini-2.0-flash',
    name='SearchAgent',
    instruction="You're a specialist in Google Search",
    tools=[google_search],  # Built-in tool
)
coding_agent = Agent(
    model='gemini-2.0-flash',
    name='CodeAgent',
    instruction="You're a specialist in Code Execution",
    tools=[built_in_code_execution],  # Built-in tool
)
root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    sub_agents=[
        search_agent,  # NOT SUPPORTED
        coding_agent   # NOT SUPPORTED
    ],
)
```

### Workaround Using Agent Tools

To use multiple built-in tools or to combine built-in tools with other tools, you can use the `AgentTool` approach:

```python
from google.adk.tools import agent_tool

search_agent = Agent(
    model='gemini-2.0-flash',
    name='SearchAgent',
    instruction="You're a specialist in Google Search",
    tools=[google_search],
)
coding_agent = Agent(
    model='gemini-2.0-flash',
    name='CodeAgent',
    instruction="You're a specialist in Code Execution",
    tools=[built_in_code_execution],
)
root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    tools=[
        agent_tool.AgentTool(agent=search_agent), 
        agent_tool.AgentTool(agent=coding_agent)
    ],
)
```

This approach wraps agents as tools, allowing the root agent to delegate to specialized agents that each use a single built-in tool.

## Our Multi-Agent Example

This example implements a manager agent that works with three specialized agents:

1. **Stock Analyst** (Sub-agent): Provides financial information and stock market insights
2. **Funny Nerd** (Sub-agent): Creates nerdy jokes about technical topics
3. **News Analyst** (Agent Tool): Gives summaries of current technology news

The manager agent routes queries to the appropriate specialist based on the content of the user's request.

## Getting Started

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
   - Rename `.env.example` to `.env` in the manager folder
   - Add your Google API key to the `GOOGLE_API_KEY` variable in the `.env` file

## Running the Example

To run the multi-agent example:

1. Navigate to the 6-multi-agent directory containing your agent folders.

2. Start the interactive web UI:
```bash
adk web
```

3. Access the web UI by opening the URL shown in your terminal (typically http://localhost:8000)

4. Select the "manager" agent from the dropdown menu in the top-left corner of the UI

5. Start chatting with your agent in the textbox at the bottom of the screen

### Troubleshooting

If your multi-agent setup doesn't appear properly in the dropdown menu:
- Make sure you're running `adk web` from the parent directory (6-multi-agent)
- Verify that each agent's `__init__.py` properly imports its respective `agent.py`
- Check that the root agent properly imports all sub-agents

### Example Prompts to Try

- "Can you tell me about the stock market today?"
- "Tell me something funny about programming"
- "What's the latest tech news?"
- "What time is it right now?"

You can exit the conversation or stop the server by pressing `Ctrl+C` in your terminal.

## Additional Resources

- [ADK Multi-Agent Systems Documentation](https://google.github.io/adk-docs/agents/multi-agent-systems/)
- [Agent Tools Documentation](https://google.github.io/adk-docs/tools/function-tools/#3-agent-as-a-tool)
