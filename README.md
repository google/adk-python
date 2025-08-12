# Agent Development Kit (ADK)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<html>
    <h2 align="center">
      <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
    </h2>
    <h3 align="center">
      An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.
    </h3>
    <h3 align="center">
      Important Links:
      <a href="https://google.github.io/adk-docs/">Docs</a>,
      <a href="https://github.com/google/adk-samples">Samples</a>,
      <a href="https://github.com/google/adk-java">Java ADK</a> &
      <a href="https://github.com/google/adk-web">ADK Web</a>.
    </h3>
</html>

Agent Development Kit (ADK) is a flexible and modular framework for developing and deploying AI agents. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and is built for compatibility with other frameworks. ADK was designed to make agent development feel more like software development, to make it easier for developers to create, deploy, and orchestrate agentic architectures that range from simple tasks to complex workflows.


---

## ✨ Key Features

- **Rich Tool Ecosystem**: Utilize pre-built tools, custom functions,
  OpenAPI specs, or integrate existing tools to give agents diverse
  capabilities, all for tight integration with the Google ecosystem.

- **Code-First Development**: Define agent logic, tools, and orchestration
  directly in Python for ultimate flexibility, testability, and versioning.

- **Modular Multi-Agent Systems**: Design scalable applications by composing
  multiple specialized agents into flexible hierarchies.

- **Deploy Anywhere**: Easily containerize and deploy agents on Cloud Run or
  scale seamlessly with Vertex AI Agent Engine.

## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

For remote agent-to-agent communication, ADK integrates with the
[A2A protocol](https://github.com/google-a2a/A2A/).
See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents)
for how they can work together.

## 🚀 Installation

### Stable Release (Recommended)

You can install the latest stable version of ADK using `pip`:

```bash
pip install google-adk
```

The release cadence is weekly.

This version is recommended for most users as it represents the most recent official release.

### Development Version
Bug fixes and new features are merged into the main branch on GitHub first. If you need access to changes that haven't been included in an official PyPI release yet, you can install directly from the main branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Note: The development version is built directly from the latest code commits. While it includes the newest fixes and features, it may also contain experimental changes or bugs not present in the stable release. Use it primarily for testing upcoming changes or accessing critical fixes before they are officially released.

## 📚 Documentation

Explore the full documentation for detailed guides on building, evaluating, and
deploying agents:

* **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlight

### Define a single agent:

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash", # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a multi-agent system:

Define a multi-agent system with coordinator agent, greeter agent, and task execution agent. Then ADK engine and the model will guide the agents works together to accomplish the task.

```python
from google.adk.agents import LlmAgent, BaseAgent

# Define individual agents
greeter = LlmAgent(name="greeter", model="gemini-2.0-flash", ...)
task_executor = LlmAgent(name="task_executor", model="gemini-2.0-flash", ...)

# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash",
    description="I coordinate greetings and tasks.",
    sub_agents=[ # Assign sub_agents here
        greeter,
        task_executor
    ]
)
```

### Development UI

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

###  Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions from the community! Whether it's bug reports, feature requests, documentation improvements, or code contributions, please see our
- [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
- Then if you want to contribute code, please read [Code Contributing Guidelines](./CONTRIBUTING.md) to get started.

## Vibe Coding

If you are to develop agent via vibe coding the [llms.txt](./llms.txt) and the [llms-full.txt](./llms-full.txt) can be used as context to LLM. While the former one is a summarized one and the later one has the full information in case your LLM has big enough context window.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*
