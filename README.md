# Agent Development Kit (ADK)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

<html>
    <h1 align="center">
      <img src="assets/agent-development-kit.png" width="256"/>
    </h1>
    <h3 align="center">
      An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.
    </h3>
    <h3 align="center">
      Important Links:
      <a href="https://google.github.io/adk-docs/">Docs</a> &
      <a href="https://github.com/google/adk-samples">Samples</a>.
    </h3>
</html>

The Agent Development Kit (ADK) empowers developers to build sophisticated AI agents integrated with Google Cloud services. Define agent behavior, orchestration, and tool usage directly in Python code for enhanced control, testability, versioning, and deployment flexibility—from local development to cloud environments.

---

## ✨ Key Features

*   **Code-First Development:** Define agents, tools, and orchestration logic for maximum control, testability, and versioning.
*   **Multi-Agent Architecture:** Build modular and scalable applications by composing multiple specialized agents in flexible hierarchies.
*   **Rich Tool Ecosystem:** Equip agents with diverse capabilities using pre-built tools, custom Python functions, OpenAPI specifications, or by integrating existing tools.
*   **Flexible Orchestration:** Define workflows using built-in agents for predictable pipelines, or leverage LLM-driven dynamic routing for adaptive behavior.
*   **Integrated Developer Experience:** Develop, test, and debug locally with a CLI and visual web UI.
*   **Built-in Evaluation:** Measure agent performance by evaluating response quality and step-by-step execution trajectory.
*   **Deployment Ready:** Containerize and deploy your agents anywhere – scale with Vertex AI Agent Engine, Cloud Run, or Docker.
*   **Native Streaming Support:** Build real-time, interactive experiences with native support for bidirectional streaming (text and audio).
*   **State, Memory & Artifacts:** Manage short-term conversational context, configure long-term persistent memory, and handle file uploads/downloads seamlessly.
*   **Extensibility:** Customize agent behavior deeply with callbacks and easily integrate third-party tools and services.

## 🚀 Installation

Install the ADK using `pip`:

```bash
pip install google-adk
```

## 🏁 Getting Started

Create your first agent (`my_agent/agent.py`):

```python
# my_agent/agent.py
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash-exp", # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

Create `my_agent/__init__.py`:

```python
# my_agent/__init__.py
from . import agent
```

Run it via the CLI (from the directory *containing* `my_agent`):

```bash
adk run my_agent
```

Or launch the Web UI from the folder that contains `my_agent` folder:

```bash
adk web
```

For a full step-by-step guide, check out the [quickstart](https://google.github.io/adk-docs/get-started/quickstart/) or [sample agents](https://github.com/google/adk-samples).

## 📚 Learn More

Explore the full documentation and resources for detailed guides:

*   **[Get Started](https://google.github.io/adk-docs/get-started/)**
*   **[Browse Sample Agents](https://github.com/google/adk-samples)**
*   **[Evaluate Agents](https://google.github.io/adk-docs/evaluate/)**
*   **[Deploy Agents](https://google.github.io/adk-docs/deploy/)**
*   **[API Reference](https://google.github.io/adk-docs/api-reference/)**

## 🤝 Contributing

We welcome contributions from the community! Whether it's bug reports, feature requests, documentation improvements, or code contributions, please see our [**Contributing Guidelines**](./CONTRIBUTING.md) to get started.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*