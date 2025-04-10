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

The Agent Development Kit (ADK) is designed for developers seeking fine-grained control and flexibility when building advanced AI agents that are tightly integrated with services in Google Cloud. It allows you to define agent behavior, orchestration, and tool use directly in code, enabling robust debugging, versioning, and deployment anywhere – from your laptop to the cloud.

---

## ✨ Key Features

* **Code-First Development:** Define agents, tools, and orchestration logic for maximum control, testability, and versioning.
* **Multi-Agent Architecture:** Build modular and scalable applications by composing multiple specialized agents in flexible hierarchies.
* **Rich Tool Ecosystem:** Equip agents with diverse capabilities using pre-built tools, custom Python functions, API specifications, or integrating existing tools.
* **Flexible Orchestration:** Define workflows using built-in agents for predictable pipelines, or leverage LLM-driven dynamic routing for adaptive behavior.
* **Integrated Developer Experience:** Develop, test, and debug locally with a CLI and visual web UI.
* **Built-in Evaluation:** Measure agent performance by evaluating response quality and step-by-step execution trajectory.
* **Deployment Ready:** Containerize and deploy your agents anywhere – scale with Vertex AI Agent Engine, Cloud Run, or Docker.
* **Native Streaming Support:** Build real-time, interactive experiences with native support for bidirectional streaming (text and audio).
* **State, Memory & Artifacts:** Manage short-term conversational context, configure long-term memory, and handle file uploads/downloads.
* **Extensibility:** Customize agent behavior deeply with callbacks and easily integrate third-party tools and services.

## 🚀 Installation

You can install the ADK using `pip`:

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

## 📚 Resources

Explore the full documentation for detailed guides on building, evaluating, and deploying agents:

*   **[Get Started](https://google.github.io/adk-docs/get-started/)**
*   **[Browse Sample Agents](https://github.com/google/adk-samples)**
*   **[Evaluate Agents](https://google.github.io/adk-docs/evaluate/)**
*   **[Deploy Agents](https://google.github.io/adk-docs/deploy/)**
*   **[API Reference](https://google.github.io/adk-docs/api-reference/)**

## Setting Up BigQuery Connectors

### Granting IAM Roles

To set up a BigQuery connector, you need to grant the necessary IAM roles to the service account. Follow these steps:

1. Go to the [IAM & Admin](https://console.cloud.google.com/iam-admin) page in the Google Cloud Console.
2. Select your project.
3. Click on the "Add" button to add a new member.
4. Enter the service account email address.
5. Assign the following roles:
   - `BigQuery Data Viewer`
   - `BigQuery User`
   - `BigQuery Job User`
6. Click "Save" to apply the changes.

### Example Code Snippets for BigQuery Operations

Here are some example code snippets for performing BigQuery operations using the ADK:

```python
from google.adk.agents import Agent
from google.adk.tools.google_api_tool.google_api_tool_sets import bigquery_tool_set

# Create an agent with BigQuery tools
bigquery_agent = Agent(
    name="bigquery_agent",
    model="gemini-2.0-flash-exp",
    instruction="You are a BigQuery assistant. Help the user with BigQuery operations.",
    description="An assistant for BigQuery operations.",
    tools=bigquery_tool_set.get_tools()
)

# Example usage of BigQuery tool
async def query_bigquery():
    query = "SELECT * FROM `project.dataset.table` LIMIT 10"
    result = await bigquery_agent.run_async(
        args={"query": query},
        tool_context=None
    )
    print(result)

# Run the example
import asyncio
asyncio.run(query_bigquery())
```

## 🤝 Contributing

We welcome contributions from the community! Whether it's bug reports, feature requests, documentation improvements, or code contributions, please see our [**Contributing Guidelines**](./CONTRIBUTING.md) to get started.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*
