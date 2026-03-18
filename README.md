# Agent Development Kit (ADK) 2.0 Alpha

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

<h2 align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
</h2>
<h3 align="center">
  An open-source, code-first Python framework for building, evaluating, and deploying sophisticated AI agents with flexibility and control.
</h3>
<h3 align="center">
  Important Links:
  <a href="https://google.github.io/adk-docs/">Docs</a>,
  <a href="https://github.com/google/adk-samples">Samples</a> &
  <a href="https://github.com/google/adk-web">ADK Web</a>.
</h3>

______________________________________________________________________

> **⚠️ EARLY PREVIEW — BREAKING CHANGES FROM 1.x**
>
> This is an early alpha of ADK 2.0. It includes breaking changes to the
> agent API, event model, and session schema. **Do NOT use with ADK 1.x
> databases or sessions** — they are incompatible. APIs are subject to
> change without notice.
>
> Install only with an explicit version pin:
>
> ```bash
> pip install google-adk==2.0.0a1
> ```
>
> `pip install google-adk` will NOT install this version.

______________________________________________________________________

## 🔥 What's New in 2.0

- **Workflow Runtime**: A graph-based execution engine for composing
  deterministic execution flows for agentic apps, with support for routing,
  fan-out/fan-in, loops, retry, state management, dynamic nodes,
  human-in-the-loop, and nested workflows.

- **Task API**: Structured agent-to-agent delegation with multi-turn task
  mode, single-turn controlled output, mixed delegation patterns,
  human-in-the-loop, and task agents as workflow nodes.

## 🚀 Installation

```bash
pip install google-adk==2.0.0a1
```

**Requirements:** Python 3.11+.

## Quick Start

### Agent

```python
from google.adk import Agent

root_agent = Agent(
    name="greeting_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant. Greet the user warmly.",
)
```

### Workflow

```python
from google.adk import Agent, Workflow

generate_fruit_agent = Agent(
    name="generate_fruit_agent",
    instruction="Return the name of a random fruit. Return only the name.",
)

generate_benefit_agent = Agent(
    name="generate_benefit_agent",
    instruction="Tell me a health benefit about the specified fruit.",
)

root_agent = Workflow(
    name="root_agent",
    edges=[("START", generate_fruit_agent, generate_benefit_agent)],
)
```

### Run Locally

```bash
# Interactive CLI
adk run path/to/my_agent

# Web UI
adk web path/to/agents_dir
```

## 📚 Documentation

- **Getting Started**: https://google.github.io/adk-docs/
- **Samples**: See `contributing/workflow_samples/` and
  `contributing/task_samples/` for workflow and task API examples.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache 2.0 License — see the
[LICENSE](LICENSE) file for details.
