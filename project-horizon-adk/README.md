# Project Horizon

## Exploring the Future of Interoperable, Multimodal AI Agent Systems

---
![Project Horizon Banner](assets/project-horizon.png)
---

## Overview

**Project Horizon explores the future of collaborative, multimodal AI agent systems.** It investigates how sophisticated, responsive, and interoperable agents can be built by integrating cutting-edge agent development frameworks and open communication protocols.

![High-Level Architecture Diagram](assets/high-level-architecture.png)
*(This diagram illustrates the general interaction patterns explored in Project Horizon, involving a Host Agent, Specialist Agents, and Tool Servers communicating via A2A and MCP.)*

This project serves as a testbed and a collection of reference implementations for:

*   Integrating diverse agent frameworks and tools like **Google's Agent Development Kit (ADK)** and the **Gemini Live API**.
*   Implementing standardized communication patterns between independent agents using protocols like **Agent2Agent (A2A)**.
*   Enabling agents to securely interact with external tools and data sources using protocols like the **Model Context Protocol (MCP)**.
*   Exploring advanced agent architectures, design patterns, and evaluation techniques.

**Project Horizon aims to capture and demonstrate the future of agent interactions, providing technical implementations, architectural blueprints, and practical design patterns for building the next generation of AI applications.**

## Core Technologies

This project showcases and explores the integration of several key technologies:

*   **[Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/)**: For building user-facing agents, orchestrating tasks, and managing multimodal communication.
*   **[Gemini Live API](https://ai.google.dev/gemini-api/docs/live)**: Powers real-time, low-latency, bidirectional voice (and potentially video) interactions within ADK.
*   **[Agent2Agent Protocol (A2A)](https://google.github.io/A2A/)**: An open standard for communication and task delegation between independent AI agents.
    *   This repository includes examples using  a custom A2A implementation (for historical reference). An implementation with the  official **[google-a2a-python SDK](https://github.com/google/a2a-python)** is planned.
*   **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/)**: An open standard allowing agents to securely interact with dedicated "Tool Servers" that provide specific functionalities or data.

## Example Applications

This repository contains a growing collection of example applications, each demonstrating different aspects of multi-agent systems. Each example is self-contained within the `examples/` directory and includes its own specific setup instructions.

### Available Examples:

1.  **Synchronous Stock Lookup (Custom A2A/MCP Implementation)**
    *   **Directory:** [`examples/stock_lookup_custom_a2a/`](./examples/stock_lookup_custom_a2a/)
    *   **Description:** This is the original proof-of-concept demonstrating a real-time stock price lookup using ADK Live, a custom A2A implementation for delegation, and MCP over stdio for tool execution. It is preserved with its original dependencies for reproducibility.
    *   **Technologies:** ADK Live, Custom A2A, Custom MCP Client, Finnhup API.

*More examples demonstrating various agent architectures and protocol features will be added over time.*

## Getting Started

**Each example within the `examples/` directory has its own detailed `README.md` with specific setup and execution instructions.**

## Future Plans / Roadmap

This project establishes a foundation for exploring advanced multi-agent systems. Future iterations and new examples aim to explore:

*   **A2A SDK:** Create an example with the officoal A2A Python SDK. 
*   **Asynchronous A2A/MCP:** Implement non-blocking calls, leveraging A2A streaming or push notifications, especially for long-running tool operations within Specialist Agent -> MCP Server interactions using the official `a2a-sdk`.
*   **Parallel A2A Delegation:** Modify Host Agents to delegate tasks to multiple Specialist Agents concurrently (e.g., fetch stock price AND company news simultaneously).
*   **MCP over HTTP/SSE:** Explore MCP Tool Servers that use HTTP/SSE transport (instead of just stdio), requiring security considerations like authentication middleware.
*   **Advanced State Management:** Utilize ADK's persistent SessionService options (e.g., Database, Vertex AI) and explore robust state sharing patterns between agents.
*   **Error Handling & Resilience:** Implement more comprehensive error handling, retries, and fallback mechanisms across all communication layers in various examples.
*   **Additional Specialist Agents & Tools:** Expand the system with more diverse agents (e.g., NewsAgent, PortfolioAgent, DocumentAnalysisAgent) and corresponding MCP Tool Servers or direct API tool integrations.
*   **ADK Evaluation:** Integrate ADK's evaluation framework (`adk eval`) to measure performance and accuracy of different agent setups.
*   **UI Enhancements:** Improve frontend UIs to better visualize multi-agent activity, tool calls, and potentially handle more complex inputs/outputs.
*   **Agent Framework Interoperability:** Investigate and demonstrate patterns for integrating agents built with other frameworks (e.g., LangChain, CrewAI) into an A2A/MCP ecosystem, potentially via wrappers.
*   **Security Considerations:** Further explore secure authentication and authorization mechanisms for A2A and MCP communication, especially for MCP over HTTP.

## 📜 License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file.

## 🤝 Contributing & Disclaimer

This is a personal project by [Heiko Hotz](https://github.com/heiko-hotz) to explore Gemini capabilities and agent interoperability. Suggestions and feedback are welcome via Issues or Pull Requests.

**This project is developed independently and does not reflect the views or efforts of Google.**