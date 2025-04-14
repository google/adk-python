# Changelog

## 0.1.0

### Features
* Initial release of the Agent Development Kit (ADK).
* Multi-agent, agent-as-workflow, and custom agent support
* Tool authentication support
* Rich tool support, e.g. bult-in tools, google-cloud tools, thir-party tools, and MCP tools
* Rich callback support
* Built-in code execution capability
* Asynchronous runtime and execution
* Session, and memory support
* Built-in evaluation support
* Development UI that makes local devlopment easy
* Deploy to Google Cloud Run, Agent Engine
* (Experimental) Live(Bidi) auido/video agent support and Compositional Function Calling(CFC) support

## Unreleased

### Fixed
- Fixed infinite loop issue when using LiteLLM with Ollama/Gemma3 models
  - Added robust JSON parsing for malformed function call arguments
  - Implemented loop detection to prevent infinite repetition of function calls
  - Added graceful handling with informative user messages when loops are detected

## 2.0.1 - 2025-04-01
