# Vertex AI Code Execution Agent Sample

This directory contains a sample agent that demonstrates how to use the
`VertexAiCodeExecutor` for data science tasks.

## Overview

The agent is designed to assist with data analysis in a Python environment. It
can execute Python code to perform tasks like data manipulation, analysis, and
visualization. This agent is particularly useful for tasks that require a secure
and sandboxed code execution environment with common data science libraries
pre-installed.

This sample is a direct counterpart to the
[code execution sample](../code_execution/) which uses the
`BuiltInCodeExecutor`. The key difference in this sample is the use of
`VertexAiCodeExecutor`.

## `VertexAiCodeExecutor`

The `VertexAiCodeExecutor` leverages the
[Vertex AI Code Interpreter Extension](https://cloud.google.com/vertex-ai/generative-ai/docs/extensions/code-interpreter)
to run Python code. This provides several advantages:

-   **Security**: Code is executed in a sandboxed environment on Google Cloud,
    isolating it from your local system.
-   **Pre-installed Libraries**: The environment comes with many common Python
    data science libraries pre-installed, such as `pandas`, `numpy`, and
    `matplotlib`.
-   **Stateful Execution**: The execution environment is stateful, meaning
    variables and data from one code execution are available in subsequent
    executions within the same session.

## How to use

### Prerequisites

Ensure you have configured your environment for using
[Google Cloud Vertex AI](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-cloud-vertex-ai).
You will need to have a Google Cloud Project with the Vertex AI API enabled.

### Running the agent

You can run this agent using the ADK CLI from the root of the repository.

To interact with the agent through the command line:

```bash
adk run contributing/samples/vertex_code_execution "Plot a sine wave from 0 to 10"
```

To use the web interface:

```bash
adk web contributing/samples/
```

Then select `vertex_code_execution` from the list of agents and interact with
it.
