# Gemini CLI / Gemini Code Assist Context

This document provides context for the Gemini CLI and Gemini Code Assist to understand the project and assist with development.

## Project Overview

The Agent Development Kit (ADK) is an open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and is built for compatibility with other frameworks. ADK was designed to make agent development feel more like software development, to make it easier for developers to create, deploy, and orchestrate agentic architectures that range from simple tasks to complex workflows.

## Project Architecture

Please refer to [ADK Project Overview and Architecture](https://github.com/google/adk-python/blob/main/contributing/adk_project_overview_and_architecture.md) for details.

### ADK Live (Bidi-streaming)

- ADK live feature can be accessed from runner.run_live(...) and corresponding FAST api endpoint.
- ADK live feature is built on top of [Gemini Live API](https://cloud.google.com/vertex-ai/generative-ai/docs/live-api). We integrate Gemini Live API through [GenAI SDK](https://github.com/googleapis/python-genai).
- ADK live related configs are in [run_config.py](https://github.com/google/adk-python/blob/main/src/google/adk/agents/run_config.py).
- ADK live under multi-agent scenario: we convert the audio into text. This text will be passed to next agent as context.
- Most logics are in [base_llm_flow.py](https://github.com/google/adk-python/blob/main/src/google/adk/flows/llm_flows/base_llm_flow.py) and [gemini_llm_connection.py](https://github.com/google/adk-python/blob/main/src/google/adk/models/gemini_llm_connection.py).
- Tests are in [tests/unittests/streaming](https://github.com/google/adk-python/tree/main/tests/unittests/streaming).

## ADK: Style Guides

### Python Style Guide

The project follows the Google Python Style Guide. Key conventions are enforced using `pylint` with the provided `pylintrc` configuration file. Here are some of the key style points:

*   **Indentation**: 2 spaces.
*   **Line Length**: Maximum 80 characters.
*   **Naming Conventions**:
    *   `function_and_variable_names`: `snake_case`
    *   `ClassNames`: `CamelCase`
    *   `CONSTANTS`: `UPPERCASE_SNAKE_CASE`
*   **Docstrings**: Required for all public modules, functions, classes, and methods.
*   **Imports**: Organized and sorted.
*   **Error Handling**: Specific exceptions should be caught, not general ones like `Exception`.

### Autoformat

We have autoformat.sh to help solve import organize and formatting issues.

```bash
# Run in open_source_workspace/
$ ./autoformat.sh
```

### In ADK source

Below styles applies to the ADK source code (under `src/` folder of the Github.
repo).

#### Use relative imports

```python
# DO
from ..agents.llm_agent import LlmAgent

# DON'T
from google.adk.agents.llm_agent import LlmAgent
```

#### Import from module, not from `__init__.py`

```python
# DO
from ..agents.llm_agent import LlmAgent

# DON'T
from ..agents  import LlmAgent # import from agents/__init__.py
```

#### Always do `from __future__ import annotations`

```python
# DO THIS, right after the open-source header.
from __future__ import annotations
```

Like below:

```python
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

# ... the rest of the file.
```

This allows us to forward-reference a class without quotes.

Check out go/pep563 for details.

### In ADK tests

#### Use absolute imports

In tests, we use `google.adk` same as how our users uses.

```python
# DO
from google.adk.agents.llm_agent import LlmAgent

# DON'T
from ..agents.llm_agent import LlmAgent
```

## ADK: Local testing

### Unit tests

Run below command:

```bash
$ pytest tests/unittests
```

## Docstring and comments

### Comments - Explaining the Why, Not the What
Philosophy: Well-written code should be largely self-documenting. Comments
 serve a different purpose: they should explain the complex algorithms,
 non-obvious business logic, or the rationale behind a particular implementation
 choice—the things the code cannot express on its own. Avoid comments that
 merely restate what the code does (e.g., # increment i above i += 1).

Style: Comments should be written as complete sentences. Block comments must
begin with a # followed by a single space.

## Versioning
ADK adherence to Semantic Versioning 2.0.0

Core Principle: The adk-python project strictly adheres to the Semantic
Versioning 2.0.0 specification. All release versions will follow the
MAJOR.MINOR.PATCH format.

### Breaking Change

A breaking change is any modification that introduces backward-incompatible
changes to the public API. In the context of the ADK, this means a change that
could force a developer using the framework to alter their existing code to
upgrade to the new version. The public API is not limited to just the Python
function and class signatures; it also encompasses data schemas for stored
information (like evaluation datasets), the command-line interface (CLI),
and the data format used for server communications.

### Public API Surface Definition

The "public API" of ADK is a broad contract that extends beyond its Python
function signatures. A breaking change in any of the following areas can
disrupt user workflows and the wider ecosystem of agents and tools built with
ADK. The analysis of the breaking changes introduced in v1.0.0 demonstrates the
expansive nature of this contract. For the purposes of versioning, the ADK
Public API Surface is defined as:

- All public classes, methods, and functions in the google.adk namespace.

- The names, required parameters, and expected behavior of all built-in Tools
 (e.g., google_search, BuiltInCodeExecutor).

- The structure and schema of persisted data, including Session data, Memory,
 and Evaluation datasets.

- The JSON request/response format of the ADK API server(FastAPI server)
 used by adk web, including field casing conventions.

- The command-line interface (CLI) commands, arguments, and flags (e.g., adk deploy).

- The expected file structure for agent definitions that are loaded by the
 framework (e.g., the agent.py convention).

#### Checklist for Breaking Changes:

The following changes are considered breaking and necessitate a MAJOR version
 bump.

- API Signature Change: Renaming, removing, or altering the required parameters
 of any public class, method, or function (e.g., the removal of the list_events
 method from BaseSessionService).

- Architectural Shift: A fundamental change to a core component's behavior
 (e.g., making all service methods async, which requires consumers to use await).

- Data Schema Change: A non-additive change to a persisted data schema that
 renders old data unreadable or invalid (e.g., the redesign of the
  MemoryService and evaluation dataset schemas).

- Tool Interface Change: Renaming a built-in tool, changing its required
 parameters, or altering its fundamental purpose (e.g., replacing
 BuiltInCodeExecutionTool with BuiltInCodeExecutor and moving it from the tools
 parameter to the code_executor parameter of an Agent).

- Configuration Change: Altering the required structure of configuration files
 or agent definition files that the framework loads (e.g., the simplification
 of the agent.py structure for MCPToolset).

- Wire Format Change: Modifying the data format for API server interactions
 (e.g., the switch from snake_case to camelCase for all JSON payloads).

- Dependency Removal: Removing support for a previously integrated third-party
 library or tool type.

## Commit Message Format

- Please use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)
format.
- If it's not a breaking change, please add #non-breaking tag. If it's a
breaking change, please add #breaking.