# Gemini CLI / Gemini Code Assist Context

This document provides context for the Gemini CLI and Gemini Code Assist to understand the project and assist with development.

## Project Overview

The Agent Development Kit (ADK) is an open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and is built for compatibility with other frameworks. ADK was designed to make agent development feel more like software development, to make it easier for developers to create, deploy, and orchestrate agentic architectures that range from simple tasks to complex workflows.

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
