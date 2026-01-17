# Copyright 2026 Google LLC
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

import logging

from .base_code_executor import BaseCodeExecutor
from .built_in_code_executor import BuiltInCodeExecutor
from .code_executor_context import CodeExecutorContext
from .unsafe_local_code_executor import UnsafeLocalCodeExecutor

logger = logging.getLogger("google_adk." + __name__)

__all__ = [
    "BaseCodeExecutor",
    "BuiltInCodeExecutor",
    "CodeExecutorContext",
    "UnsafeLocalCodeExecutor",
    "VertexAiCodeExecutor",
    "ContainerCodeExecutor",
    "GkeCodeExecutor",
    "AgentEngineSandboxCodeExecutor",
    # CodingAgent components
    "AllowlistValidator",
    "CodingAgentCodeExecutor",
    "ToolCodeGenerator",
    "ToolExecutionServer",
]


def __getattr__(name: str):
    if name == "VertexAiCodeExecutor":
        try:
            from .vertex_ai_code_executor import VertexAiCodeExecutor

            return VertexAiCodeExecutor
        except ImportError as e:
            raise ImportError(
                "VertexAiCodeExecutor requires additional dependencies. "
                'Please install with: pip install "google-adk[extensions]"'
            ) from e
    elif name == "ContainerCodeExecutor":
        try:
            from .container_code_executor import ContainerCodeExecutor

            return ContainerCodeExecutor
        except ImportError as e:
            raise ImportError(
                "ContainerCodeExecutor requires additional dependencies. "
                'Please install with: pip install "google-adk[extensions]"'
            ) from e
    elif name == "GkeCodeExecutor":
        try:
            from .gke_code_executor import GkeCodeExecutor

            return GkeCodeExecutor
        except ImportError as e:
            raise ImportError(
                "GkeCodeExecutor requires additional dependencies. "
                'Please install with: pip install "google-adk[extensions]"'
            ) from e
    elif name == "AgentEngineSandboxCodeExecutor":
        try:
            from .agent_engine_sandbox_code_executor import (
                AgentEngineSandboxCodeExecutor,
            )

            return AgentEngineSandboxCodeExecutor
        except ImportError as e:
            raise ImportError(
                "AgentEngineSandboxCodeExecutor requires additional dependencies. "
                'Please install with: pip install "google-adk[extensions]"'
            ) from e
    elif name == "AllowlistValidator":
        try:
            from .allowlist_validator import AllowlistValidator

            return AllowlistValidator
        except ImportError as e:
            raise ImportError(
                "AllowlistValidator requires additional dependencies. "
                'Please install with: pip install "google-adk[extensions]"'
            ) from e
    elif name == "CodingAgentCodeExecutor":
        try:
            from .coding_agent_code_executor import CodingAgentCodeExecutor

            return CodingAgentCodeExecutor
        except ImportError as e:
            raise ImportError(
                "CodingAgentCodeExecutor requires additional dependencies. "
                'Please install with: pip install "google-adk[extensions]"'
            ) from e
    elif name == "ToolCodeGenerator":
        try:
            from .tool_code_generator import generate_full_code_with_stubs
            from .tool_code_generator import generate_runtime_header
            from .tool_code_generator import generate_system_prompt
            from .tool_code_generator import generate_tool_stubs

            # Return module-like object with functions
            class ToolCodeGenerator:
                generate_full_code_with_stubs = staticmethod(
                    generate_full_code_with_stubs
                )
                generate_runtime_header = staticmethod(generate_runtime_header)
                generate_system_prompt = staticmethod(generate_system_prompt)
                generate_tool_stubs = staticmethod(generate_tool_stubs)

            return ToolCodeGenerator
        except ImportError as e:
            raise ImportError(
                "ToolCodeGenerator requires additional dependencies. "
                'Please install with: pip install "google-adk[extensions]"'
            ) from e
    elif name == "ToolExecutionServer":
        try:
            from .tool_execution_server import ToolExecutionServer

            return ToolExecutionServer
        except ImportError as e:
            raise ImportError(
                "ToolExecutionServer requires additional dependencies. "
                'Please install with: pip install "google-adk[extensions]"'
            ) from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
