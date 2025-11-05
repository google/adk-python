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

import logging

from .base_code_executor import BaseCodeExecutor
from .built_in_code_executor import BuiltInCodeExecutor
from .code_executor_context import CodeExecutorContext
from .unsafe_local_code_executor import UnsafeLocalCodeExecutor

logger = logging.getLogger('google_adk.' + __name__)

__all__ = [
    'BaseCodeExecutor',
    'BuiltInCodeExecutor',
    'CodeExecutorContext',
    'UnsafeLocalCodeExecutor',
    'VertexAiCodeExecutor',
    'ContainerCodeExecutor',
    'GkeCodeExecutor',
    'AgentEngineSandboxCodeExecutor',
]


def __getattr__(name: str):
  if name == 'VertexAiCodeExecutor':
    try:
      from .vertex_ai_code_executor import VertexAiCodeExecutor

      return VertexAiCodeExecutor
    except ImportError as e:
      raise ImportError(
          'VertexAiCodeExecutor requires additional dependencies. '
          'Please install with: pip install "google-adk[extensions]"'
      ) from e
  elif name == 'ContainerCodeExecutor':
    try:
      from .container_code_executor import ContainerCodeExecutor

      return ContainerCodeExecutor
    except ImportError as e:
      raise ImportError(
          'ContainerCodeExecutor requires additional dependencies. '
          'Please install with: pip install "google-adk[extensions]"'
      ) from e
  elif name == 'GkeCodeExecutor':
    try:
      from .gke_code_executor import GkeCodeExecutor

      return GkeCodeExecutor
    except ImportError as e:
      raise ImportError(
          'GkeCodeExecutor requires additional dependencies. '
          'Please install with: pip install "google-adk[extensions]"'
      ) from e
  elif name == 'AgentEngineSandboxCodeExecutor':
    try:
      from .agent_engine_sandbox_code_executor import AgentEngineSandboxCodeExecutor

      return AgentEngineSandboxCodeExecutor
    except ImportError as e:
      raise ImportError(
          'AgentEngineSandboxCodeExecutor requires additional dependencies. '
          'Please install with: pip install "google-adk[extensions]"'
      ) from e
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
