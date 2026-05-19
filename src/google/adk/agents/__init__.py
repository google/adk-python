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

import importlib
from typing import TYPE_CHECKING

from .base_agent import BaseAgent
from .context import Context
from .invocation_context import InvocationContext
from .live_request_queue import LiveRequest
from .live_request_queue import LiveRequestQueue
from .llm_agent import Agent
from .llm_agent import LlmAgent
from .loop_agent import LoopAgent
from .parallel_agent import ParallelAgent
from .run_config import RunConfig
from .sequential_agent import SequentialAgent

if TYPE_CHECKING:
  from .mcp_instruction_provider import McpInstructionProvider

__all__ = [
    'Agent',
    'BaseAgent',
    'Context',
    'LlmAgent',
    'LoopAgent',
    'McpInstructionProvider',
    'ParallelAgent',
    'SequentialAgent',
    'InvocationContext',
    'LiveRequest',
    'LiveRequestQueue',
    'RunConfig',
]


def __getattr__(name: str):
  if name == 'McpInstructionProvider':
    try:
      module = importlib.import_module(f'{__name__}.mcp_instruction_provider')
    except ImportError as e:
      raise ImportError(
          '`McpInstructionProvider` requires the `mcp` package.'
          ' Install with: pip install google-adk[extensions]'
      ) from e
    return module.McpInstructionProvider
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
