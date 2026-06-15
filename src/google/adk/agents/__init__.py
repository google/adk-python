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
from typing import Any
from typing import TYPE_CHECKING

from .base_agent import BaseAgent
from .base_agent_config import BaseAgentConfig
from .context import Context
from .invocation_context import InvocationContext
from .live_request_queue import LiveRequest
from .live_request_queue import LiveRequestQueue
from .llm_agent import Agent
from .llm_agent import LlmAgent
from .llm_agent_config import LlmAgentConfig
from .loop_agent import LoopAgent
from .loop_agent_config import LoopAgentConfig
from .parallel_agent import ParallelAgent
from .parallel_agent_config import ParallelAgentConfig
from .run_config import RunConfig
from .sequential_agent import SequentialAgent
from .sequential_agent_config import SequentialAgentConfig

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
    'BaseAgentConfig',
    'LlmAgentConfig',
    'LoopAgentConfig',
    'ParallelAgentConfig',
    'SequentialAgentConfig',
]


def __getattr__(name: str) -> Any:
  if name == 'McpInstructionProvider':
    module = importlib.import_module('.mcp_instruction_provider', __name__)
    attr = getattr(module, 'McpInstructionProvider')
    globals()[name] = attr
    return attr
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
  return list(globals().keys()) + __all__
