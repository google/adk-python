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

from typing import TYPE_CHECKING

from ..utils import _lazy

if TYPE_CHECKING:
  from ._managed_agent import ManagedAgent
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
  from .mcp_instruction_provider import McpInstructionProvider
  from .parallel_agent import ParallelAgent
  from .parallel_agent_config import ParallelAgentConfig
  from .run_config import RunConfig
  from .sequential_agent import SequentialAgent
  from .sequential_agent_config import SequentialAgentConfig

_LAZY_MEMBERS: dict[str, str] = {
    'Agent': '.llm_agent',
    'BaseAgent': '.base_agent',
    'BaseAgentConfig': '.base_agent_config',
    'Context': '.context',
    'InvocationContext': '.invocation_context',
    'LiveRequest': '.live_request_queue',
    'LiveRequestQueue': '.live_request_queue',
    'LlmAgent': '.llm_agent',
    'LlmAgentConfig': '.llm_agent_config',
    'LoopAgent': '.loop_agent',
    'LoopAgentConfig': '.loop_agent_config',
    'ManagedAgent': '._managed_agent',
    'McpInstructionProvider': '.mcp_instruction_provider',
    'ParallelAgent': '.parallel_agent',
    'ParallelAgentConfig': '.parallel_agent_config',
    'RunConfig': '.run_config',
    'SequentialAgent': '.sequential_agent',
    'SequentialAgentConfig': '.sequential_agent_config',
}
__all__ = [
    'Agent',
    'BaseAgent',
    'Context',
    'LlmAgent',
    'LoopAgent',
    'ManagedAgent',
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

__getattr__, __dir__ = _lazy.accessors(globals(), _LAZY_MEMBERS)
