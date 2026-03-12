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

if TYPE_CHECKING:
    from .base_agent import BaseAgent
    from .context import Context
    from .invocation_context import InvocationContext
    from .live_request_queue import LiveRequest
    from .live_request_queue import LiveRequestQueue
    from .llm_agent import Agent
    from .llm_agent import LlmAgent
    from .loop_agent import LoopAgent
    from .mcp_instruction_provider import McpInstructionProvider
    from .parallel_agent import ParallelAgent
    from .run_config import RunConfig
    from .sequential_agent import SequentialAgent

def __getattr__(name: str):
    if name == 'BaseAgent':
        from .base_agent import BaseAgent
        return BaseAgent
    if name == 'Context':
        from .context import Context
        return Context
    if name == 'InvocationContext':
        from .invocation_context import InvocationContext
        return InvocationContext
    if name == 'LiveRequest':
        from .live_request_queue import LiveRequest
        return LiveRequest
    if name == 'LiveRequestQueue':
        from .live_request_queue import LiveRequestQueue
        return LiveRequestQueue
    if name == 'Agent':
        from .llm_agent import Agent
        return Agent
    if name == 'LlmAgent':
        from .llm_agent import LlmAgent
        return LlmAgent
    if name == 'LoopAgent':
        from .loop_agent import LoopAgent
        return LoopAgent
    if name == 'McpInstructionProvider':
        from .mcp_instruction_provider import McpInstructionProvider
        return McpInstructionProvider
    if name == 'ParallelAgent':
        from .parallel_agent import ParallelAgent
        return ParallelAgent
    if name == 'RunConfig':
        from .run_config import RunConfig
        return RunConfig
    if name == 'SequentialAgent':
        from .sequential_agent import SequentialAgent
        return SequentialAgent
    raise AttributeError(f"module {__name__} has no attribute {name}")

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
