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
else:
    import importlib

    _LAZY_IMPORTS = {
        "Agent": ".llm_agent",
        "BaseAgent": ".base_agent",
        "Context": ".context",
        "InvocationContext": ".invocation_context",
        "LiveRequest": ".live_request_queue",
        "LiveRequestQueue": ".live_request_queue",
        "LlmAgent": ".llm_agent",
        "LoopAgent": ".loop_agent",
        "McpInstructionProvider": ".mcp_instruction_provider",
        "ParallelAgent": ".parallel_agent",
        "RunConfig": ".run_config",
        "SequentialAgent": ".sequential_agent",
    }

    def __getattr__(name: str):
        if name in _LAZY_IMPORTS:
            if name == "McpInstructionProvider":
                try:
                    module = importlib.import_module(
                        _LAZY_IMPORTS[name], __name__
                    )
                except ImportError as e:
                    raise ImportError(
                        "`McpInstructionProvider` requires the `mcp` package."
                        " Install with: pip install google-adk[extensions]"
                    ) from e
                return getattr(module, name)
            else:
                module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
                return getattr(module, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Agent",
    "BaseAgent",
    "Context",
    "InvocationContext",
    "LiveRequest",
    "LiveRequestQueue",
    "LlmAgent",
    "LoopAgent",
    "McpInstructionProvider",
    "ParallelAgent",
    "RunConfig",
    "SequentialAgent",
]
