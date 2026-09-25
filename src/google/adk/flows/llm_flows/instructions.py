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

"""Backward compatibility shim for instructions.

Use google.adk.flows.llm_flows.prompt._instructions instead.
"""

from __future__ import annotations

from .prompt._instructions import _build_instructions as _build_instructions
from .prompt._instructions import _INSTRUCTION_BEGIN as _INSTRUCTION_BEGIN
from .prompt._instructions import _INSTRUCTION_END as _INSTRUCTION_END
from .prompt._instructions import _INSTRUCTION_PREAMBLE as _INSTRUCTION_PREAMBLE
from .prompt._instructions import _InstructionsLlmRequestProcessor as _InstructionsLlmRequestProcessor
from .prompt._instructions import _label_dynamic_instruction as _label_dynamic_instruction
from .prompt._instructions import _process_agent_instruction as _process_agent_instruction
from .prompt._instructions import request_processor as request_processor

__all__ = [
    '_INSTRUCTION_BEGIN',
    '_INSTRUCTION_END',
    '_INSTRUCTION_PREAMBLE',
    '_InstructionsLlmRequestProcessor',
    '_build_instructions',
    '_label_dynamic_instruction',
    '_process_agent_instruction',
    'request_processor',
]
