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

"""Backward compatibility module for agent_transfer.

Agent transfer functionality has moved to
``google.adk.flows.llm_flows.extensions._agent_transfer``. This module re-exports
all symbols for backward compatibility.
"""

from __future__ import annotations

from .extensions._agent_transfer import _AgentLike as _AgentLike
from .extensions._agent_transfer import _AgentTransferLlmRequestProcessor as _AgentTransferLlmRequestProcessor
from .extensions._agent_transfer import _build_target_agents_info as _build_target_agents_info
from .extensions._agent_transfer import _build_transfer_instruction_body as _build_transfer_instruction_body
from .extensions._agent_transfer import _build_transfer_instructions as _build_transfer_instructions
from .extensions._agent_transfer import _build_transfer_tool as _build_transfer_tool
from .extensions._agent_transfer import _get_incompatible_builtin_tool_error as _get_incompatible_builtin_tool_error
from .extensions._agent_transfer import _get_transfer_targets as _get_transfer_targets
from .extensions._agent_transfer import line_break as line_break
from .extensions._agent_transfer import request_processor as request_processor

__all__ = [
    '_AgentLike',
    '_AgentTransferLlmRequestProcessor',
    '_build_target_agents_info',
    '_build_transfer_instruction_body',
    '_build_transfer_instructions',
    '_build_transfer_tool',
    '_get_incompatible_builtin_tool_error',
    '_get_transfer_targets',
    'line_break',
    'request_processor',
]
