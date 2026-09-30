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

"""Backward-compatibility shim for instruction templates and state injection.

Core implementation has moved to
google.adk.flows.llm_flows.prompt._instructions_utils.
"""

from __future__ import annotations

from ..agents.readonly_context import ReadonlyContext as ReadonlyContext
from ..flows.llm_flows.prompt._instructions_utils import _is_valid_state_name as _is_valid_state_name
from ..flows.llm_flows.prompt._instructions_utils import _render_with_jinja2 as _render_with_jinja2
from ..flows.llm_flows.prompt._instructions_utils import _render_with_regex as _render_with_regex
from ..flows.llm_flows.prompt._instructions_utils import _TEMPLATE_VAR_PATTERN as _TEMPLATE_VAR_PATTERN
from ..flows.llm_flows.prompt._instructions_utils import inject_session_state as inject_session_state
from ..flows.llm_flows.prompt._instructions_utils import InstructionProvider as InstructionProvider

__all__ = [
    'InstructionProvider',
    'ReadonlyContext',
    'inject_session_state',
]
