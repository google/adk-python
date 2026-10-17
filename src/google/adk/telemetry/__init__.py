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
  from .context import ContentCapturingMode
  from .context import TelemetryConfig
  from .tracing import trace_call_llm
  from .tracing import trace_merged_tool_calls
  from .tracing import trace_send_data
  from .tracing import trace_tool_call
  from .tracing import tracer

_LAZY_MEMBERS: dict[str, str] = {
    'ContentCapturingMode': '.context',
    'TelemetryConfig': '.context',
    'trace_call_llm': '.tracing',
    'trace_merged_tool_calls': '.tracing',
    'trace_send_data': '.tracing',
    'trace_tool_call': '.tracing',
    'tracer': '.tracing',
}
__all__ = [
    'ContentCapturingMode',
    'TelemetryConfig',
    'trace_call_llm',
    'trace_merged_tool_calls',
    'trace_send_data',
    'trace_tool_call',
    'tracer',
]

__getattr__, __dir__ = _lazy.accessors(globals(), _LAZY_MEMBERS)
