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

"""Backward compatibility module for event compaction.

Event compaction has moved to ``google.adk.flows.llm_flows.context._compaction``.
This module re-exports all symbols for backward compatibility.
"""

from __future__ import annotations

from .context._compaction import CompactionRequestProcessor as CompactionRequestProcessor
from .context._compaction import request_processor as request_processor

__all__ = [
    'CompactionRequestProcessor',
    'request_processor',
]
