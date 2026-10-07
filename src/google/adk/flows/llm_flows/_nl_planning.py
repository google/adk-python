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

"""Backward compatibility module for NL planning.

NL planning processing has moved to
``google.adk.flows.llm_flows.extensions._planning``. This module re-exports
all symbols for backward compatibility.
"""

from __future__ import annotations

from .extensions._planning import _get_planner as _get_planner
from .extensions._planning import _NlPlanningRequestProcessor as _NlPlanningRequestProcessor
from .extensions._planning import _NlPlanningResponse as _NlPlanningResponse
from .extensions._planning import _remove_thought_from_request as _remove_thought_from_request
from .extensions._planning import request_processor as request_processor
from .extensions._planning import response_processor as response_processor

__all__ = [
    '_NlPlanningRequestProcessor',
    '_NlPlanningResponse',
    '_get_planner',
    '_remove_thought_from_request',
    'request_processor',
    'response_processor',
]
