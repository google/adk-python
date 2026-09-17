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

"""Backward compatibility shim for _output_schema_processor.

Use google.adk.flows.llm_flows.prompt._schema instead.
"""

from __future__ import annotations

from .prompt._schema import _OutputSchemaRequestProcessor as _OutputSchemaRequestProcessor
from .prompt._schema import create_final_model_response_event as create_final_model_response_event
from .prompt._schema import get_structured_model_response as get_structured_model_response
from .prompt._schema import request_processor as request_processor
