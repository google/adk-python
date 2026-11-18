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

"""Shared constants for ADK synthetic client and HITL function calls."""

from __future__ import annotations

AF_FUNCTION_CALL_ID_PREFIX = 'adk-'
REQUEST_EUC_FUNCTION_CALL_NAME = 'adk_request_credential'
REQUEST_CONFIRMATION_FUNCTION_CALL_NAME = 'adk_request_confirmation'
REQUEST_INPUT_FUNCTION_CALL_NAME = 'adk_request_input'

CLIENT_FUNCTION_CALL_NAMES: frozenset[str] = frozenset({
    REQUEST_EUC_FUNCTION_CALL_NAME,
    REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
    REQUEST_INPUT_FUNCTION_CALL_NAME,
})
