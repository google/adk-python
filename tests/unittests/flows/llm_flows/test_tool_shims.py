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

"""Unit tests for backward-compatibility re-export shims for tools components."""

from __future__ import annotations

from google.adk.flows.llm_flows import request_confirmation
from google.adk.flows.llm_flows.tools import _confirmation


def test_request_confirmation_shim() -> None:
  assert (
      request_confirmation.request_processor is _confirmation.request_processor
  )
  assert (
      request_confirmation._resolve_confirmation_targets
      is _confirmation._resolve_confirmation_targets
  )
  assert (
      request_confirmation._parse_tool_confirmation
      is _confirmation._parse_tool_confirmation
  )
  assert (
      request_confirmation._get_original_function_call_args
      is _confirmation._get_original_function_call_args
  )
