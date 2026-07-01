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

from google.adk.utils.content_utils import SKIP_THOUGHT_SIGNATURE_VALIDATOR
from google.genai import types


def test_skip_thought_signature_validator_wire_value():
  # The backend recognizes this exact byte string to bypass validation;
  # changing it would break every replayed synthetic part.
  assert SKIP_THOUGHT_SIGNATURE_VALIDATOR == b'skip_thought_signature_validator'


def test_skip_thought_signature_validator_assignable_to_part():
  part = types.Part(
      text='injected',
      thought_signature=SKIP_THOUGHT_SIGNATURE_VALIDATOR,
  )
  assert part.thought_signature == SKIP_THOUGHT_SIGNATURE_VALIDATOR
