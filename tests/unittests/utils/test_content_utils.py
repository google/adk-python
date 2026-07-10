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
from google.adk.utils.content_utils import to_user_content
from google.genai import types
from pydantic import BaseModel


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


def test_to_user_content_str_input_becomes_user_text():
  content = to_user_content('hello')
  assert content.role == 'user'
  assert content.parts[0].text == 'hello'


def test_to_user_content_input_is_normalized_to_user_role():
  original = types.Content(role='model', parts=[types.Part(text='hi')])
  content = to_user_content(original)
  assert content.role == 'user'
  assert content.parts[0].text == 'hi'


def test_to_user_content_basemodel_input_is_json():
  class _M(BaseModel):
    a: int

  content = to_user_content(_M(a=1))
  assert content.role == 'user'
  assert '"a":1' in content.parts[0].text.replace(' ', '')


def test_to_user_content_dict_input_is_json():
  content = to_user_content({'a': 1})
  assert content.role == 'user'
  assert content.parts[0].text.replace(' ', '') == '{"a":1}'


def test_to_user_content_other_input_is_str():
  content = to_user_content(42)
  assert content.role == 'user'
  assert content.parts[0].text == '42'
