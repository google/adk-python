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

"""Serialization of content values into OTel-friendly attribute values."""

from __future__ import annotations

from google.adk.telemetry._serialization import serialize_content
from google.genai import types


def test_serialize_content_none_is_preserved():
  """``None`` must survive as ``None``; OTel treats it as an absent value,

  whereas a stringified ``'None'`` would be recorded as real content.
  """
  assert serialize_content(None) is None


def test_serialize_content_string_is_returned_unchanged():
  """A bare string is already an OTel value, so it must not be re-encoded

  into a JSON string literal (which would add surrounding quotes).
  """
  assert serialize_content('hello') == 'hello'


def test_serialize_content_pydantic_model_becomes_a_mapping():
  """A genai model is dumped to a mapping so OTel sees structured content

  rather than a repr.
  """
  content = types.Content(role='user', parts=[types.Part(text='hello')])

  result = serialize_content(content)

  assert isinstance(result, dict)
  assert result['role'] == 'user'
  assert result['parts'][0]['text'] == 'hello'


def test_serialize_content_list_is_serialized_element_wise():
  """A list stays a list: each element is serialized by the same rules, so a

  mixed list keeps its strings as strings and its models as mappings.
  """
  result = serialize_content([types.Part(text='a'), 'b'])

  assert isinstance(result, list)
  assert len(result) == 2
  assert isinstance(result[0], dict) and result[0]['text'] == 'a'
  assert result[1] == 'b'


def test_serialize_content_nested_list_recurses():
  """Recursion is depth-unbounded, not one level deep."""
  result = serialize_content([[types.Part(text='deep')]])

  assert isinstance(result, list) and isinstance(result[0], list)
  assert result[0][0]['text'] == 'deep'


def test_serialize_content_unknown_type_falls_back_to_json_string():
  """Anything outside the known shapes is JSON-encoded rather than dropped."""
  result = serialize_content({'k': 'v'})

  assert result == '{"k": "v"}'


def test_serialize_content_unserializable_value_yields_the_sentinel():
  """A value JSON cannot encode must degrade to the sentinel instead of

  raising out of the telemetry path.
  """
  assert serialize_content(object()) == '"<not serializable>"'
