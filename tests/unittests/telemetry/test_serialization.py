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

import json

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.telemetry._serialization import safe_json_serialize
from google.adk.telemetry._serialization import serialize_content
from google.adk.utils._credential_names import MAX_REDACT_DEPTH
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


_TOKEN = 'ya29.access-token-value'


def _oauth_credential() -> AuthCredential:
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(client_id='cid', access_token=_TOKEN),
  )


def test_safe_json_serialize_masks_a_credential_model():
  """A credential model must not reach a span attribute in the clear.

  `Field(repr=False)` keeps these values out of `repr()`, but `model_dump()`
  renders them, so the serializer has to mask them itself.
  """
  assert _TOKEN not in safe_json_serialize(_oauth_credential())


def test_safe_json_serialize_masks_a_credential_nested_in_tool_args():
  """Tool args and tool responses arrive as plain dicts, so the type is gone.

  `adk_request_credential` carries its `AuthConfig` under `auth_config`, which
  is why the field name has to be masked as well as the declared type.
  """
  args = {
      'function_call_id': 'fc-1',
      'auth_config': {'exchanged_auth_credential': {'access_token': _TOKEN}},
  }
  serialized = safe_json_serialize(args)
  assert _TOKEN not in serialized
  # The key survives, so a trace still shows a credential was present.
  assert 'auth_config' in serialized
  assert 'fc-1' in serialized


def test_safe_json_serialize_leaves_ordinary_content_unchanged():
  """Redaction must not alter a payload that holds no credential."""
  payload = {'city': 'Paris', 'temps': [1, 2.5, None, True], 'n': {'k': 'v'}}
  assert json.loads(safe_json_serialize(payload)) == payload


def test_safe_json_serialize_elides_past_its_depth_bound():
  """Hitting the walk's bound must elide the subtree, never emit it raw."""
  nested: dict[str, object] = {'access_token': _TOKEN}
  for _ in range(MAX_REDACT_DEPTH + 2):
    nested = {'wrap': nested}
  assert _TOKEN not in safe_json_serialize(nested)
