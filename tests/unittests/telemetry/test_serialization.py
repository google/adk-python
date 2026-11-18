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

import collections
import dataclasses
import json
from typing import Annotated
from typing import Any

from google.adk.auth import auth_credential
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.telemetry._serialization import safe_json_serialize
from google.adk.telemetry._serialization import serialize_content
from google.adk.telemetry.tracing import _serialize_tool_response
from google.adk.utils._credential_names import is_credential_arg_name
from google.genai import types
import pydantic
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic.alias_generators import to_camel
import pytest


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


@pytest.mark.parametrize(
    'key, cred_key, token_key',
    [
        ('auth_config', 'exchanged_auth_credential', 'access_token'),
        ('authConfig', 'exchangedAuthCredential', 'accessToken'),
    ],
)
def test_safe_json_serialize_masks_a_credential_nested_in_tool_args(
    key, cred_key, token_key
):
  """Tool args and tool responses arrive as plain dicts, so the type is gone.

  `adk_request_credential` carries its `AuthConfig` under `auth_config` (or
  `authConfig` when dumped with `by_alias=True`), so the credential has to be
  recognized by its shape in both snake_case and camelCase.
  """
  args = {
      'function_call_id': 'fc-1',
      key: {cred_key: {token_key: _TOKEN}},
  }

  serialized = safe_json_serialize(args)

  assert _TOKEN not in serialized
  # The key survives, so a trace still shows a credential was present.
  assert key in serialized
  assert 'fc-1' in serialized


def _service_account() -> ServiceAccountCredential:
  return ServiceAccountCredential(
      type='service_account',
      project_id='pid',
      private_key_id='secret-key-id',
      private_key='secret-private-key',
      client_email='sa@example.com',
      client_id='cid',
      auth_uri='https://example.com/auth',
      token_uri='https://example.com/token',
      auth_provider_x509_cert_url='https://example.com/certs',
      client_x509_cert_url='https://example.com/cert',
      universe_domain='googleapis.com',
  )


def test_safe_json_serialize_masks_credential_fields_in_nested_base_model():
  """A credential nested in a model is masked before the dump drops its type."""

  class _ToolResult(BaseModel):
    result: AuthCredential
    sa: ServiceAccountCredential

  credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='cid',
          code_verifier='pkce-verifier-secret',
          auth_response_uri='https://example.com/cb?code=secret-uri-code',
          auth_code='secret-auth-code',
      ),
      http=HttpAuth(
          scheme='bearer',
          credentials=HttpCredentials(
              username='alice',
              password='secret-password',
              token=_TOKEN,
          ),
          additional_headers={'Authorization': 'Bearer secret-header'},
      ),
  )
  oauth_and_http_secrets = (
      'pkce-verifier-secret',
      'secret-uri-code',
      'secret-auth-code',
      'secret-password',
      _TOKEN,
      'secret-header',
  )

  serialized = safe_json_serialize(
      _ToolResult(result=credential, sa=_service_account())
  )
  for secret in oauth_and_http_secrets + (
      'secret-key-id',
      'secret-private-key',
  ):
    assert secret not in serialized
  assert json.loads(serialized) == {'result': '<redacted>', 'sa': '<redacted>'}

  # Already dumped, the type is gone and the `authType` shape is what matches.
  serialized = safe_json_serialize({
      'result': credential.model_dump(
          mode='json', exclude_none=True, by_alias=True
      )
  })
  for secret in oauth_and_http_secrets:
    assert secret not in serialized
  assert 'cid' in serialized
  assert 'bearer' in serialized


def test_credentials_in_list_and_dict_fields_are_masked():
  """A credential inside a list or dict field keeps its type until masked."""

  class _CredentialHolder(BaseModel):
    items: list[ServiceAccountCredential]
    by_name: dict[str, HttpCredentials]

  holder = _CredentialHolder(
      items=[_service_account()],
      by_name={'api': HttpCredentials(username='alice', password=_TOKEN)},
  )
  expected = {'items': ['<redacted>'], 'by_name': {'api': '<redacted>'}}

  assert json.loads(safe_json_serialize(holder)) == expected
  # Tool responses are flattened with `to_jsonable_python` before
  # serialization, which must not drop the credential type either.
  assert json.loads(_serialize_tool_response({'result': holder})) == {
      'result': expected
  }
  assert json.loads(_serialize_tool_response({'sa': _service_account()})) == {
      'sa': '<redacted>'
  }


@pytest.mark.parametrize(
    'serialize', [safe_json_serialize, _serialize_tool_response]
)
def test_credential_fields_dumped_by_alias_are_masked(serialize):
  """A credential field is masked when the dump keys it by its alias."""

  class _AliasedResult(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, serialize_by_alias=True)
    service_account: ServiceAccountCredential
    other: ServiceAccountCredential = Field(serialization_alias='myCred')

  result = _AliasedResult(
      serviceAccount=_service_account(), other=_service_account()
  )

  assert json.loads(serialize(result)) == {
      'serviceAccount': '<redacted>',
      'myCred': '<redacted>',
  }


@pytest.mark.parametrize(
    'serialize, expected',
    # `model_dump` keys a field by name; `to_jsonable_python` by its alias.
    [
        (safe_json_serialize, {'cred': '<redacted>', 'note': 'hi'}),
        (_serialize_tool_response, {'note': '<redacted>', 'memo': 'hi'}),
    ],
)
def test_alias_equal_to_another_field_name_masks_the_right_key(
    serialize, expected
):
  """A field is matched by the key the dump used, not another field's name."""

  class _Result(BaseModel):
    cred: ServiceAccountCredential = Field(alias='note')
    note: str = Field(alias='memo')

  result = _Result(note=_service_account(), memo='hi')

  assert json.loads(serialize(result)) == expected


@pytest.mark.parametrize(
    'serialize', [safe_json_serialize, _serialize_tool_response]
)
def test_credential_in_a_dataclass_field_is_masked(serialize):
  """A credential held by a dataclass is masked before the dump drops it."""

  @dataclasses.dataclass
  class _Result:
    sa: ServiceAccountCredential

  class _Holder(BaseModel):
    result: _Result

  holder = _Holder(result=_Result(sa=_service_account()))

  assert json.loads(serialize(holder)) == {'result': {'sa': '<redacted>'}}


@pytest.mark.parametrize(
    'serialize, key',
    # `model_dump` keys a field by name; `to_jsonable_python` by its alias.
    [(safe_json_serialize, 'sa'), (_serialize_tool_response, 'myCred')],
)
def test_credential_in_an_aliased_dataclass_field_is_masked(serialize, key):
  """A dataclass credential field is masked when the dump keys it by alias."""

  @pydantic.dataclasses.dataclass
  class _PydanticResult:
    sa: ServiceAccountCredential = Field(alias='myCred')

  @dataclasses.dataclass
  class _StdlibResult:
    sa: Annotated[ServiceAccountCredential, Field(alias='myCred')]

  class _Holder(BaseModel):
    pydantic_result: _PydanticResult
    stdlib_result: _StdlibResult

  holder = _Holder(
      pydantic_result=_PydanticResult(myCred=_service_account()),
      stdlib_result=_StdlibResult(sa=_service_account()),
  )

  serialized = json.loads(serialize({'result': holder}))['result']

  assert serialized['pydantic_result'] == {key: '<redacted>'}
  assert serialized['stdlib_result'] == {key: '<redacted>'}


@pytest.mark.parametrize(
    'serialize', [safe_json_serialize, _serialize_tool_response]
)
def test_excluded_field_does_not_pair_with_a_computed_field(serialize):
  """A field left out of the dump does not mask an unrelated dumped key."""

  class _Result(BaseModel):
    hidden: ServiceAccountCredential = Field(exclude=True)

    @pydantic.computed_field
    @property
    def total(self) -> int:
      return 42

    @pydantic.model_serializer(mode='wrap')
    def _add_source(self, handler):
      return {**handler(self), 'source': 'cache'}

  result = _Result(hidden=_service_account())

  assert json.loads(serialize({'result': result})) == {
      'result': {'total': 42, 'source': 'cache'}
  }


@pytest.mark.parametrize(
    'serialize', [safe_json_serialize, _serialize_tool_response]
)
def test_credential_in_a_computed_field_is_masked(serialize):
  """A credential returned by a computed field is masked like a field."""

  class _Result(BaseModel):

    @pydantic.computed_field
    @property
    def sa(self) -> ServiceAccountCredential:
      return _service_account()

  assert json.loads(serialize({'result': _Result()})) == {
      'result': {'sa': '<redacted>'}
  }


@pytest.mark.parametrize(
    'serialize', [safe_json_serialize, _serialize_tool_response]
)
def test_credential_in_a_root_model_is_masked(serialize):
  """A root model dumps its root without a key, and is masked through it."""
  single = pydantic.RootModel[ServiceAccountCredential](_service_account())
  many = pydantic.RootModel[list[ServiceAccountCredential]](
      [_service_account()]
  )

  assert json.loads(serialize({'single': single, 'many': many})) == {
      'single': '<redacted>',
      'many': ['<redacted>'],
  }


@pytest.mark.parametrize(
    'serialize', [safe_json_serialize, _serialize_tool_response]
)
def test_credential_under_an_enum_key_is_masked(serialize):
  """The dump keys an `Enum` by its value, which is where the mask must go."""

  class _Holder(BaseModel):
    by_type: dict[AuthCredentialTypes, ServiceAccountCredential]

  holder = _Holder(
      by_type={AuthCredentialTypes.SERVICE_ACCOUNT: _service_account()}
  )

  assert json.loads(serialize({'result': holder})) == {
      'result': {'by_type': {'serviceAccount': '<redacted>'}}
  }


@pytest.mark.parametrize(
    'serialize', [safe_json_serialize, _serialize_tool_response]
)
def test_credential_under_a_bool_or_tuple_key_is_masked(serialize):
  """The dump formats keys unlike `str()`; the mask follows the dump's key."""

  class _Holder(BaseModel):
    by_flag: dict[bool, ServiceAccountCredential]
    by_pair: dict[tuple[int, int], ServiceAccountCredential]

  holder = _Holder(
      by_flag={True: _service_account()},
      by_pair={(1, 2): _service_account()},
  )

  assert json.loads(serialize({'result': holder})) == {
      'result': {
          'by_flag': {'true': '<redacted>'},
          'by_pair': {'1,2': '<redacted>'},
      }
  }


@pytest.mark.parametrize(
    'serialize', [safe_json_serialize, _serialize_tool_response]
)
def test_credential_under_a_colliding_key_is_masked(serialize):
  """Keys that dump alike collapse to the last value, which is masked."""

  class _Holder(BaseModel):
    by_id: dict[Any, Any]

  holder = _Holder(by_id={1: 'public', '1': _service_account()})

  assert json.loads(serialize({'result': holder})) == {
      'result': {'by_id': {'1': '<redacted>'}}
  }


@pytest.mark.parametrize(
    'serialize', [safe_json_serialize, _serialize_tool_response]
)
def test_credential_under_a_reordered_key_is_masked(serialize):
  """A serializer that reorders keys still masks the credential's own key."""

  class _Holder(BaseModel):
    d: dict[str, Any]

    @pydantic.field_serializer('d', mode='wrap')
    def _sort(self, v, handler):
      return dict(sorted(handler(v).items()))

  holder = _Holder(d={'z': _service_account(), 'a': 'public'})

  assert json.loads(serialize({'result': holder})) == {
      'result': {'d': {'a': 'public', 'z': '<redacted>'}}
  }


@pytest.mark.parametrize(
    'serialize', [safe_json_serialize, _serialize_tool_response]
)
def test_credential_in_a_deque_field_is_masked(serialize):
  """A non-list sequence dumps to a list, and its items are masked."""

  class _Holder(BaseModel):
    queue: collections.deque[ServiceAccountCredential]

  holder = _Holder(queue=collections.deque([_service_account()]))

  assert json.loads(serialize({'result': holder})) == {
      'result': {'queue': ['<redacted>']}
  }


def test_safe_json_serialize_keeps_a_str_enum_key_as_its_value():
  """A `(str, Enum)` key renders as its value, as `json.dumps` renders it."""
  assert safe_json_serialize({AuthCredentialTypes.OAUTH2: 'x'}) == (
      '{"oauth2": "x"}'
  )


def test_safe_json_serialize_leaves_ordinary_content_unchanged():
  """Redaction must not alter a payload that holds no credential.

  Key names alone are not credentials: a tool parameter named `page_token` or
  `api_key` keeps its schema, and a tool's own `token` field keeps its value.
  """
  payload = {
      'model': 'gemini-2.5-flash',
      'config': {
          'tools': [{
              'function_declarations': [{
                  'name': 'search',
                  'parameters': {
                      'type': 'OBJECT',
                      'properties': {
                          'filter': {
                              'type': 'OBJECT',
                              'properties': {
                                  'range': {
                                      'type': 'OBJECT',
                                      'properties': {
                                          'unit': {'type': 'STRING'},
                                      },
                                  },
                              },
                          },
                          'page_token': {'type': 'STRING'},
                          'api_key': {'type': 'STRING'},
                      },
                  },
              }],
          }],
      },
      'contents': [{'role': 'user', 'parts': [{'text': 'Paris'}]}],
      'temps': [1, 2.5, None, True],
      'response': {'token': 'next-page-token', 'apiKey': 'public-id'},
  }
  assert json.loads(safe_json_serialize(payload)) == payload


def test_safe_json_serialize_masks_deeply_nested_and_cyclic_credentials():
  """Deeply nested and cyclic credentials must never be emitted in the clear."""
  nested: dict[str, object] = {
      'auth_type': 'oauth2',
      'oauth2': {'access_token': _TOKEN},
  }
  for _ in range(20):
    nested = {'wrap': nested}
  assert _TOKEN not in safe_json_serialize(nested)

  cycle: dict[str, object] = {'access_token': _TOKEN}
  cycle['self'] = cycle
  assert safe_json_serialize(cycle) == '<not serializable>'


@pytest.mark.parametrize(
    'name', ['APIKey', 'myAPIKey', 'XApiKey', 'oauth2IDToken', 'X-Api-Key']
)
def test_is_credential_arg_name_splits_acronym_boundaries(name):
  """An acronym followed by a PascalCase word splits at the acronym's end."""
  assert is_credential_arg_name(name) is True


@pytest.mark.parametrize('key', sorted(auth_credential._CREDENTIAL_SECRET_KEYS))
def test_is_credential_arg_name_covers_every_auth_credential_secret_key(key):
  """The argument-capture name rule must not drift from `AuthCredential`."""
  assert is_credential_arg_name(key) is True
