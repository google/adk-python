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

"""Tests for the a2a-sdk version shim.

The shim's contract is that, for every SDK shape it claims to support, it
returns the normalized form, and for a shape it does not recognize it fails
loudly instead of silently returning ``None``.

Only one a2a-sdk major is installed at a time, so the branch for the *other*
major can only be exercised where the shim is duck-typed: those tests flip
``_compat.IS_A2A_V1`` and feed the shim the protobuf objects that branch
expects. Branches that import 1.x-only SDK symbols are not reachable here.
"""

from __future__ import annotations

import json

from a2a.client.client_factory import ClientFactory
from a2a.types import AgentCapabilities
from a2a.types import AgentProvider
from a2a.types import AgentSkill
from a2a.types import Artifact
from a2a.types import TaskArtifactUpdateEvent
from google.adk.a2a import _compat
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
import pytest

v03_only = pytest.mark.skipif(
    _compat.IS_A2A_V1, reason='0.3-only SDK object shapes'
)


def _struct(payload: dict) -> Struct:
  return ParseDict(payload, Struct())


class _FakeStreamResponse:
  """Duck-typed stand-in for the 1.x ``StreamResponse`` proto.

  ``stream_item_kind``'s 1.x branch only needs ``HasField`` plus attribute
  access, so the oneof can be modelled without the 1.x SDK installed.
  """

  def __init__(self, field=None, payload=None):
    self._field = field
    if field is not None:
      setattr(self, field, payload)

  def HasField(self, name: str) -> bool:  # noqa: N802 - proto API name.
    return name == self._field


class _FakeStructEvent:
  """Stand-in for a 1.x event whose ``metadata`` is a proto ``Struct``."""

  def __init__(self):
    self.metadata = Struct()


def _task():
  return _compat.make_task(
      id='task-1',
      context_id='ctx-1',
      status=_compat.make_task_status(_compat.TS_WORKING),
  )


# -----------------------------------------------------------------------------
# build_agent_card
# -----------------------------------------------------------------------------
def _build_card(**overrides):
  kwargs = dict(
      name='card-name',
      description='card-description',
      version='1.2.3',
      url='https://agent.example/a2a',
      protocol_binding=_compat.TP_JSONRPC,
  )
  kwargs.update(overrides)
  return _compat.build_agent_card(**kwargs)


@v03_only
def test_build_agent_card_strips_trailing_slash_from_url():
  # The RPC URL is concatenated with paths by callers, so the card must not
  # carry a trailing separator.
  assert _build_card(url='https://agent.example/a2a/').url == (
      'https://agent.example/a2a'
  )


@v03_only
def test_build_agent_card_without_protocol_version_uses_v03_default():
  assert _build_card().protocol_version == '0.3.0'


@v03_only
def test_build_agent_card_with_protocol_version_keeps_caller_value():
  assert _build_card(protocol_version='0.2.9').protocol_version == '0.2.9'


@pytest.mark.parametrize('streaming', [True, False])
@v03_only
def test_build_agent_card_default_capabilities_follow_streaming_flag(streaming):
  capabilities = _build_card(streaming=streaming).capabilities
  assert capabilities.streaming is streaming
  # Push notifications are never advertised by the default capabilities.
  assert capabilities.push_notifications is False


@v03_only
def test_build_agent_card_explicit_capabilities_override_streaming_flag():
  card = _build_card(
      streaming=False,
      capabilities=AgentCapabilities(streaming=True, push_notifications=True),
  )
  assert card.capabilities.streaming is True
  assert card.capabilities.push_notifications is True


@v03_only
def test_build_agent_card_omits_optional_fields_when_not_supplied():
  card = _build_card(provider=None, security_schemes=None, doc_url=None)
  assert card.provider is None
  assert card.security_schemes is None
  assert card.documentation_url is None
  assert card.supports_authenticated_extended_card is False


@v03_only
def test_build_agent_card_converts_model_arguments_to_card_fields():
  card = _build_card(
      protocol_binding=_compat.TP_HTTP_JSON,
      skills=[
          AgentSkill(id='skill-1', name='Skill One', description='d', tags=[])
      ],
      provider=AgentProvider(organization='acme', url='https://acme.example'),
      security_schemes={'api': _compat.make_api_key_scheme(name='X-Api-Key')},
      doc_url='https://agent.example/docs',
      default_input_modes=('text/plain', 'application/json'),
      supports_authenticated_extended_card=True,
  )
  assert card.preferred_transport == _compat.TP_HTTP_JSON
  assert [skill.id for skill in card.skills] == ['skill-1']
  assert card.provider.organization == 'acme'
  assert card.security_schemes['api'].root.name == 'X-Api-Key'
  assert card.documentation_url == 'https://agent.example/docs'
  assert card.default_input_modes == ['text/plain', 'application/json']
  assert card.supports_authenticated_extended_card is True


# -----------------------------------------------------------------------------
# rebind_client_factory_httpx
# -----------------------------------------------------------------------------
def _factory_with_custom_transport(httpx_client, consumers):
  factory = ClientFactory(
      _compat.make_client_config(httpx_client=httpx_client, streaming=True),
      consumers=consumers,
  )
  factory.register('custom-transport', _custom_transport_producer)
  return factory


def _custom_transport_producer(*args, **kwargs):
  raise AssertionError('the producer is only used as an identity marker')


@v03_only
def test_rebind_client_factory_httpx_returns_new_factory_on_new_client():
  old_client, new_client = object(), object()
  factory = _factory_with_custom_transport(old_client, consumers=[])

  rebound = _compat.rebind_client_factory_httpx(factory, new_client)

  assert rebound is not factory
  assert rebound._config.httpx_client is new_client
  # The caller may still be using the original factory; it must be untouched.
  assert factory._config.httpx_client is old_client


@v03_only
def test_rebind_client_factory_httpx_preserves_config_consumers_transports():
  consumer = lambda event, card: None
  factory = _factory_with_custom_transport(object(), consumers=[consumer])

  rebound = _compat.rebind_client_factory_httpx(factory, object())

  # ``streaming=True`` is not the value ADK's config builder defaults to, so
  # seeing it survive proves the rest of the config came along.
  assert rebound._config.streaming is True
  assert rebound._consumers == [consumer]
  assert rebound._registry['custom-transport'] is _custom_transport_producer


# -----------------------------------------------------------------------------
# stream_item_kind
# -----------------------------------------------------------------------------
@v03_only
def test_stream_item_kind_task_without_update_is_a_task_item():
  task = _task()
  assert _compat.stream_item_kind((task, None)) == ('task', task)


@v03_only
def test_stream_item_kind_status_update_tuple_returns_the_update():
  update = _compat.make_task_status_update_event(
      task_id='task-1',
      context_id='ctx-1',
      status=_compat.make_task_status(_compat.TS_WORKING),
      final=False,
  )
  assert _compat.stream_item_kind((_task(), update)) == (
      'status_update',
      update,
  )


@v03_only
def test_stream_item_kind_artifact_update_tuple_returns_the_update():
  update = TaskArtifactUpdateEvent(
      task_id='task-1',
      context_id='ctx-1',
      artifact=Artifact(
          artifact_id='artifact-1', parts=[_compat.make_text_part('hi')]
      ),
  )
  assert _compat.stream_item_kind((_task(), update)) == (
      'artifact_update',
      update,
  )


@v03_only
def test_stream_item_kind_bare_message_is_a_message_item():
  message = _compat.make_message(message_id='m-1', role='user')
  assert _compat.stream_item_kind(message) == ('message', message)


@v03_only
def test_stream_item_kind_unknown_update_raises_rather_than_returning_none():
  with pytest.raises(ValueError, match='Unknown v0.3 update event'):
    _compat.stream_item_kind((_task(), 'not-an-update-event'))


@pytest.mark.parametrize(
    'field', ['task', 'message', 'status_update', 'artifact_update']
)
def test_stream_item_kind_v1_reports_the_set_oneof_field(monkeypatch, field):
  monkeypatch.setattr(_compat, 'IS_A2A_V1', True)
  payload = object()
  assert _compat.stream_item_kind(_FakeStreamResponse(field, payload)) == (
      field,
      payload,
  )


def test_stream_item_kind_v1_without_payload_raises(monkeypatch):
  monkeypatch.setattr(_compat, 'IS_A2A_V1', True)
  with pytest.raises(ValueError, match='no known payload field'):
    _compat.stream_item_kind(_FakeStreamResponse())


# -----------------------------------------------------------------------------
# data_part_blob_bytes / make_data_part_from_blob
# -----------------------------------------------------------------------------
@v03_only
def test_data_part_blob_bytes_serializes_the_whole_data_part():
  part = _compat.make_data_part(data={'a': 1}, metadata={'m': 'v'})

  blob = json.loads(_compat.data_part_blob_bytes(part))

  # 0.3.x embeds the metadata (and the discriminator) in the blob; only the
  # data dict would survive on 1.x.
  assert blob == {'data': {'a': 1}, 'metadata': {'m': 'v'}, 'kind': 'data'}


@v03_only
def test_data_part_blob_bytes_omits_unset_fields():
  blob = json.loads(
      _compat.data_part_blob_bytes(_compat.make_data_part(data={'a': 1}))
  )
  assert 'metadata' not in blob


@v03_only
def test_make_data_part_from_blob_restores_data_and_embedded_metadata():
  # ``DataPart`` only exists on 0.3.x, so it is imported where the shim does:
  # inside the branch that needs it, not at module scope.
  from a2a.types import DataPart

  original = _compat.make_data_part(data={'a': 1}, metadata={'m': 'v'})

  restored = _compat.make_data_part_from_blob(
      _compat.data_part_blob_bytes(original)
  )

  assert isinstance(restored.root, DataPart)
  assert restored.root.data == {'a': 1}
  assert restored.root.metadata == {'m': 'v'}


@v03_only
def test_make_data_part_from_blob_merges_extra_metadata():
  blob = _compat.data_part_blob_bytes(
      _compat.make_data_part(data={'a': 1}, metadata={'m': 'v', 'keep': 'yes'})
  )

  restored = _compat.make_data_part_from_blob(
      blob, extra_metadata={'m': 'overridden', 'extra': 'e'}
  )

  assert restored.root.metadata == {
      'm': 'overridden',
      'keep': 'yes',
      'extra': 'e',
  }


@v03_only
def test_make_data_part_from_blob_adds_metadata_when_blob_has_none():
  blob = _compat.data_part_blob_bytes(_compat.make_data_part(data={'a': 1}))

  restored = _compat.make_data_part_from_blob(
      blob, extra_metadata={'extra': 'e'}
  )

  assert restored.root.metadata == {'extra': 'e'}


# -----------------------------------------------------------------------------
# metadata_get
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('metadata', [None, {}])
def test_metadata_get_empty_metadata_returns_default(metadata):
  assert _compat.metadata_get(metadata, 'k', 'fallback') == 'fallback'


def test_metadata_get_reads_and_defaults_on_a_dict():
  assert _compat.metadata_get({'k': 'v'}, 'k', 'fallback') == 'v'
  assert _compat.metadata_get({'k': 'v'}, 'other', 'fallback') == 'fallback'
  assert _compat.metadata_get({'k': 'v'}, 'other') is None


def test_metadata_get_v1_reads_and_defaults_on_a_struct(monkeypatch):
  monkeypatch.setattr(_compat, 'IS_A2A_V1', True)
  metadata = _struct({'k': 'v'})
  assert _compat.metadata_get(metadata, 'k', 'fallback') == 'v'
  assert _compat.metadata_get(metadata, 'other', 'fallback') == 'fallback'


def test_metadata_get_v1_unusable_key_returns_default(monkeypatch):
  # A proto Struct raises on a non-string key; the shim must degrade to the
  # default rather than propagate that to the caller.
  monkeypatch.setattr(_compat, 'IS_A2A_V1', True)
  assert _compat.metadata_get(_struct({'k': 'v'}), 5, 'fallback') == 'fallback'


# -----------------------------------------------------------------------------
# set_event_metadata
# -----------------------------------------------------------------------------
@v03_only
def test_set_event_metadata_assigns_the_given_keys():
  event = _compat.make_task_status_update_event(
      task_id='task-1',
      context_id='ctx-1',
      status=_compat.make_task_status(_compat.TS_WORKING),
  )

  _compat.set_event_metadata(event, {'a': 'b'})

  assert event.metadata == {'a': 'b'}


@pytest.mark.parametrize('metadata', [None, {}])
@v03_only
def test_set_event_metadata_empty_leaves_existing_metadata_intact(metadata):
  event = _compat.make_task_status_update_event(
      task_id='task-1',
      context_id='ctx-1',
      status=_compat.make_task_status(_compat.TS_WORKING),
      metadata={'already': 'here'},
  )

  _compat.set_event_metadata(event, metadata)

  assert event.metadata == {'already': 'here'}


def test_set_event_metadata_v1_copies_into_the_struct_field(monkeypatch):
  monkeypatch.setattr(_compat, 'IS_A2A_V1', True)
  event = _FakeStructEvent()

  _compat.set_event_metadata(event, {'a': 'b'})

  assert dict(event.metadata) == {'a': 'b'}


# -----------------------------------------------------------------------------
# meta_to_dict
# -----------------------------------------------------------------------------
def test_meta_to_dict_none_returns_empty_dict():
  assert _compat.meta_to_dict(None) == {}


def test_meta_to_dict_dict_is_returned_unchanged():
  assert _compat.meta_to_dict({'a': 1}) == {'a': 1}


def test_meta_to_dict_unsupported_shape_returns_empty_dict():
  # Callers json.dumps() the result, so anything unrecognized must normalize
  # to an empty dict rather than leak through.
  assert _compat.meta_to_dict('not-metadata') == {}


def test_meta_to_dict_v1_converts_a_struct(monkeypatch):
  monkeypatch.setattr(_compat, 'IS_A2A_V1', True)
  assert _compat.meta_to_dict(_struct({'a': 'b'})) == {'a': 'b'}


# -----------------------------------------------------------------------------
# role_to_str / part_kind_label
# -----------------------------------------------------------------------------
def test_role_to_str_maps_user_role_to_user():
  assert _compat.role_to_str(_compat.ROLE_USER) == 'user'


@pytest.mark.parametrize('role', [_compat.ROLE_AGENT, None, 'nonsense'])
def test_role_to_str_maps_every_other_role_to_model(role):
  assert _compat.role_to_str(role) == 'model'


def test_part_kind_label_is_fixed_on_v03_and_concrete_on_v1(monkeypatch):
  file_part = _compat.make_file_part_with_uri(uri='gs://bucket/object')

  # 0.3.x wraps every file payload as a FilePart, so the log label is fixed
  # even though the object handed in is a ``Part``.
  monkeypatch.setattr(_compat, 'IS_A2A_V1', False)
  assert _compat.part_kind_label(file_part) == 'FilePart'

  # 1.x has no wrapper type, so the label is the concrete class name.
  monkeypatch.setattr(_compat, 'IS_A2A_V1', True)
  assert _compat.part_kind_label(file_part) == 'Part'
