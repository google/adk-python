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

import collections
import datetime
import decimal
import os
import pathlib
import pickle
from unittest import mock
import uuid

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_tool import AuthConfig
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.events.ui_widget import UiWidget
from google.adk.sessions import _restricted_pickle
from google.adk.sessions.schemas.v0 import Base
from google.adk.sessions.schemas.v0 import DynamicPickleType
from google.adk.sessions.schemas.v0 import StorageEvent
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types
import pytest
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import sessionmaker

_EXECUTED_PAYLOAD_TAGS: list[str] = []


def _record_payload_execution(tag: str) -> str:
  """Stands in for the arbitrary callable a crafted blob would reach."""
  _EXECUTED_PAYLOAD_TAGS.append(tag)
  return tag


class _CraftedActionsBlob:
  """Pickles into a call of a global that the actions allowlist omits."""

  def __reduce__(self):
    return (_record_payload_execution, ("executed",))


_detonations: list[str] = []


def _detonate() -> str:
  """Stands in for attacker-chosen code; must never run during unpickling."""
  _detonations.append("boom")
  return "boom"


class _Payload:
  """Pickles into a payload that names a global outside the allowlist."""

  def __reduce__(self):
    return (_detonate, ())


def _call_global_payload(module: str, name: str, argument: str) -> bytes:
  """Handcrafts a pickle that calls `module.name(argument)` when loaded.

  `pickle.dumps` cannot express a global the writing process does not hold, and
  it resolves `os.system` to its `posix` alias, so the payloads an attacker
  would actually write have to be assembled by hand.

  Args:
      module: The module the payload resolves the callable from.
      name: The callable's name within that module.
      argument: The single string argument the payload passes.

  Returns:
      The handcrafted pickle payload.
  """

  def short_unicode(value: str) -> bytes:
    encoded = value.encode()
    return pickle.SHORT_BINUNICODE + bytes([len(encoded)]) + encoded

  return b"".join([
      pickle.PROTO,
      b"\x04",
      short_unicode(module),
      short_unicode(name),
      pickle.STACK_GLOBAL,
      short_unicode(argument),
      pickle.TUPLE1,
      pickle.REDUCE,
      pickle.STOP,
  ])


def _fully_populated_event_actions() -> EventActions:
  """Builds an `EventActions` exercising every field it can hold.

  Every nested model, enum and stdlib type reachable from `EventActions` has to
  survive a restricted round trip, so a legacy database keeps loading.
  """
  service_account = ServiceAccount(
      service_account_credential=ServiceAccountCredential(
          type="service_account",
          project_id="project",
          private_key_id="key-id",
          private_key="private-key",
          client_email="agent@example.com",
          client_id="client-id",
          auth_uri="https://example.com/auth",
          token_uri="https://example.com/token",
          auth_provider_x509_cert_url="https://example.com/certs",
          client_x509_cert_url="https://example.com/client-cert",
          universe_domain="googleapis.com",
      ),
      scopes=["https://example.com/scope"],
  )
  auth_config = AuthConfig(
      auth_scheme=OpenIdConnectWithConfig(
          authorization_endpoint="https://example.com/auth",
          token_endpoint="https://example.com/token",
          scopes=["openid"],
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
          service_account=service_account,
          http=HttpAuth(
              scheme="bearer", credentials=HttpCredentials(token="token")
          ),
          oauth2=OAuth2Auth(client_id="client-id", client_secret="secret"),
      ),
  )
  content = types.Content(
      role="model",
      parts=[
          types.Part(text="hello", thought=True, thought_signature=b"sig"),
          types.Part(
              code_execution_result=types.CodeExecutionResult(
                  outcome=types.Outcome.OUTCOME_OK, output="ok"
              )
          ),
          types.Part(
              executable_code=types.ExecutableCode(
                  code="print(1)", language=types.Language.PYTHON
              )
          ),
          types.Part(
              function_call=types.FunctionCall(
                  name="fn", args={"a": 1}, id="call-1"
              )
          ),
          types.Part(
              function_response=types.FunctionResponse(
                  name="fn",
                  response={"a": 1},
                  scheduling=types.FunctionResponseScheduling.INTERRUPT,
                  parts=[
                      types.FunctionResponsePart(
                          inline_data=types.FunctionResponseBlob(
                              data=b"x", mime_type="text/plain"
                          )
                      ),
                      types.FunctionResponsePart(
                          file_data=types.FunctionResponseFileData(
                              file_uri="gs://bucket/object",
                              mime_type="text/plain",
                          )
                      ),
                  ],
              )
          ),
          types.Part(
              inline_data=types.Blob(data=b"x", mime_type="image/png"),
              video_metadata=types.VideoMetadata(fps=1.0),
          ),
          types.Part(
              file_data=types.FileData(
                  file_uri="gs://bucket/video", mime_type="video/mp4"
              )
          ),
      ],
  )
  return EventActions(
      skip_summarization=True,
      state_delta={
          "text": "value",
          "number": 1,
          "float": 1.5,
          "bool": True,
          "bytes": b"value",
          "list": [1, 2],
          "dict": {"key": "value"},
          "tuple": (1, 2),
          "set": {1, 2},
          "datetime": datetime.datetime.now(datetime.timezone.utc),
          "timedelta": datetime.timedelta(seconds=1),
          "date": datetime.date(2026, 1, 1),
          "time": datetime.time(12, 30, tzinfo=datetime.timezone.utc),
          "ordered_dict": collections.OrderedDict(a=1, b=2),
          "default_dict": collections.defaultdict(list, a=[1]),
          "uuid": uuid.UUID("12345678-1234-5678-1234-567812345678"),
          "decimal": decimal.Decimal("1.5"),
          "path": pathlib.PurePosixPath("/data/artifact.txt"),
          "complex": complex(1, 2),
      },
      artifact_delta={"artifact.txt": 1},
      transfer_to_agent="another_agent",
      escalate=True,
      requested_auth_configs={"call-1": auth_config},
      requested_tool_confirmations={
          "call-1": ToolConfirmation(
              hint="hint", confirmed=True, payload={"key": "value"}
          )
      },
      compaction=EventCompaction(
          start_timestamp=1.0, end_timestamp=2.0, compacted_content=content
      ),
      end_of_agent=True,
      agent_state={"key": "value"},
      rewind_before_invocation_id="invocation-1",
      route=["edge", 1, True],
      render_ui_widgets=[
          UiWidget(id="widget-1", provider="mcp", payload={"key": "value"})
      ],
  )


@pytest.fixture
def pickle_type():
  """Fixture for DynamicPickleType instance."""
  return DynamicPickleType()


@pytest.fixture
def crafted_blob():
  """Fixture for a pickled blob that runs code when loaded unrestricted."""
  _EXECUTED_PAYLOAD_TAGS.clear()
  yield pickle.dumps(_CraftedActionsBlob())
  _EXECUTED_PAYLOAD_TAGS.clear()


@pytest.fixture(autouse=True)
def clear_detonations():
  _detonations.clear()
  yield
  _detonations.clear()


def test_load_dialect_impl_mysql(pickle_type):
  """Test that MySQL dialect uses LONGBLOB."""
  # Mock the MySQL dialect
  mock_dialect = mock.Mock()
  mock_dialect.name = "mysql"

  # Mock the return value of type_descriptor
  mock_longblob_type = mock.Mock()
  mock_dialect.type_descriptor.return_value = mock_longblob_type

  impl = pickle_type.load_dialect_impl(mock_dialect)

  # SQLAlchemy dialect descriptors operate on type instances, not classes.
  mock_dialect.type_descriptor.assert_called_once()
  assert isinstance(
      mock_dialect.type_descriptor.call_args.args[0], mysql.LONGBLOB
  )
  # Verify the return value is what we expect
  assert impl == mock_longblob_type


def test_load_dialect_impl_spanner(pickle_type):
  """Test that Spanner dialect uses SpannerPickleType."""
  # Mock the spanner dialect
  mock_dialect = mock.Mock()
  mock_dialect.name = "spanner+spanner"

  with mock.patch(
      "google.cloud.sqlalchemy_spanner.sqlalchemy_spanner.SpannerPickleType"
  ) as mock_spanner_type:
    pickle_type.load_dialect_impl(mock_dialect)
    mock_spanner_type.assert_called_once_with()
    mock_dialect.type_descriptor.assert_called_once_with(
        mock_spanner_type.return_value
    )


def test_load_dialect_impl_default(pickle_type):
  """Test that other dialects use default PickleType."""
  engine = create_engine("sqlite:///:memory:")
  dialect = engine.dialect
  impl = pickle_type.load_dialect_impl(dialect)
  # Should return the default impl (PickleType)
  assert impl == pickle_type.impl


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_process_bind_param_pickle_dialects(pickle_type, dialect_name):
  """Test that MySQL and Spanner dialects pickle the value."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name

  test_data = {"key": "value", "nested": [1, 2, 3]}
  result = pickle_type.process_bind_param(test_data, mock_dialect)

  # Should be pickled bytes
  assert isinstance(result, bytes)
  # Should be able to unpickle back to original
  assert pickle.loads(result) == test_data


def test_process_bind_param_default(pickle_type):
  """Test that other dialects return value as-is."""
  mock_dialect = mock.Mock()
  mock_dialect.name = "sqlite"

  test_data = {"key": "value"}
  result = pickle_type.process_bind_param(test_data, mock_dialect)

  # Should return value unchanged (SQLAlchemy's PickleType handles it)
  assert result == test_data


def test_process_bind_param_none(pickle_type):
  """Test that None values are handled correctly."""
  mock_dialect = mock.Mock()
  mock_dialect.name = "mysql"

  result = pickle_type.process_bind_param(None, mock_dialect)
  assert result is None


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_process_result_value_pickle_dialects(pickle_type, dialect_name):
  """Test that MySQL and Spanner dialects unpickle the value."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name

  test_data = {"key": "value", "nested": [1, 2, 3]}
  pickled_data = pickle.dumps(test_data)

  result = pickle_type.process_result_value(pickled_data, mock_dialect)

  # Should be unpickled back to original
  assert result == test_data


def test_process_result_value_default(pickle_type):
  """Test that other dialects return value as-is."""
  mock_dialect = mock.Mock()
  mock_dialect.name = "sqlite"

  test_data = {"key": "value"}
  result = pickle_type.process_result_value(test_data, mock_dialect)

  # Should return value unchanged (SQLAlchemy's PickleType handles it)
  assert result == test_data


def test_process_result_value_none(pickle_type):
  """Test that None values are handled correctly."""
  mock_dialect = mock.Mock()
  mock_dialect.name = "mysql"

  result = pickle_type.process_result_value(None, mock_dialect)
  assert result is None


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_roundtrip_pickle_dialects(pickle_type, dialect_name):
  """Test full roundtrip for MySQL and Spanner: bind -> result."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name

  original_data = {
      "string": "test",
      "number": 42,
      "list": [1, 2, 3],
      "nested": {"a": 1, "b": 2},
  }

  # Simulate bind (Python -> DB)
  bound_value = pickle_type.process_bind_param(original_data, mock_dialect)
  assert isinstance(bound_value, bytes)

  # Simulate result (DB -> Python)
  result_value = pickle_type.process_result_value(bound_value, mock_dialect)
  assert result_value == original_data


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_process_result_value_rejects_disallowed_global(
    pickle_type, dialect_name, crafted_blob
):
  """MySQL and Spanner blobs may only name globals on the allowlist."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name

  with pytest.raises(pickle.UnpicklingError):
    pickle_type.process_result_value(crafted_blob, mock_dialect)

  assert _EXECUTED_PAYLOAD_TAGS == []


def test_reading_event_rejects_actions_blob_with_disallowed_global(
    crafted_blob,
):
  """Dialects handled by SQLAlchemy's PickleType are restricted too."""
  engine = create_engine("sqlite://")
  Base.metadata.create_all(engine)
  with engine.begin() as conn:
    conn.execute(
        text(
            "INSERT INTO events (id, app_name, user_id, session_id,"
            " invocation_id, author, actions, timestamp) VALUES ('event1',"
            " 'app1', 'user1', 'session1', 'invoke1', 'user', :actions,"
            " '2026-01-01 00:00:00')"
        ),
        {"actions": crafted_blob},
    )

  with sessionmaker(bind=engine)() as sql_session:
    with pytest.raises(pickle.UnpicklingError):
      sql_session.execute(select(StorageEvent)).scalars().first()

  assert _EXECUTED_PAYLOAD_TAGS == []


def test_reading_event_still_loads_stored_actions():
  """Allowlisted action payloads keep round-tripping through the database."""
  engine = create_engine("sqlite://")
  Base.metadata.create_all(engine)
  Session = sessionmaker(bind=engine)
  with Session() as sql_session:
    sql_session.add(
        StorageEvent(
            id="event1",
            app_name="app1",
            user_id="user1",
            session_id="session1",
            invocation_id="invoke1",
            author="user",
            actions=EventActions(state_delta={"key": "value"}),
        )
    )
    sql_session.commit()

  with Session() as sql_session:
    stored = sql_session.execute(select(StorageEvent)).scalars().first()
    assert stored is not None
    assert stored.actions.state_delta == {"key": "value"}


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_process_result_value_blocks_disallowed_global(
    pickle_type, dialect_name
):
  """Stored bytes naming a class outside the allowlist must not be loaded."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name

  with pytest.raises(pickle.UnpicklingError):
    pickle_type.process_result_value(pickle.dumps(_Payload()), mock_dialect)

  assert not _detonations


def test_default_dialect_impl_blocks_disallowed_global(pickle_type):
  """Dialects served by the default impl must be restricted too."""
  engine = create_engine("sqlite:///:memory:")
  processor = pickle_type.impl.result_processor(engine.dialect, None)

  with pytest.raises(pickle.UnpicklingError):
    processor(pickle.dumps(_Payload()))

  assert not _detonations


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_roundtrip_event_actions(pickle_type, dialect_name):
  """A legitimately stored EventActions payload still round-trips."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name

  actions = EventActions(
      skip_summarization=True,
      state_delta={"key": "value"},
      transfer_to_agent="another_agent",
  )

  bound_value = pickle_type.process_bind_param(actions, mock_dialect)
  result_value = pickle_type.process_result_value(bound_value, mock_dialect)

  assert result_value == actions


def test_default_dialect_impl_roundtrips_event_actions(pickle_type):
  """The default impl still round-trips a legitimate EventActions payload."""
  engine = create_engine("sqlite:///:memory:")
  actions = EventActions(state_delta={"key": "value"})

  bind_processor = pickle_type.impl.bind_processor(engine.dialect)
  result_processor = pickle_type.impl.result_processor(engine.dialect, None)

  assert result_processor(bind_processor(actions)) == actions


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_roundtrip_fully_populated_event_actions(pickle_type, dialect_name):
  """Every type a legacy `events.actions` blob can hold must still load."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name
  actions = _fully_populated_event_actions()

  bound_value = pickle_type.process_bind_param(actions, mock_dialect)
  result_value = pickle_type.process_result_value(bound_value, mock_dialect)

  assert result_value == actions


def test_dialect_impl_roundtrips_fully_populated_event_actions(pickle_type):
  """The same holds on the copy `_gen_dialect_impl` builds for real reads."""
  engine = create_engine("sqlite:///:memory:")
  dialect_impl = pickle_type.dialect_impl(engine.dialect)
  actions = _fully_populated_event_actions()

  bind_processor = dialect_impl.bind_processor(engine.dialect)
  result_processor = dialect_impl.result_processor(engine.dialect, None)

  assert result_processor(bind_processor(actions)) == actions


def test_dialect_impl_blocks_disallowed_global(pickle_type):
  """The copy real reads go through must be restricted too."""
  engine = create_engine("sqlite:///:memory:")
  dialect_impl = pickle_type.dialect_impl(engine.dialect)
  result_processor = dialect_impl.result_processor(engine.dialect, None)

  with pytest.raises(pickle.UnpicklingError):
    result_processor(pickle.dumps(_Payload()))

  assert not _detonations


def test_blocked_global_error_is_diagnosable():
  """A payload that cannot be loaded must say what was refused, and why."""
  with pytest.raises(pickle.UnpicklingError) as exc_info:
    _restricted_pickle.loads(pickle.dumps(_Payload()))

  message = str(exc_info.value)
  assert _detonate.__module__ in message
  assert "_detonate" in message
  assert "adk migrate session --allow-unsafe-unpickling" in message


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("google.adk.auth.auth_credential", "ServiceAccount"),
        ("google.genai.types", "Outcome"),
        ("google.genai.types", "Language"),
        ("google.genai.types", "FunctionResponseScheduling"),
        ("google.genai.types", "PartMediaResolutionLevel"),
    ],
)
def test_allowed_globals_are_derived_from_the_model_tree(
    module_name, class_name
):
  """Nested models and enums are admitted without being hand-listed."""
  assert (module_name, class_name) not in (
      _restricted_pickle._STATIC_ALLOWED_GLOBALS
  )
  assert (module_name, class_name) in _restricted_pickle._allowed_globals()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(collections.OrderedDict(a=1, b=2), id="ordered_dict"),
        pytest.param(collections.defaultdict(list, a=[1]), id="default_dict"),
        pytest.param(datetime.date(2026, 1, 1), id="date"),
        pytest.param(datetime.time(12, 30), id="time"),
        pytest.param(
            datetime.time(12, 30, tzinfo=datetime.timezone.utc), id="time_tz"
        ),
        pytest.param(
            uuid.UUID("12345678-1234-5678-1234-567812345678"), id="uuid"
        ),
        pytest.param(decimal.Decimal("1.5"), id="decimal"),
        pytest.param(pathlib.PurePosixPath("/data/x.txt"), id="pure_path"),
        pytest.param(pathlib.Path("/data/x.txt"), id="path"),
        pytest.param(complex(1, 2), id="complex"),
    ],
)
def test_plain_stdlib_state_values_still_load(value):
  """State written by an earlier ADK holds these, so they must keep loading."""
  assert _restricted_pickle.loads(pickle.dumps(value)) == value


@pytest.mark.parametrize(
    "module_name,attribute_name",
    [
        pytest.param("os", "system", id="os_system"),
        pytest.param("builtins", "eval", id="builtins_eval"),
        pytest.param("pathlib", "os.system", id="via_pathlib"),
        pytest.param("uuid", "os.system", id="via_uuid"),
        pytest.param("collections", "OrderedDict.fromkeys", id="via_ordered"),
    ],
)
def test_dangerous_globals_stay_refused(module_name, attribute_name):
  """Widening the allowlist must not reach a callable through a module on it."""
  payload = _call_global_payload(module_name, attribute_name, "echo unreached")

  with pytest.raises(pickle.UnpicklingError):
    _restricted_pickle.loads(payload)


def test_call_global_payload_would_execute_unrestricted():
  """Guards the adversarial cases above from silently becoming inert."""
  # The unrestricted load is the assertion: it proves the handcrafted payload
  # really does reach a callable, so the restricted loader refusing it above
  # means something. The payload evaluates "1 + 1" and touches nothing else.
  payload = _call_global_payload("builtins", "eval", "1 + 1")
  assert pickle.loads(payload) == 2  # pylint: disable=g-unsafe-pickle-load


def test_defaultdict_factory_must_itself_be_allowlisted():
  """A `defaultdict` carries a callable, which the allowlist must cover too."""
  with pytest.raises(pickle.UnpicklingError):
    _restricted_pickle.loads(pickle.dumps(collections.defaultdict(os.system)))
