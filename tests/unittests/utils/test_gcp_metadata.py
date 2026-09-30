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

"""Tests for GCP metadata runtime defaults."""

from __future__ import annotations

import os

from google.adk.utils import _gcp_metadata
from google.adk.utils._gcp_metadata import get_gcp_client_defaults
from google.auth import _cloud_sdk
from google.auth import exceptions as auth_exceptions
import pytest


@pytest.fixture(autouse=True)
def _clear_gcp_env(monkeypatch: pytest.MonkeyPatch) -> None:
  for name in (
      'GOOGLE_CLOUD_PROJECT',
      'GOOGLE_CLOUD_LOCATION',
      'GOOGLE_GENAI_USE_ENTERPRISE',
      'GOOGLE_GENAI_USE_VERTEXAI',
      'GOOGLE_API_KEY',
      'GEMINI_API_KEY',
      'GOOGLE_APPLICATION_CREDENTIALS',
  ):
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setattr(
      _cloud_sdk, 'get_application_default_credentials_path', lambda: ''
  )
  # Reset the probed project ID between tests.
  monkeypatch.setattr(_gcp_metadata, '_cached_project_id', None)


def test_get_gcp_client_defaults_noop_off_gcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Offline / local environments leave env vars and client defaults empty."""
  from unittest import mock

  from google.auth.compute_engine import _metadata

  ping_mock = mock.MagicMock(return_value=False)
  monkeypatch.setattr(_metadata, 'ping', ping_mock)

  assert get_gcp_client_defaults(client_kwargs={'enterprise': True}) == {}
  ping_mock.assert_called_once()
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in os.environ


def test_get_gcp_client_defaults_skips_defaults_without_enterprise_on_gcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """On GCP without enterprise mode, defaults are skipped to preserve Gemini API."""
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults()

  assert defaults == {}
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in os.environ


def test_get_gcp_client_defaults_never_overrides_explicit_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Shell / .env values win over metadata defaults."""
  monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'explicit-project')
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', 'true')
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults()

  assert defaults == {}
  assert os.environ['GOOGLE_CLOUD_PROJECT'] == 'explicit-project'
  assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == 'true'


def test_get_gcp_client_defaults_skips_defaults_when_api_key_present_in_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """API-key auth in env must not force enterprise mode or set project."""
  monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults(client_kwargs={'enterprise': True})

  assert defaults == {}
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in os.environ


def test_get_gcp_client_defaults_skips_defaults_when_api_key_present_in_client_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """API-key auth in client_kwargs must not force enterprise mode or set project."""
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults(
      client_kwargs={'api_key': 'test-key', 'enterprise': True}
  )

  assert defaults == {}
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in os.environ


def test_get_gcp_client_defaults_preserves_vertex_express_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Vertex express mode (vertex flag + api key, no project) leaves project unset."""
  monkeypatch.setenv('GOOGLE_GENAI_USE_VERTEXAI', 'true')
  monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults()

  assert defaults == {}
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ


@pytest.mark.parametrize(
    'error',
    [
        auth_exceptions.TransportError('unreachable'),
        auth_exceptions.GoogleAuthError('auth failed'),
        auth_exceptions.DefaultCredentialsError('no credentials'),
        OSError('connection reset'),
    ],
)
def test_metadata_get_returns_none_on_auth_exception(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
  """Project lookup fails soft when metadata service raises errors."""
  from google.auth.compute_engine import _metadata

  monkeypatch.setattr(_metadata, 'ping', lambda *a, **kw: True)
  monkeypatch.setattr(
      _metadata,
      'get',
      lambda *a, **kw: (_ for _ in ()).throw(error),
  )
  assert _gcp_metadata.get_project_id_from_metadata() is None


def test_get_project_id_from_metadata_transient_error_does_not_latch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Transient transport error fails soft without latching so retries can succeed."""
  from google.auth.compute_engine import _metadata

  calls = {'ping': 0, 'get': 0}

  def fake_ping(*args, **kwargs):
    calls['ping'] += 1
    return True

  def fake_get(*args, **kwargs):
    calls['get'] += 1
    if calls['get'] == 1:
      raise auth_exceptions.TransportError('Cold start timeout')
    return 'retry-project'

  monkeypatch.setattr(_metadata, 'ping', fake_ping)
  monkeypatch.setattr(_metadata, 'get', fake_get)

  # First call fails soft due to transient error.
  assert _gcp_metadata.get_project_id_from_metadata() is None
  assert calls['get'] == 1

  # Second call retries probe without latching failure.
  assert _gcp_metadata.get_project_id_from_metadata() == 'retry-project'
  assert calls['get'] == 2


def test_get_project_id_from_metadata_cold_start_ping_failure_does_not_latch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Cold-start ping failure fails soft without latching so retries can succeed."""
  from google.auth.compute_engine import _metadata

  calls = {'ping': 0, 'get': 0}

  def fake_ping(*args, **kwargs):
    calls['ping'] += 1
    if calls['ping'] == 1:
      return False
    return True

  def fake_get(*args, **kwargs):
    calls['get'] += 1
    return 'retry-project'

  monkeypatch.setattr(_metadata, 'ping', fake_ping)
  monkeypatch.setattr(_metadata, 'get', fake_get)

  # First call fails soft due to cold-start ping failure.
  assert _gcp_metadata.get_project_id_from_metadata() is None
  assert calls['ping'] == 1

  # Second call retries ping without latching off-GCP, succeeding.
  assert _gcp_metadata.get_project_id_from_metadata() == 'retry-project'
  assert calls['ping'] == 2
  assert calls['get'] == 1


def test_is_enterprise_mode_enabled_does_not_apply_metadata_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """is_enterprise_mode_enabled is a pure env check and does not mutate env."""
  from google.adk.utils.env_utils import is_enterprise_mode_enabled

  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  assert is_enterprise_mode_enabled() is False
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in os.environ


def test_get_gcp_client_defaults_skips_defaults_when_gemini_api_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """GEMINI_API_KEY auth must not force enterprise mode or set project."""
  monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults(client_kwargs={'enterprise': True})

  assert defaults == {}
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in os.environ


def test_get_gcp_client_defaults_rechecks_gate_on_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Gate is re-evaluated on every call, reusing cached project ID on GCP."""
  calls = {'n': 0}

  def fake_project() -> str:
    calls['n'] += 1
    return 'meta-project'

  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', fake_project
  )

  # First call with API key present: gate evaluates to False, defaults skipped.
  monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
  assert get_gcp_client_defaults(client_kwargs={'enterprise': True}) == {}
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert calls['n'] == 0

  # Second call without API key: gate re-evaluates and fills missing defaults.
  monkeypatch.delenv('GOOGLE_API_KEY')
  defaults = get_gcp_client_defaults(client_kwargs={'enterprise': True})
  assert defaults == {
      'project': 'meta-project',
  }
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert calls['n'] == 1


def test_get_project_id_from_metadata_caches_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Project ID from metadata server is cached across multiple calls."""
  from google.auth.compute_engine import _metadata

  calls = {'n': 0}

  def fake_get(*args, **kwargs):
    calls['n'] += 1
    return 'cached-project'

  monkeypatch.setattr(_metadata, 'ping', lambda *a, **kw: True)
  monkeypatch.setattr(_metadata, 'get', fake_get)
  assert _gcp_metadata.get_project_id_from_metadata() == 'cached-project'
  assert _gcp_metadata.get_project_id_from_metadata() == 'cached-project'
  assert calls['n'] == 1


def test_get_project_id_from_metadata_empty_project_latches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Empty project from metadata server latches and does not re-probe."""
  from google.auth.compute_engine import _metadata

  calls = {'ping': 0, 'get': 0}

  def fake_ping(*args, **kwargs):
    calls['ping'] += 1
    return True

  def fake_get(*args, **kwargs):
    calls['get'] += 1
    return None

  monkeypatch.setattr(_metadata, 'ping', fake_ping)
  monkeypatch.setattr(_metadata, 'get', fake_get)

  assert _gcp_metadata.get_project_id_from_metadata() is None
  assert _gcp_metadata.get_project_id_from_metadata() is None
  assert calls['ping'] == 1
  assert calls['get'] == 1


def test_gcp_defaults_isolation_between_keyless_and_api_key_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Keyless model stays Gemini API while enterprise model gets GCP metadata defaults."""
  from unittest import mock

  from google.adk.models import Gemini

  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  keyless_model = Gemini(model='gemini-2.5-flash')
  enterprise_model = Gemini(
      model='gemini-2.5-flash', client_kwargs={'enterprise': True}
  )
  api_key_model = Gemini(
      model='gemini-2.5-flash', client_kwargs={'api_key': 'ai-studio-key'}
  )

  captured_clients = []

  def fake_client(**kwargs):
    client = mock.MagicMock()
    client.vertexai = kwargs.get('enterprise', False) or kwargs.get(
        'vertexai', False
    )
    captured_clients.append((kwargs, client))
    return client

  with mock.patch('google.genai.Client', side_effect=fake_client):
    _ = keyless_model.api_client
    _ = enterprise_model.api_client
    _ = api_key_model.api_client

  assert len(captured_clients) == 3
  keyless_kwargs, keyless_client = captured_clients[0]
  enterprise_kwargs, enterprise_client = captured_clients[1]
  api_key_kwargs, api_key_client = captured_clients[2]

  # Keyless model does NOT flip to Vertex on GCP when enterprise mode is unset.
  assert 'enterprise' not in keyless_kwargs
  assert 'project' not in keyless_kwargs
  assert keyless_client.vertexai is False

  # Explicit enterprise model gets project default and resolves to Vertex.
  assert enterprise_kwargs.get('enterprise') is True
  assert enterprise_kwargs.get('project') == 'meta-project'
  assert enterprise_client.vertexai is True

  # API key model skips GCP defaults.
  assert 'enterprise' not in api_key_kwargs
  assert 'project' not in api_key_kwargs
  assert api_key_kwargs.get('api_key') == 'ai-studio-key'
  assert api_key_client.vertexai is False
  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in os.environ


def test_get_gcp_client_defaults_skips_project_when_enterprise_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Explicit enterprise flag (e.g. GOOGLE_GENAI_USE_ENTERPRISE=0) skips project default."""
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '0')
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults()

  assert defaults == {}
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ


def test_get_gcp_client_defaults_fills_project_when_enterprise_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Explicit enterprise flag without project fills project default from metadata."""
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '1')
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults()

  assert defaults == {'project': 'meta-project'}


def test_get_gcp_client_defaults_skips_project_when_client_kwargs_enterprise_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Explicit enterprise=False in client_kwargs skips project default."""
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults(client_kwargs={'enterprise': False})

  assert defaults == {}


def test_get_gcp_client_defaults_fills_project_when_client_kwargs_enterprise_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Explicit enterprise=True in client_kwargs without project fills project default."""
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults(client_kwargs={'enterprise': True})

  assert defaults == {'project': 'meta-project'}


def test_local_dev_with_user_adc_does_not_trigger_gcp_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
  """Developer machine with user ADC does not default to Vertex AI when off GCP."""
  from unittest import mock

  from google.auth import _cloud_sdk
  from google.auth.compute_engine import _metadata

  adc_file = tmp_path / 'application_default_credentials.json'
  adc_file.write_text('{"type": "authorized_user"}')
  monkeypatch.setattr(
      _cloud_sdk,
      'get_application_default_credentials_path',
      lambda: str(adc_file),
  )
  ping_mock = mock.MagicMock(return_value=False)
  monkeypatch.setattr(_metadata, 'ping', ping_mock)
  defaults = get_gcp_client_defaults(client_kwargs={'enterprise': True})
  assert defaults == {}
  ping_mock.assert_not_called()
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in os.environ


def test_get_project_id_from_metadata_whitespace_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Whitespace project ID from metadata server returns None instead of empty string."""
  from google.auth.compute_engine import _metadata

  monkeypatch.setattr(_metadata, 'ping', lambda *a, **kw: True)
  monkeypatch.setattr(_metadata, 'get', lambda *a, **kw: '   \n')

  assert get_gcp_client_defaults(client_kwargs={'enterprise': True}) == {}
  assert _gcp_metadata.get_project_id_from_metadata() is None


def test_get_gcp_client_defaults_skips_defaults_when_credentials_in_client_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Credentials in client_kwargs must not inject host metadata project."""
  from unittest import mock

  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  mock_creds = mock.MagicMock()
  defaults = get_gcp_client_defaults(
      client_kwargs={'enterprise': True, 'credentials': mock_creds}
  )

  assert defaults == {}
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'project' not in defaults


def test_get_gcp_client_defaults_skips_defaults_when_application_credentials_in_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """GOOGLE_APPLICATION_CREDENTIALS must not be overridden by host metadata project."""
  monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', '/path/to/key.json')
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults(client_kwargs={'enterprise': True})

  assert defaults == {}
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'project' not in defaults


def test_get_gcp_client_defaults_skips_defaults_when_user_adc_file_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
  """gcloud ADC file on disk prevents metadata project from overriding ADC project."""
  from google.auth import _cloud_sdk

  adc_file = tmp_path / 'application_default_credentials.json'
  adc_file.write_text('{"type": "authorized_user"}')
  monkeypatch.setattr(
      _cloud_sdk,
      'get_application_default_credentials_path',
      lambda: str(adc_file),
  )
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )

  defaults = get_gcp_client_defaults(client_kwargs={'enterprise': True})

  assert defaults == {}
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'project' not in defaults
