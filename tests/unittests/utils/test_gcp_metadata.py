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
from urllib.error import URLError

from google.adk.utils import _gcp_metadata
from google.adk.utils.env_utils import apply_gcp_runtime_defaults
import pytest


@pytest.fixture(autouse=True)
def _clear_gcp_env(monkeypatch: pytest.MonkeyPatch) -> None:
  for name in (
      'GOOGLE_CLOUD_PROJECT',
      'GOOGLE_CLOUD_LOCATION',
      'GOOGLE_GENAI_USE_ENTERPRISE',
      'GOOGLE_GENAI_USE_VERTEXAI',
      'GOOGLE_API_KEY',
      'GOOGLE_GENAI_API_KEY',
  ):
    monkeypatch.delenv(name, raising=False)
  # Reset the process-level off-GCP cache between tests.
  monkeypatch.setattr(_gcp_metadata, '_off_gcp_cached', False)


def test_get_location_from_metadata_parses_cloud_run_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Cloud Run region paths resolve to the bare region name."""

  def fake_get(path: str) -> str | None:
    if path == 'instance/region':
      return 'projects/123456789/regions/europe-west1'
    return None

  monkeypatch.setattr(_gcp_metadata, '_metadata_get', fake_get)
  assert _gcp_metadata.get_location_from_metadata() == 'europe-west1'


def test_get_location_from_metadata_derives_region_from_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """GCE zone paths fall back to the parent region."""

  def fake_get(path: str) -> str | None:
    if path == 'instance/zone':
      return 'projects/123456789/zones/us-central1-a'
    return None

  monkeypatch.setattr(_gcp_metadata, '_metadata_get', fake_get)
  assert _gcp_metadata.get_location_from_metadata() == 'us-central1'


def test_apply_gcp_runtime_defaults_noop_off_gcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Offline / local environments leave env vars untouched."""
  calls = {'n': 0}

  def fake_project() -> None:
    calls['n'] += 1
    return None

  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', fake_project
  )

  assert apply_gcp_runtime_defaults() == {}
  assert apply_gcp_runtime_defaults() == {}
  # Second call should hit the off-GCP cache and not re-probe.
  assert calls['n'] == 1
  assert 'GOOGLE_CLOUD_PROJECT' not in os.environ
  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in os.environ


def test_apply_gcp_runtime_defaults_fills_unset_values_on_gcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """On GCP, missing project/location/enterprise flags are filled."""
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )
  monkeypatch.setattr(
      _gcp_metadata, 'get_location_from_metadata', lambda: 'asia-northeast1'
  )

  applied = apply_gcp_runtime_defaults()

  assert applied == {
      'GOOGLE_CLOUD_PROJECT': 'meta-project',
      'GOOGLE_CLOUD_LOCATION': 'asia-northeast1',
      'GOOGLE_GENAI_USE_ENTERPRISE': 'true',
  }
  assert os.environ['GOOGLE_CLOUD_PROJECT'] == 'meta-project'
  assert os.environ['GOOGLE_CLOUD_LOCATION'] == 'asia-northeast1'
  assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == 'true'


def test_apply_gcp_runtime_defaults_never_overrides_explicit_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Shell / .env values win over metadata defaults."""
  monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'explicit-project')
  monkeypatch.setenv('GOOGLE_CLOUD_LOCATION', 'us-east1')
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', 'false')
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )
  monkeypatch.setattr(
      _gcp_metadata, 'get_location_from_metadata', lambda: 'asia-northeast1'
  )

  applied = apply_gcp_runtime_defaults()

  assert applied == {}
  assert os.environ['GOOGLE_CLOUD_PROJECT'] == 'explicit-project'
  assert os.environ['GOOGLE_CLOUD_LOCATION'] == 'us-east1'
  assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == 'false'


def test_apply_gcp_runtime_defaults_skips_enterprise_when_api_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """API-key auth must not force enterprise/Vertex mode."""
  monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )
  monkeypatch.setattr(
      _gcp_metadata, 'get_location_from_metadata', lambda: 'us-central1'
  )

  applied = apply_gcp_runtime_defaults()

  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in applied
  assert 'GOOGLE_GENAI_USE_ENTERPRISE' not in os.environ
  assert applied['GOOGLE_CLOUD_PROJECT'] == 'meta-project'


def test_metadata_get_returns_none_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Metadata lookups fail soft when the server is unreachable."""

  def boom(*_args, **_kwargs):
    raise URLError('timed out')

  monkeypatch.setattr(_gcp_metadata, 'urlopen', boom)
  assert _gcp_metadata._metadata_get('project/project-id') is None


def test_load_dotenv_for_agent_applies_gcp_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
  """Agent dotenv loading applies metadata defaults after `.env` resolution."""
  from google.adk.cli.utils import envs

  agent_dir = tmp_path / 'my_agent'
  agent_dir.mkdir()
  (agent_dir / '.env').write_text('SOME_OTHER=1\n')

  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'dotenv-project'
  )
  monkeypatch.setattr(
      _gcp_metadata, 'get_location_from_metadata', lambda: 'us-west1'
  )
  # Clear lru_cache so explicit-env snapshot is re-taken for this test.
  envs._get_explicit_env_keys.cache_clear()

  envs.load_dotenv_for_agent('my_agent', str(tmp_path))

  assert os.environ['GOOGLE_CLOUD_PROJECT'] == 'dotenv-project'
  assert os.environ['GOOGLE_CLOUD_LOCATION'] == 'us-west1'
  assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == 'true'
  assert os.environ['SOME_OTHER'] == '1'


def test_is_enterprise_mode_enabled_uses_gcp_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Enterprise-mode checks apply metadata defaults on GCP when unset."""
  from google.adk.utils.env_utils import is_enterprise_mode_enabled

  monkeypatch.setattr(
      _gcp_metadata, 'get_project_id_from_metadata', lambda: 'meta-project'
  )
  monkeypatch.setattr(
      _gcp_metadata, 'get_location_from_metadata', lambda: 'us-central1'
  )

  assert is_enterprise_mode_enabled() is True
  assert os.environ['GOOGLE_CLOUD_PROJECT'] == 'meta-project'
  assert os.environ['GOOGLE_CLOUD_LOCATION'] == 'us-central1'
  assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == 'true'
