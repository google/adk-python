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

"""Resolution of the ADK telemetry schema version from the environment."""

from __future__ import annotations

from typing import Optional

from google.adk.telemetry._schema_version import ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN
from google.adk.telemetry._schema_version import GOOGLE_CLOUD_AGENT_ENGINE_ID
from google.adk.telemetry._schema_version import resolve_schema_version
import pytest


def _set_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    opt_in: Optional[str] = None,
    agent_engine_id: Optional[str] = None,
) -> None:
  """Pins both inputs so an ambient env var cannot leak into the result."""
  for name, value in (
      (ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN, opt_in),
      (GOOGLE_CLOUD_AGENT_ENGINE_ID, agent_engine_id),
  ):
    if value is None:
      monkeypatch.delenv(name, raising=False)
    else:
      monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    'opt_in,expected',
    [
        ('1', 1),
        ('2', 2),
        # The env value is stripped before it is matched. Only version 2 is
        # exercised here: a stripped '1' is indistinguishable from the
        # legacy default, so it would pass even with the stripping removed.
        (' 2 ', 2),
        ('\n2\t', 2),
    ],
)
def test_resolve_schema_version_honors_recognized_opt_in(
    monkeypatch: pytest.MonkeyPatch, opt_in: str, expected: int
):
  """A recognized opt-in value selects that schema version verbatim."""
  _set_env(monkeypatch, opt_in=opt_in)

  assert resolve_schema_version() == expected


@pytest.mark.parametrize('opt_in', ['', ' ', '3', '0', 'two', 'v2'])
def test_resolve_schema_version_unrecognized_opt_in_falls_back_to_legacy(
    monkeypatch: pytest.MonkeyPatch, opt_in: str
):
  """Only '1' and '2' are recognized; anything else defers to the default."""
  _set_env(monkeypatch, opt_in=opt_in)

  assert resolve_schema_version() == 1


def test_resolve_schema_version_defaults_to_legacy_off_agent_engine(
    monkeypatch: pytest.MonkeyPatch,
):
  """Neither env var set: the documented default is the legacy schema."""
  _set_env(monkeypatch)

  assert resolve_schema_version() == 1


def test_resolve_schema_version_defaults_to_semconv_on_agent_engine(
    monkeypatch: pytest.MonkeyPatch,
):
  """Agent Engine is detected by the presence of its id env var."""
  _set_env(monkeypatch, agent_engine_id='some-agent-engine')

  assert resolve_schema_version() == 2


def test_resolve_schema_version_empty_agent_engine_id_is_not_agent_engine(
    monkeypatch: pytest.MonkeyPatch,
):
  """An id set to the empty string carries no deployment, so it must not flip

  the default -- otherwise a blank value in a deployment template silently
  changes the emitted telemetry format.
  """
  _set_env(monkeypatch, agent_engine_id='')

  assert resolve_schema_version() == 1


@pytest.mark.parametrize('opt_in,expected', [('1', 1), ('2', 2)])
def test_resolve_schema_version_opt_in_overrides_agent_engine_default(
    monkeypatch: pytest.MonkeyPatch, opt_in: str, expected: int
):
  """The opt-in outranks the Agent Engine default, including pinning back to

  the legacy schema on Agent Engine.
  """
  _set_env(monkeypatch, opt_in=opt_in, agent_engine_id='some-agent-engine')

  assert resolve_schema_version() == expected


def test_resolve_schema_version_unrecognized_opt_in_keeps_agent_engine_default(
    monkeypatch: pytest.MonkeyPatch,
):
  """An unrecognized opt-in is ignored, not treated as an opt-out."""
  _set_env(monkeypatch, opt_in='bogus', agent_engine_id='some-agent-engine')

  assert resolve_schema_version() == 2
