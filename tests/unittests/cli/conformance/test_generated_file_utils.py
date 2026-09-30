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

"""Tests for conformance generated-file loading helpers."""

from __future__ import annotations

import textwrap

from google.adk.agents.run_config import StreamingMode
from google.adk.cli.conformance._generated_file_utils import load_recorded_session
from google.adk.cli.conformance._generated_file_utils import load_test_case
import pydantic
import pytest

_SESSION_YAML = """\
id: {session_id}
appName: {app_name}
userId: u1
state: {{}}
events: []
"""


def _write_spec(test_case_dir, body: str) -> None:
  (test_case_dir / 'spec.yaml').write_text(textwrap.dedent(body))


def test_load_test_case_parses_spec_and_applies_declared_defaults(tmp_path):
  _write_spec(
      tmp_path,
      """\
      description: checks the dice agent
      agent: dice_agent
      user_messages:
        - text: roll a die
        - text: roll again
          state_delta:
            rolls: 1
      """,
  )

  spec = load_test_case(tmp_path)

  assert spec.description == 'checks the dice agent'
  assert spec.agent == 'dice_agent'
  # Omitted field falls back to its documented empty default.
  assert spec.initial_state == {}
  assert [m.text for m in spec.user_messages] == ['roll a die', 'roll again']
  assert spec.user_messages[0].state_delta is None
  assert spec.user_messages[1].state_delta == {'rolls': 1}


def test_load_test_case_rejects_unknown_spec_field(tmp_path):
  """TestSpec forbids extras so a typo in a hand-written spec is not silent."""
  _write_spec(
      tmp_path,
      """\
      description: d
      agent: a
      user_mesages:
        - text: typo in the key above
      """,
  )

  with pytest.raises(pydantic.ValidationError):
    load_test_case(tmp_path)


def test_load_test_case_rejects_spec_missing_required_agent(tmp_path):
  _write_spec(tmp_path, 'description: no agent named\n')

  with pytest.raises(pydantic.ValidationError):
    load_test_case(tmp_path)


def test_load_recorded_session_picks_file_matching_streaming_mode(tmp_path):
  (tmp_path / 'generated-session.yaml').write_text(
      _SESSION_YAML.format(session_id='non-streaming', app_name='app_none')
  )
  (tmp_path / 'generated-session-sse.yaml').write_text(
      _SESSION_YAML.format(session_id='streaming', app_name='app_sse')
  )

  none_session = load_recorded_session(tmp_path, StreamingMode.NONE)
  sse_session = load_recorded_session(tmp_path, StreamingMode.SSE)

  assert none_session.id == 'non-streaming'
  assert none_session.app_name == 'app_none'
  assert sse_session.id == 'streaming'
  assert sse_session.app_name == 'app_sse'


def test_load_recorded_session_returns_none_when_file_absent(tmp_path):
  assert load_recorded_session(tmp_path, StreamingMode.NONE) is None
  assert load_recorded_session(tmp_path, StreamingMode.SSE) is None


def test_load_recorded_session_returns_none_quietly_for_empty_file(
    tmp_path, capsys
):
  """An empty recording is "nothing recorded yet", not a parse failure."""
  (tmp_path / 'generated-session.yaml').write_text('')

  assert load_recorded_session(tmp_path, StreamingMode.NONE) is None
  assert capsys.readouterr().err == ''


def test_load_recorded_session_returns_none_on_unparseable_session(
    tmp_path, capsys
):
  """A corrupt recording is reported, not raised, so replay can report it."""
  (tmp_path / 'generated-session.yaml').write_text(
      'id: only-an-id\nappName: app\n'
  )

  assert load_recorded_session(tmp_path, StreamingMode.NONE) is None
  assert 'Failed to parse session data' in capsys.readouterr().err


def test_load_recorded_session_rejects_unsupported_streaming_mode(tmp_path):
  with pytest.raises(ValueError, match='Unsupported streaming mode'):
    load_recorded_session(tmp_path, StreamingMode.BIDI)
