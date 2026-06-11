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

"""Tests for Antigravity trajectory resumption bookkeeping in save_dir.

Verifies trajectory detection, resume step index persistence, and renaming the
harness's randomly-named trajectory to a deterministic name.
"""

from __future__ import annotations

from google.adk.labs.antigravity import _trajectory_files


def test_has_trajectory_false_when_absent(tmp_path):
  """No trajectory file means no prior conversation to resume."""
  assert not _trajectory_files.has_trajectory(str(tmp_path), 'sess_agy')


def test_has_trajectory_true_when_present(tmp_path):
  """An existing traj file is detected for the conversation."""
  (tmp_path / 'traj-sess_agy').write_bytes(b'data')

  assert _trajectory_files.has_trajectory(str(tmp_path), 'sess_agy')


def test_load_resume_step_index_minus_one_when_absent(tmp_path):
  """Missing resume step index reads as -1 (fresh)."""
  assert (
      _trajectory_files.load_resume_step_index(str(tmp_path), 'sess_agy') == -1
  )


def test_resume_step_index_round_trips(tmp_path):
  """A saved resume step index reads back as the same value."""
  _trajectory_files.save_resume_step_index(str(tmp_path), 'sess_agy', 12)

  assert (
      _trajectory_files.load_resume_step_index(str(tmp_path), 'sess_agy') == 12
  )


def test_load_resume_step_index_minus_one_when_corrupt(tmp_path):
  """A non-integer resume step index is treated as fresh."""
  (tmp_path / 'traj-sess_agy.resume').write_text('not-an-int')

  assert (
      _trajectory_files.load_resume_step_index(str(tmp_path), 'sess_agy') == -1
  )


def test_rename_trajectory_to_conversation_id(tmp_path):
  """The harness's random trajectory is renamed to the deterministic name."""
  (tmp_path / 'traj-random123').write_bytes(b'data')

  _trajectory_files.rename_trajectory(str(tmp_path), 'sess_agy', 'random123')

  assert not (tmp_path / 'traj-random123').exists()
  assert (tmp_path / 'traj-sess_agy').read_bytes() == b'data'


def test_rename_trajectory_noop_when_already_named(tmp_path):
  """Renaming is a no-op when the harness id already matches."""
  (tmp_path / 'traj-sess_agy').write_bytes(b'data')

  _trajectory_files.rename_trajectory(str(tmp_path), 'sess_agy', 'sess_agy')

  assert (tmp_path / 'traj-sess_agy').read_bytes() == b'data'
