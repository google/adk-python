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

"""Tracks Antigravity conversation resumption state in the local save_dir.

The Antigravity local harness persists conversation state to a ``traj-*`` file
in its ``save_dir`` and rehydrates it when a matching ``conversation_id`` is
passed on a later turn. This module adds the small bit of bookkeeping the
wrapper
needs around that file:

- detecting whether a prior trajectory exists (so resumption can be requested),
- persisting the *resume step index* next to it. On resume the harness replays
  the whole trajectory through its step stream; this index (the highest harness
  ``step_index`` already emitted to ADK) lets the wrapper skip those replayed
  steps so prior turns are not re-recorded.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger('google_adk.' + __name__)


def trajectory_path(save_dir: str, conversation_id: str) -> str:
  """Returns the harness trajectory file path for a conversation."""
  return os.path.join(save_dir, f'traj-{conversation_id}')


def _resume_index_path(save_dir: str, conversation_id: str) -> str:
  return os.path.join(save_dir, f'traj-{conversation_id}.resume')


def has_trajectory(save_dir: str, conversation_id: str) -> bool:
  """Returns True if a prior trajectory exists for this conversation."""
  return os.path.exists(trajectory_path(save_dir, conversation_id))


def rename_trajectory(
    save_dir: str, conversation_id: str, harness_conversation_id: str
) -> None:
  """Renames a fresh trajectory from the harness's id to our deterministic id.

  On a fresh turn the harness assigns a random ``conversation_id`` and writes
  ``traj-<random>``. Renaming it to ``traj-<conversation_id>`` lets later turns
  locate and resume it deterministically from the ADK session id.
  """
  if not harness_conversation_id or harness_conversation_id == conversation_id:
    return
  src = trajectory_path(save_dir, harness_conversation_id)
  dst = trajectory_path(save_dir, conversation_id)
  if os.path.exists(src):
    os.replace(src, dst)


def load_resume_step_index(save_dir: str, conversation_id: str) -> int:
  """Returns the resume step index, or -1 if absent or unreadable.

  This is the highest harness ``step_index`` emitted in earlier turns; replayed
  steps at or below it are skipped on resume.
  """
  path = _resume_index_path(save_dir, conversation_id)
  if not os.path.exists(path):
    return -1
  try:
    with open(path, encoding='utf-8') as f:
      return int(f.read().strip())
  except (OSError, ValueError):
    logger.warning(
        '[ADK] Corrupt Antigravity resume step index; treating as fresh.'
    )
    return -1


def save_resume_step_index(
    save_dir: str, conversation_id: str, resume_step_index: int
) -> None:
  """Persists the resume step index next to the trajectory file."""
  with open(
      _resume_index_path(save_dir, conversation_id), 'w', encoding='utf-8'
  ) as f:
    f.write(str(resume_step_index))
