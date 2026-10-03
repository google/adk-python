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

"""Speculative tool dispatch: run the read before the model finishes asking.

The model is told to answer by emitting a single directive line:

    TOOL_CALL: {"name": "read_file", "arguments": {"path": "<path>"}}

``SpeculativeRouterNode`` watches that line stream in. As soon as the (still
incomplete) JSON can be repaired into a plausible call, it dispatches the
``read_file`` target **immediately** -- overlapping the read with the rest of
generation -- then verifies against the finalized call: keep the result on a
match, cancel and re-read on a mismatch.

The target here is a read (idempotent, safe to cancel), which is the only kind of
work speculation is appropriate for. ``enable_uvloop()`` runs the overlapped work
on a libuv loop.
"""

import pathlib

from google.adk import Agent
from google.adk import enable_uvloop
from google.adk import Event
from google.adk.workflow import SpeculativeRouterNode

enable_uvloop()


def _path_of(payload: dict) -> str:
  return (payload or {}).get('arguments', {}).get('path', '')


def read_file(node_input: dict):
  """The speculatively-dispatched target: read a file (idempotent)."""
  path = _path_of(node_input)
  try:
    text = pathlib.Path(path).read_text()
    preview = text[:500]
    yield Event(output={'path': path, 'ok': True, 'preview': preview})
  except OSError as e:
    yield Event(output={'path': path, 'ok': False, 'error': str(e)})


reader = Agent(
    name='planner',
    instruction=(
        'The user will ask about a file. Respond with EXACTLY one line and'
        ' nothing else:\n'
        '  TOOL_CALL: {"name": "read_file", "arguments": {"path": "<path>"}}\n'
        'Use the most likely repository-relative path for what they asked.'
    ),
)


root_agent = SpeculativeRouterNode(
    name='speculative_read',
    agent=reader,
    target=read_file,
    # Only speculate once the path looks substantial enough to be worth a guess;
    # short prefixes are too likely to be revised.
    should_speculate=lambda payload: len(_path_of(payload)) >= 6,
    # Verify hit/miss on the resolved path.
    same=lambda a, b: _path_of(a) == _path_of(b),
    # Bound the speculative read so a bad guess can't hang the turn.
    timeout=30,
)
