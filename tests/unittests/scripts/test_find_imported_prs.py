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

"""Unit tests for find_imported_prs.py."""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from scripts import find_imported_prs

_URL = 'https://github.com/google/adk-python/pull/'


def test_a_title_naming_another_pr_does_not_change_which_pr_is_imported() -> (
    None
):
  message = f'fix: thing Merges #9\n\nMerge {_URL}10\n\nPiperOrigin-RevId: 1\n'
  assert find_imported_prs.imported_pr(message) == 10


@pytest.mark.parametrize(
    'line',
    [
        f'see Merge {_URL}9',
        f'Merge {_URL}9 and more',
        f'Merge: {_URL}9',
        f'Merges {_URL}9',
        'Merge #9',
        f'merge {_URL}9',
        f'this merges {_URL}9 too',
    ],
)
def test_only_the_whole_import_line_names_a_pr(line: str) -> None:
  message = f'chore: a change\n\n{line}\n\nPiperOrigin-RevId: 1\n'
  assert find_imported_prs.imported_pr(message) is None


def test_a_second_import_line_leaves_the_pr_unknown() -> None:
  message = f'fix: thing\n\nMerge {_URL}10\n\nMerge {_URL}9\n'
  assert find_imported_prs.imported_pr(message) is None


def test_main_prints_each_imported_pr_for_the_workflow(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  event = {
      'commits': [
          {'id': 'a' * 40, 'message': f'fix: x Merges #9\n\nMerge {_URL}10\n'},
          {'id': 'b' * 40, 'message': 'chore: an internal change\n'},
      ]
  }
  path = tmp_path / 'event.json'
  path.write_text(json.dumps(event), encoding='utf-8')

  assert find_imported_prs.main(['find_imported_prs.py', str(path)]) == 0
  assert capsys.readouterr().out == (
      'prs=[{"prNumber": 10, "commitSha": "aaaaaaa"}]\n'
  )


def test_the_workflow_closes_only_the_prs_the_script_finds() -> None:
  root = pathlib.Path(find_imported_prs.__file__).resolve().parents[1]
  path = root / '.github' / 'workflows' / 'copybara-pr-handler.yml'
  workflow = yaml.safe_load(path.read_text(encoding='utf-8'))

  checkout, find, close = workflow['jobs']['close-imported-pr']['steps']

  assert checkout['uses'].startswith('actions/checkout@')
  assert checkout['with'] == {'persist-credentials': False}
  assert find['id'] == 'imported'
  assert find['if'] == "github.event_name == 'push'"
  assert find['run'] == (
      'python3 scripts/find_imported_prs.py "$GITHUB_EVENT_PATH" >>'
      ' "$GITHUB_OUTPUT"'
  )
  assert close['env'] == {'PRS_TO_CLOSE': '${{ steps.imported.outputs.prs }}'}
  script = close['with']['script']
  assert "JSON.parse(process.env.PRS_TO_CLOSE || '[]')" in script
  assert 'context.payload.commits' not in script
