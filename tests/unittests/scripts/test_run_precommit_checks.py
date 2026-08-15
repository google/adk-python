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

"""Tests for the pre-commit hook runner."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

precommit = pytest.importorskip(
    'scripts.run_precommit_checks',
    reason='scripts/run_precommit_checks.py is not present in this checkout',
)


def _hook(hook_id: str) -> precommit.Hook:
  return precommit.Hook(hook_id=hook_id, files=None, exclude=None, args=[])


def _main(hooks: list[precommit.Hook], *, installed: bool) -> int:
  """Runs main() over `hooks` with every tool present or absent."""
  which = (
      (lambda tool: f'/usr/bin/{tool}') if installed else (lambda tool: None)
  )
  with (
      mock.patch.object(precommit, 'load_config', return_value=(hooks, None)),
      mock.patch.object(precommit, 'collect_files', return_value=['a.py']),
      mock.patch.object(precommit.shutil, 'which', side_effect=which),
      mock.patch.object(sys, 'argv', ['run_precommit_checks.py', '--check']),
  ):
    return precommit.main()


def test_missing_tool_does_not_report_a_pass(
    capsys: pytest.CaptureFixture[str],
) -> None:
  exit_code = _main([_hook('pyink')], installed=False)

  out = capsys.readouterr().out
  assert exit_code == 1
  assert 'All checks passed.' not in out
  assert 'NOT INSTALLED: pyink' in out


def test_missing_tool_does_not_run_the_hook() -> None:
  with mock.patch.object(precommit, 'run_standard_hook') as runner:
    _main([_hook('pyink')], installed=False)

  runner.assert_not_called()


def test_passing_hook_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
  with mock.patch.object(precommit, 'run_standard_hook', return_value=True):
    exit_code = _main([_hook('pyink')], installed=True)

  assert exit_code == 0
  assert 'All checks passed.' in capsys.readouterr().out


def test_failing_hook_exits_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
  with mock.patch.object(precommit, 'run_standard_hook', return_value=False):
    exit_code = _main([_hook('pyink')], installed=True)

  assert exit_code == 1
  assert 'FAILED: pyink' in capsys.readouterr().out


@pytest.mark.parametrize(
    'hook_id, expected',
    [
        ('pyink', 'pyink'),
        ('trailing-whitespace', 'trailing-whitespace-fixer'),
        ('addlicense', 'addlicense'),
        ('compliance-checks', None),
    ],
)
def test_required_tool(hook_id: str, expected: str | None) -> None:
  assert precommit.required_tool(_hook(hook_id)) == expected
