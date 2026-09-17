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

"""Unit tests for check_new_py_files.py."""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

from scripts import check_new_py_files


def test_is_exempt_from_unit_guide() -> None:
  assert check_new_py_files.is_exempt_from_unit_guide(
      '__init__.py', '__init__.py'
  )
  assert check_new_py_files.is_exempt_from_unit_guide(
      'cli/runner.py', 'runner.py'
  )
  assert check_new_py_files.is_exempt_from_unit_guide(
      'sub/cli/runner.py', 'runner.py'
  )
  assert check_new_py_files.is_exempt_from_unit_guide(
      'tools/utils/helpers.py', 'helpers.py'
  )
  assert check_new_py_files.is_exempt_from_unit_guide(
      'agents/_agent_utils.py', '_agent_utils.py'
  )
  assert check_new_py_files.is_exempt_from_unit_guide(
      'agents/_agent_types.py', '_agent_types.py'
  )
  assert check_new_py_files.is_exempt_from_unit_guide(
      'agents/_agent_errors.py', '_agent_errors.py'
  )
  assert check_new_py_files.is_exempt_from_unit_guide(
      'agents/_agent_constants.py', '_agent_constants.py'
  )
  assert check_new_py_files.is_exempt_from_unit_guide(
      'agents/_agent_helpers.py', '_agent_helpers.py'
  )

  # Non-exempt files
  assert not check_new_py_files.is_exempt_from_unit_guide(
      'agents/_custom_agent.py', '_custom_agent.py'
  )
  assert not check_new_py_files.is_exempt_from_unit_guide(
      'flows/_workflow.py', '_workflow.py'
  )


def test_has_no_unit_guide_tag(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv('NO_UNIT_GUIDE', raising=False)
  monkeypatch.delenv('SKIP_UNIT_GUIDE', raising=False)

  assert not check_new_py_files.has_no_unit_guide_tag('Initial commit')
  assert check_new_py_files.has_no_unit_guide_tag(
      'Add agent\nNO_UNIT_GUIDE=internal'
  )
  assert check_new_py_files.has_no_unit_guide_tag(
      'Add agent\nSKIP_UNIT_GUIDE=reason'
  )

  monkeypatch.setenv('NO_UNIT_GUIDE', '1')
  assert check_new_py_files.has_no_unit_guide_tag('Initial commit')


def test_check_files_prefix_violation(tmp_path: pathlib.Path) -> None:
  # Missing '_' prefix
  files = [('src/google/adk/agents/agent.py', 'agents/agent.py', 'agent.py')]
  prefix_errs, guide_errs = check_new_py_files.check_files(
      files,
      repo_root=str(tmp_path),
      skip_unit_guide=True,
  )
  assert len(prefix_errs) == 1
  assert (
      "New Python file 'src/google/adk/agents/agent.py' must have a '_'"
      in prefix_errs[0]
  )
  assert len(guide_errs) == 0


def test_check_files_guide_violation(tmp_path: pathlib.Path) -> None:
  # Proper '_' prefix, but missing unit guide
  files = [('src/google/adk/agents/_agent.py', 'agents/_agent.py', '_agent.py')]
  prefix_errs, guide_errs = check_new_py_files.check_files(
      files,
      repo_root=str(tmp_path),
      commit_msg='clean commit',
  )
  assert len(prefix_errs) == 0
  assert len(guide_errs) == 1
  assert 'requires a unit guide in docs/guides/' in guide_errs[0]


def test_check_files_guide_found(tmp_path: pathlib.Path) -> None:
  guide_file = tmp_path / 'docs' / 'guides' / 'agents' / 'agent.md'
  guide_file.parent.mkdir(parents=True, exist_ok=True)
  guide_file.write_text('# Agent Guide', encoding='utf-8')

  files = [('src/google/adk/agents/_agent.py', 'agents/_agent.py', '_agent.py')]
  prefix_errs, guide_errs = check_new_py_files.check_files(
      files,
      repo_root=str(tmp_path),
      commit_msg='clean commit',
  )
  assert len(prefix_errs) == 0
  assert len(guide_errs) == 0


def test_guide_name_strips_only_one_underscore(tmp_path: pathlib.Path) -> None:
  """'__thing.py' documents '_thing', not 'thing'.

  The shell implementation this replaced used `${name%.py}` with a single
  `#_`, so stripping every leading underscore would quietly move where a
  dunder-ish private file is expected to be documented.
  """
  guide_file = tmp_path / 'docs' / 'guides' / 'agents' / '_thing.md'
  guide_file.parent.mkdir(parents=True, exist_ok=True)
  guide_file.write_text('# Guide', encoding='utf-8')

  files = [(
      'src/google/adk/agents/__thing.py',
      'agents/__thing.py',
      '__thing.py',
  )]
  prefix_errs, guide_errs = check_new_py_files.check_files(
      files,
      repo_root=str(tmp_path),
      commit_msg='clean commit',
  )
  assert not prefix_errs
  assert not guide_errs

  # And the name it suggests when the guide is absent is '_thing' too.
  guide_file.unlink()
  _, guide_errs = check_new_py_files.check_files(
      files,
      repo_root=str(tmp_path),
      commit_msg='clean commit',
  )
  assert len(guide_errs) == 1
  assert 'agents/_thing' in guide_errs[0]


def test_excluded_dirs_are_anchored_at_the_package_root(
    tmp_path: pathlib.Path,
) -> None:
  """A nested 'tests' directory holds source, so it must still be checked.

  The shell implementation compared against `$ADK_REAL_ROOT/tests`, so only a
  top-level directory was excluded.
  """
  adk_root = tmp_path / 'src' / 'google' / 'adk'
  for rel in ('tests/_top.py', 'agents/tests/_nested.py'):
    path = adk_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('', encoding='utf-8')

  results = check_new_py_files._normalize_and_filter_files(
      [
          str(adk_root / 'tests' / '_top.py'),
          str(adk_root / 'agents' / 'tests' / '_nested.py'),
      ],
      repo_root=str(tmp_path),
  )

  assert [rel for _, rel, _ in results] == ['agents/tests/_nested.py']


def test_baseline_diff_detection(tmp_path: pathlib.Path) -> None:
  baseline_dir = tmp_path / 'baseline'
  new_dir = tmp_path / 'new'

  (baseline_dir / 'src' / 'google' / 'adk').mkdir(parents=True)
  (new_dir / 'src' / 'google' / 'adk' / 'agents').mkdir(parents=True)

  (baseline_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'src' / 'google' / 'adk' / 'agents' / '_agent.py').write_text(
      '', encoding='utf-8'
  )

  added = check_new_py_files.added_py_files_from_baseline(
      str(new_dir), str(baseline_dir)
  )
  assert added == {'src/google/adk/agents/_agent.py'}


def test_main_baseline_dir_violations(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  baseline_dir = tmp_path / 'baseline'
  new_dir = tmp_path / 'new'

  (baseline_dir / 'src' / 'google' / 'adk').mkdir(parents=True)
  (new_dir / 'src' / 'google' / 'adk' / 'agents').mkdir(parents=True)

  (baseline_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  # Invalid: no '_' prefix and no unit guide
  (new_dir / 'src' / 'google' / 'adk' / 'agents' / 'agent.py').write_text(
      '', encoding='utf-8'
  )

  exit_code = check_new_py_files.main([
      '--baseline-dir',
      str(baseline_dir),
      '--new-dir',
      str(new_dir),
  ])
  assert exit_code == 1
  err = capsys.readouterr().err
  assert "must have a '_' prefix" in err
  assert 'requires a unit guide in docs/guides/' in err


def test_main_baseline_dir_clean(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  baseline_dir = tmp_path / 'baseline'
  new_dir = tmp_path / 'new'

  (baseline_dir / 'src' / 'google' / 'adk').mkdir(parents=True)
  (new_dir / 'src' / 'google' / 'adk' / 'agents').mkdir(parents=True)
  (new_dir / 'docs' / 'guides' / 'agents').mkdir(parents=True)

  (baseline_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'src' / 'google' / 'adk' / 'agents' / '_agent.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'docs' / 'guides' / 'agents' / 'agent.md').write_text(
      '# Guide', encoding='utf-8'
  )

  exit_code = check_new_py_files.main([
      '--baseline-dir',
      str(baseline_dir),
      '--new-dir',
      str(new_dir),
  ])
  assert exit_code == 0
  err = capsys.readouterr().err
  assert err == ''


def test_main_baseline_dir_with_commit_msg_tag(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  baseline_dir = tmp_path / 'baseline'
  new_dir = tmp_path / 'new'

  (baseline_dir / 'src' / 'google' / 'adk').mkdir(parents=True)
  (new_dir / 'src' / 'google' / 'adk' / 'agents').mkdir(parents=True)

  (baseline_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  # Private file without unit guide
  (new_dir / 'src' / 'google' / 'adk' / 'agents' / '_agent.py').write_text(
      '', encoding='utf-8'
  )

  # Mock get_commit_message to return NO_UNIT_GUIDE tag
  monkeypatch.setattr(
      check_new_py_files,
      'get_commit_message',
      lambda root: 'Add agent\nNO_UNIT_GUIDE=helper module',
  )

  exit_code = check_new_py_files.main([
      '--baseline-dir',
      str(baseline_dir),
      '--new-dir',
      str(new_dir),
  ])
  assert exit_code == 0
  assert capsys.readouterr().err == ''


def test_main_baseline_dir_env_tag_waives_without_a_commit_message(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  """NO_UNIT_GUIDE works, and is advertised, where there is no commit message.

  Baseline mode can run against an exported tree with no VCS, where
  `get_commit_message` returns '', so the commit-message tag cannot be the
  only remedy the violation text offers.
  """
  baseline_dir = tmp_path / 'baseline'
  new_dir = tmp_path / 'new'

  (baseline_dir / 'src' / 'google' / 'adk').mkdir(parents=True)
  (new_dir / 'src' / 'google' / 'adk' / 'agents').mkdir(parents=True)

  (baseline_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'src' / 'google' / 'adk' / 'agents' / '_agent.py').write_text(
      '', encoding='utf-8'
  )

  # No VCS to read a commit message from.
  monkeypatch.setattr(check_new_py_files, 'get_commit_message', lambda root: '')
  monkeypatch.delenv('NO_UNIT_GUIDE', raising=False)
  monkeypatch.delenv('SKIP_UNIT_GUIDE', raising=False)

  argv = ['--baseline-dir', str(baseline_dir), '--new-dir', str(new_dir)]

  assert check_new_py_files.main(argv) == 1
  err = capsys.readouterr().err
  assert 'requires a unit guide in docs/guides/' in err
  # The remedy offered has to be one that works here.
  assert 'NO_UNIT_GUIDE' in err
  assert 'in the environment' in err

  monkeypatch.setenv('NO_UNIT_GUIDE', 'helper module')
  assert check_new_py_files.main(argv) == 0
  assert capsys.readouterr().err == ''


def test_sh_forwarder_execution(tmp_path: pathlib.Path) -> None:
  baseline_dir = tmp_path / 'baseline'
  new_dir = tmp_path / 'new'

  (baseline_dir / 'src' / 'google' / 'adk').mkdir(parents=True)
  (new_dir / 'src' / 'google' / 'adk' / 'agents').mkdir(parents=True)
  (new_dir / 'docs' / 'guides' / 'agents').mkdir(parents=True)

  (baseline_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'src' / 'google' / 'adk' / 'agents' / '_agent.py').write_text(
      '', encoding='utf-8'
  )
  (new_dir / 'docs' / 'guides' / 'agents' / 'agent.md').write_text(
      '# Guide', encoding='utf-8'
  )

  script_path = (
      pathlib.Path(check_new_py_files.__file__).resolve().parent
      / 'check_new_py_files.sh'
  )

  proc = subprocess.run(
      [
          'bash',
          str(script_path),
          '--baseline-dir',
          str(baseline_dir),
          '--new-dir',
          str(new_dir),
      ],
      capture_output=True,
      text=True,
  )
  assert proc.returncode == 0
  assert proc.stderr == ''


def test_symlinked_layout_normalization(tmp_path: pathlib.Path) -> None:
  # Simulate symlinked layout where open_source_workspace/src/google/adk/__init__.py
  # is a symlink pointing to the real upstream package root.
  upstream_adk = tmp_path / 'repo' / 'third_party' / 'adk'
  upstream_adk.mkdir(parents=True)
  (upstream_adk / '__init__.py').write_text('', encoding='utf-8')

  oss_workspace = upstream_adk / 'open_source_workspace'
  oss_src_adk = oss_workspace / 'src' / 'google' / 'adk'
  oss_src_adk.mkdir(parents=True)
  # Symlink __init__.py pointing back to upstream_adk/__init__.py
  (oss_src_adk / '__init__.py').symlink_to(upstream_adk / '__init__.py')

  # A file added in upstream package tree
  added_file = str(upstream_adk / 'agents' / '_agent.py')
  results = check_new_py_files._normalize_and_filter_files(
      [added_file], repo_root=str(oss_workspace)
  )
  assert len(results) == 1
  display_path, rel_to_adk, filename = results[0]
  assert rel_to_adk == 'agents/_agent.py'
  assert filename == '_agent.py'


def test_get_vcs_added_files_git(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'git' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if 'rev-parse' in cmd:
      return 0, 'true'
    if '--cached' in cmd:
      return 0, 'src/google/adk/agents/_staged.py'
    return 0, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  added = check_new_py_files.get_vcs_added_files('.')
  assert added == {'src/google/adk/agents/_staged.py'}


def test_get_vcs_added_files_git_head_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'git' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if 'rev-parse' in cmd:
      return 0, 'true'
    if '--cached' in cmd:
      return 0, ''
    if 'HEAD~1..HEAD' in cmd:
      return 0, 'src/google/adk/agents/_committed.py'
    return 0, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  added = check_new_py_files.get_vcs_added_files('.')
  assert added == {'src/google/adk/agents/_committed.py'}


def test_get_vcs_added_files_jj(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'jj' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['jj', 'root']:
      return 0, '/workspace'
    if cmd == ['jj', 'diff', '--summary']:
      return 0, 'A src/google/adk/agents/_jj_agent.py\nM existing.py'
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  added = check_new_py_files.get_vcs_added_files('.')
  assert added == {'/workspace/src/google/adk/agents/_jj_agent.py'}


def test_get_vcs_added_files_hg(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'hg' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['hg', 'root']:
      return 0, '/workspace'
    if cmd == ['hg', 'status', '--added', '--no-status']:
      return 0, 'src/google/adk/agents/_hg_agent.py'
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  added = check_new_py_files.get_vcs_added_files('.')
  assert added == {'/workspace/src/google/adk/agents/_hg_agent.py'}


def test_get_vcs_added_files_g4(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'g4' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['g4', 'info']:
      return 0, 'Server: ...'
    if cmd == ['g4', 'opened']:
      return (
          0,
          (
              '//depot/mirror/src/google/adk/agents/_g4_agent.py#1'
              ' - add default change (text)'
          ),
      )
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  added = check_new_py_files.get_vcs_added_files('.')
  assert added == {'//depot/mirror/src/google/adk/agents/_g4_agent.py'}


def test_get_vcs_added_files_p4(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'p4' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['p4', 'info']:
      return 0, 'Server: ...'
    if cmd == ['p4', 'opened']:
      return (
          0,
          (
              '//depot/mirror/src/google/adk/agents/_p4_agent.py#1'
              ' - add default change (text)'
          ),
      )
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  added = check_new_py_files.get_vcs_added_files('.')
  assert added == {'//depot/mirror/src/google/adk/agents/_p4_agent.py'}


def test_get_vcs_added_files_none_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(check_new_py_files.shutil, 'which', lambda _: None)
  added = check_new_py_files.get_vcs_added_files('.')
  assert added is None


def test_get_commit_message_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'git' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if 'rev-parse' in cmd and '--is-inside-work-tree' in cmd:
      return 0, 'true'
    if 'rev-parse' in cmd and '--git-dir' in cmd:
      return 0, str(tmp_path / '.git')
    if 'log' in cmd:
      return 0, 'Git Commit Message'
    return 0, ''

  (tmp_path / '.git').mkdir(parents=True)
  (tmp_path / '.git' / 'COMMIT_EDITMSG').write_text(
      'NO_UNIT_GUIDE=1', encoding='utf-8'
  )

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  msg = check_new_py_files.get_commit_message(str(tmp_path))
  assert 'Git Commit Message' in msg
  assert 'NO_UNIT_GUIDE=1' in msg


def test_get_commit_message_jj(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'jj' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['jj', 'root']:
      return 0, str(tmp_path)
    if 'jj' in cmd and 'log' in cmd:
      return 0, 'JJ Description'
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  msg = check_new_py_files.get_commit_message(str(tmp_path))
  assert msg == 'JJ Description'


def test_get_commit_message_hg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'hg' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['hg', 'root']:
      return 0, str(tmp_path)
    if 'hg' in cmd and 'log' in cmd:
      return 0, 'HG Description'
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  msg = check_new_py_files.get_commit_message(str(tmp_path))
  assert msg == 'HG Description'


def test_get_commit_message_g4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'g4' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['g4', 'info']:
      return 0, 'Server: ...'
    if cmd == ['g4', 'change', '-o']:
      return 0, 'G4 Change Description'
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  msg = check_new_py_files.get_commit_message(str(tmp_path))
  assert msg == 'G4 Change Description'


def test_get_commit_message_p4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'p4' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['p4', 'info']:
      return 0, 'Server: ...'
    if cmd == ['p4', 'change', '-o']:
      return 0, 'P4 Change Description'
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  msg = check_new_py_files.get_commit_message(str(tmp_path))
  assert msg == 'P4 Change Description'


def test_normalize_depot_path(tmp_path: pathlib.Path) -> None:
  upstream_adk = tmp_path / 'third_party' / 'py' / 'google' / 'adk'
  upstream_adk.mkdir(parents=True)
  (upstream_adk / '__init__.py').write_text('', encoding='utf-8')

  workspace = upstream_adk / 'open_source_workspace'
  src_adk = workspace / 'src' / 'google' / 'adk'
  src_adk.mkdir(parents=True)
  (src_adk / '__init__.py').symlink_to(upstream_adk / '__init__.py')

  depot_path = '//depot/mirror/src/google/adk/agents/_g4_agent.py'
  results = check_new_py_files._normalize_and_filter_files(
      [depot_path], repo_root=str(workspace)
  )
  assert len(results) == 1
  display_path, rel_to_adk, filename = results[0]
  assert display_path == depot_path
  assert rel_to_adk == 'agents/_g4_agent.py'
  assert filename == '_g4_agent.py'


def test_main_no_vcs_no_baseline(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  new_dir = tmp_path / 'new'
  (new_dir / 'src' / 'google' / 'adk').mkdir(parents=True)
  (new_dir / 'src' / 'google' / 'adk' / '__init__.py').write_text(
      '', encoding='utf-8'
  )

  monkeypatch.setattr(check_new_py_files.shutil, 'which', lambda _: None)

  exit_code = check_new_py_files.main(['--new-dir', str(new_dir)])
  # 3, not 1 or 2: nothing was checked, which is neither a pass nor a
  # violation. run_precommit_checks reports this as skipped.
  assert exit_code == check_new_py_files._EXIT_INDETERMINATE
  err = capsys.readouterr().err
  assert 'Could not determine the added files' in err
  assert 'not a clean bill of health' in err


def test_sh_forwarder_execution_from_any_cwd(tmp_path: pathlib.Path) -> None:
  script_path = (
      pathlib.Path(check_new_py_files.__file__).resolve().parent
      / 'check_new_py_files.sh'
  )
  proc = subprocess.run(
      ['bash', str(script_path), '--help'],
      cwd=str(tmp_path),
      capture_output=True,
      text=True,
  )
  assert proc.returncode == 0
  assert (
      'usage:' in proc.stdout.lower()
      or 'show this help message' in proc.stdout.lower()
  )
