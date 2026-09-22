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


def test_has_no_unit_guide_tag_ignores_a_prose_mention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Only a tag line waives the rule, not a description that discusses it.

  The matcher used to search for the bare word anywhere in the text, so any
  change whose description talked about the rule waived it -- including the
  change that introduced the gating job, which waived itself.
  """
  monkeypatch.delenv('NO_UNIT_GUIDE', raising=False)
  monkeypatch.delenv('SKIP_UNIT_GUIDE', raising=False)

  assert not check_new_py_files.has_no_unit_guide_tag(
      'Gate the unit guide rule.\n\n'
      'The waiver is a `NO_UNIT_GUIDE` line in the commit message, and CI'
      ' skips the check entirely when it sees one. The prefix rule is gated'
      ' separately and keeps --no-unit-guide for that reason.\n'
  )
  # A real tag still waives, wherever in the message it sits.
  assert check_new_py_files.has_no_unit_guide_tag(
      'Add a seam.\n\nNO_UNIT_GUIDE=internal plumbing\nTAG=agy\n'
  )
  assert check_new_py_files.has_no_unit_guide_tag('SKIP_UNIT_GUIDE=reason')

  # But no looser than a tag parser: a line the surrounding tooling would not
  # read as a tag must not waive here either, or an author is told they are
  # covered by something that will not in fact cover them.
  assert not check_new_py_files.has_no_unit_guide_tag('  NO_UNIT_GUIDE=x')
  assert not check_new_py_files.has_no_unit_guide_tag('NO_UNIT_GUIDE = x')
  assert not check_new_py_files.has_no_unit_guide_tag('no_unit_guide=x')


def test_run_turns_a_crash_into_a_setup_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """A crash must not be reported as a violation.

  Exit 1 means "the rules were checked and the change breaks one", so a caller
  reports a violation and offers its remedy -- for a check that never ran. A
  real trigger: a commit message holding bytes invalid in the process encoding
  makes get_commit_message raise UnicodeDecodeError.
  """

  def boom(argv):
    del argv
    raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')

  monkeypatch.setattr(check_new_py_files, 'main', boom)

  assert check_new_py_files.run(['--new-dir', '.']) == 2
  assert 'crashed' in capsys.readouterr().err


def test_no_waiver_ignores_a_tag_the_caller_did_not_mean(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """--no-waiver closes both waiver channels.

  A caller that applies the waiver itself must not have the rule suppressed by
  a stray tag in the environment, or in some unrelated repository at or above
  the directory it runs in.
  """
  added = _tree_with_added_file(tmp_path, 'agents/_agent.py')
  argv = ['--new-dir', str(tmp_path), '--no-prefix-check', str(added)]

  monkeypatch.setenv('NO_UNIT_GUIDE', 'stray')
  monkeypatch.setattr(
      check_new_py_files, 'get_commit_message', lambda root: 'NO_UNIT_GUIDE=x'
  )
  # Without the flag, either channel waives the rule.
  assert check_new_py_files.main(argv) == 0
  # With it, neither does.
  assert check_new_py_files.main(argv + ['--no-waiver']) == 1


def test_get_commit_message_hg_reads_every_local_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The message range has to match the range the added-file scan covers.

  The file set spans back to the last synced revision, so reading only the
  tip's message would let a commit stacked on top bury a waiver written in the
  commit that adds the file.
  """

  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'hg' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['hg', 'root']:
      return 0, '/workspace'
    if check_new_py_files._LOCAL_COMMITS in cmd:
      return 0, 'add a seam\nNO_UNIT_GUIDE=internal\n\nlater unrelated commit\n'
    if '-r' in cmd and '.' in cmd:
      return 0, 'later unrelated commit'
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)
  monkeypatch.delenv('NO_UNIT_GUIDE', raising=False)
  monkeypatch.delenv('SKIP_UNIT_GUIDE', raising=False)

  msg = check_new_py_files.get_commit_message('.')
  assert check_new_py_files.has_no_unit_guide_tag(msg)


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


def _tree_with_added_file(root: pathlib.Path, rel: str) -> pathlib.Path:
  """Creates a checkout at `root` holding one library source file."""
  path = root / 'src' / 'google' / 'adk' / rel
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text('', encoding='utf-8')
  return path


def test_main_reads_the_added_file_list_from_a_file(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """The gating job passes its file set through a file, not a command line.

  A change can add more files than an argv will hold, and a truncated list
  would silently shrink what gets checked.
  """
  added = _tree_with_added_file(tmp_path, 'agents/_agent.py')
  _tree_with_added_file(tmp_path, 'agents/_untouched.py')

  listing = tmp_path / 'added.txt'
  listing.write_text(f'# added by this change\n\n{added}\n', encoding='utf-8')

  exit_code = check_new_py_files.main([
      '--new-dir',
      str(tmp_path),
      '--added-files-from',
      str(listing),
  ])
  assert exit_code == 1
  err = capsys.readouterr().err
  assert 'agents/_agent.py' in err
  # Only the listed file is checked, even though both exist in the tree.
  assert '_untouched.py' not in err


def test_main_checks_a_file_in_a_subpackage_with_no_symlink_yet(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """A change that adds a whole new subpackage must still be checked.

  Internally the checkout sits inside the package it points into, and its
  src/google/adk reaches the real subpackages through per-subpackage symlinks.
  A subpackage the change is adding has no symlink yet, so its files stay put
  and used to relativize against the package root as
  `<checkout>/src/google/adk/...` -- a path whose first component is an
  excluded directory name, so it was dropped and the change passed without
  being examined.
  """
  # The internal layout: a package root that *contains* the checkout.
  package_root = tmp_path / 'pkg'
  checkout = package_root / 'checkout'
  real_agents = package_root / 'agents'
  real_agents.mkdir(parents=True)
  (package_root / '__init__.py').write_text('', encoding='utf-8')

  adk_src = checkout / 'src' / 'google' / 'adk'
  adk_src.mkdir(parents=True)
  (checkout / 'docs' / 'guides').mkdir(parents=True)
  os.symlink(real_agents, adk_src / 'agents')
  os.symlink(package_root / '__init__.py', adk_src / '__init__.py')

  # The added subpackage has no symlink in the checkout, as it would not on
  # the change that introduces it.
  new_pkg = adk_src / 'brandnewpkg'
  new_pkg.mkdir()
  (new_pkg / '_thing.py').write_text('', encoding='utf-8')

  listing = tmp_path / 'added.txt'
  listing.write_text('src/google/adk/brandnewpkg/_thing.py\n', encoding='utf-8')

  exit_code = check_new_py_files.main([
      '--new-dir',
      str(checkout),
      '--added-files-from',
      str(listing),
      '--no-prefix-check',
  ])
  assert exit_code == 1
  err = capsys.readouterr().err
  assert 'requires a unit guide in docs/guides/' in err
  assert 'brandnewpkg/thing' in err


def test_main_added_files_that_all_filter_away_is_a_setup_error(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """An explicit list that resolves to nothing is a bug, not a clean result.

  The caller of --added-files-from has already decided its paths are library
  sources. If every one filters away, the two disagree about where the package
  is, and reporting success would hide the silent no-op this check exists to
  catch.
  """
  (tmp_path / 'src' / 'google' / 'adk').mkdir(parents=True)
  listing = tmp_path / 'added.txt'
  listing.write_text('/somewhere/else/_thing.py\n', encoding='utf-8')

  exit_code = check_new_py_files.main([
      '--new-dir',
      str(tmp_path),
      '--added-files-from',
      str(listing),
  ])
  assert exit_code == 2
  assert 'nothing was checked' in capsys.readouterr().err


def test_main_added_files_from_a_missing_file_is_a_setup_error(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """An unreadable list must not be read as an empty one."""
  (tmp_path / 'src' / 'google' / 'adk').mkdir(parents=True)

  exit_code = check_new_py_files.main([
      '--new-dir',
      str(tmp_path),
      '--added-files-from',
      str(tmp_path / 'nope.txt'),
  ])
  assert exit_code == 2
  assert 'names no file' in capsys.readouterr().err


def test_main_no_prefix_check_leaves_the_unit_guide_rule_enforced(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """The unit guide check gates only its own rule.

  Where the unit guide rule is enforced, a NO_UNIT_GUIDE waiver skips the
  whole check. Enforcing the prefix rule in the same place would let that
  waiver take the un-waivable rule with it.
  """
  added = _tree_with_added_file(tmp_path, 'agents/agent.py')

  exit_code = check_new_py_files.main([
      '--new-dir',
      str(tmp_path),
      '--no-prefix-check',
      str(added),
  ])
  assert exit_code == 1
  err = capsys.readouterr().err
  assert "must have a '_' prefix" not in err
  assert 'requires a unit guide in docs/guides/' in err


def test_main_no_prefix_check_and_no_unit_guide_check_nothing(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  added = _tree_with_added_file(tmp_path, 'agents/agent.py')

  exit_code = check_new_py_files.main([
      '--new-dir',
      str(tmp_path),
      '--no-prefix-check',
      '--no-unit-guide',
      str(added),
  ])
  assert exit_code == 0
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


def test_get_vcs_added_files_git_unreachable_range_is_indeterminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A range that does not resolve is unknown, not empty.

  In a depth-1 clone HEAD~1 does not exist, so the diff fails rather than
  coming back empty. Reporting "no files added" there is a clean bill of
  health nobody earned; the caller must be told it could not be determined.
  """

  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'git' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if 'rev-parse' in cmd:
      return 0, 'true'
    if '--cached' in cmd:
      return 0, ''
    if check_new_py_files._GIT_HEAD_RANGE in cmd:
      return 128, ''  # fatal: ambiguous argument 'HEAD~1..HEAD'
    return 0, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  assert check_new_py_files.get_vcs_added_files('.') is None


def test_get_vcs_added_files_git_empty_range_is_no_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A range that resolves to an empty diff really is no added files."""

  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'git' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if 'rev-parse' in cmd:
      return 0, 'true'
    return 0, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  assert check_new_py_files.get_vcs_added_files('.') == set()


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


_HG_SYNCED_BASE_STATUS = [
    'hg',
    'status',
    '--added',
    '--no-status',
    '--rev',
    check_new_py_files._SYNCED_BASE,
]
_HG_WORKING_DIR_STATUS = ['hg', 'status', '--added', '--no-status']


def test_get_vcs_added_files_hg(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'hg' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['hg', 'root']:
      return 0, '/workspace'
    if cmd == _HG_SYNCED_BASE_STATUS:
      return 0, 'src/google/adk/agents/_hg_agent.py'
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  added = check_new_py_files.get_vcs_added_files('.')
  assert added == {'/workspace/src/google/adk/agents/_hg_agent.py'}


def test_get_vcs_added_files_hg_sees_an_already_committed_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A Mercurial checkout is normally committed by the time this runs.

  `hg status --added` on its own reports only files added and not yet
  committed, so it goes empty after `hg commit` or `hg amend` and the check
  silently passed every such change. The file set has to come from a diff
  against the last synced revision instead.
  """

  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'hg' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['hg', 'root']:
      return 0, '/workspace'
    if cmd == _HG_SYNCED_BASE_STATUS:
      return 0, 'src/google/adk/agents/_committed.py'
    if cmd == _HG_WORKING_DIR_STATUS:
      return 0, ''  # Committed, so nothing is pending in the working copy.
    return 1, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  added = check_new_py_files.get_vcs_added_files('.')
  assert added == {'/workspace/src/google/adk/agents/_committed.py'}


def test_get_vcs_added_files_hg_falls_back_when_the_revset_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A plain hg repository need not have the phases the revset relies on."""

  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'hg' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if cmd == ['hg', 'root']:
      return 0, '/workspace'
    if cmd == _HG_SYNCED_BASE_STATUS:
      return 255, ''
    if cmd == _HG_WORKING_DIR_STATUS:
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


def test_get_commit_message_git_reads_the_merged_commits_on_a_pull_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  """On a pull request HEAD is a merge commit CI wrote, not the contributor.

  Its message can never carry a waiver, so the same HEAD~1..HEAD range the
  added-file scan falls back to has to be read for one.
  """

  def fake_which(cmd: str) -> str | None:
    return '/usr/bin/' + cmd if cmd == 'git' else None

  def fake_run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    if 'rev-parse' in cmd and '--is-inside-work-tree' in cmd:
      return 0, 'true'
    if 'rev-parse' in cmd and '--git-dir' in cmd:
      return 0, str(tmp_path / 'no-such-git-dir')
    if 'log' in cmd and check_new_py_files._GIT_HEAD_RANGE in cmd:
      return 0, 'feat: add a thing\n\nNO_UNIT_GUIDE=internal seam'
    if 'log' in cmd:
      return 0, 'Merge 1234abc into 5678def'
    return 0, ''

  monkeypatch.setattr(check_new_py_files.shutil, 'which', fake_which)
  monkeypatch.setattr(check_new_py_files, '_run_cmd', fake_run_cmd)

  msg = check_new_py_files.get_commit_message(str(tmp_path))
  assert check_new_py_files.has_no_unit_guide_tag(msg)


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
