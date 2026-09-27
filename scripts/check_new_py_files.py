#!/usr/bin/env python3
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

"""Checks that newly-added Python files under src/google/adk/ follow conventions.

ADK conventions enforced for newly-added Python files:
1. Private-by-default: Newly-added Python files under src/google/adk/ must
   have a '_'-prefixed basename. To expose public symbols, export them via the
   subpackage __init__.py / __all__.
   See .agents/skills/adk-style/references/visibility.md.
2. Unit guide requirement: Newly-added Python files under src/google/adk/ must
   have a corresponding unit guide in docs/guides/ (unless exempt or tagged with
   NO_UNIT_GUIDE / SKIP_UNIT_GUIDE in the commit message or environment).
   See .agents/skills/adk-unit-guide/SKILL.md.

Either rule can be switched off on its own, with --no-prefix-check and
--no-unit-guide. The two are gated by separate CI jobs for that reason: the
unit guide rule is waivable per change and the prefix rule is not, so whatever
waives one must not quietly disable the other.

Modes for finding added files:
- Baseline Diff Mode (CI):
    python scripts/check_new_py_files.py --baseline-dir /path/to/origin-main
- VCS Detection Mode (Local / Pre-commit):
    python scripts/check_new_py_files.py
- Explicit File List:
    python scripts/check_new_py_files.py file1.py file2.py
- File List From A File (CI, where the list can outgrow a command line):
    python scripts/check_new_py_files.py --added-files-from added.txt

Exit codes: 0 = ok, 1 = violation(s) found, 2 = usage/setup error,
3 = indeterminate (the set of added files could not be resolved at all).

Exit code 3 exists so that "could not determine the added files" cannot be
read as "no violations". A caller that runs this opportunistically, such as
the pre-commit hook, can report it as skipped; a caller that relies on it to
gate a change passes --baseline-dir or --added-files-from and never sees it.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import traceback

_PACKAGE_RELPATH = os.path.join('src', 'google', 'adk')
_DOCS_GUIDES_RELPATH = os.path.join('docs', 'guides')

# The package's import path, i.e. _PACKAGE_RELPATH without the src/ root that
# only this checkout uses. A repository laying the package out differently
# still ends its path to it with these components.
_PACKAGE_IMPORT_PATH = 'google/adk'

_EXIT_OK = 0
_EXIT_VIOLATIONS = 1
_EXIT_SETUP_ERROR = 2
_EXIT_INDETERMINATE = 3

# The newest revision this checkout shares with the server, as a Mercurial
# revset: everything after it is the change under construction. Revisions the
# server already has are in the public phase and local work is draft, so this
# is the base the change sits on; with no local commits it evaluates to '.'.
_SYNCED_BASE = 'last(public() & ::.)'

# The local commits themselves, as a Mercurial revset: the ones after
# `_SYNCED_BASE`, i.e. the work the change is made of. Used to read the commit
# messages over the same range the added-file scan covers.
_LOCAL_COMMITS = 'draft() & ::.'

# The commit range a git checkout's HEAD covers when nothing is staged. On a
# pull request this is the base branch to the merge commit, i.e. the pull
# request's own commits, which is both the file set to check and the place a
# NO_UNIT_GUIDE waiver would be written.
_GIT_HEAD_RANGE = 'HEAD~1..HEAD'

# Entries that bring a genuinely new path into the tree.
#
# Renames are included because renaming a private module to a public one
# creates a name no rule has ever been applied to. They are asked for
# separately from plain adds, with --name-status, because only the ones that
# change the file's *name* qualify: moving `runners.py` to another directory
# keeps a public name that was already accepted, and treating that as new
# would fail it against a prefix rule that has no waiver. When rename
# detection is off the same change arrives as an add plus a delete, which the
# add filter covers.
_GIT_ADD_FILTER = '--diff-filter=A'
_GIT_RENAME_FILTER = '--diff-filter=R'

# The unit guide waiver, as it appears in a commit message: its own line, in
# `KEY=<reason>` form, flush left and with no space before the `=`. Matching
# the bare word anywhere in the text instead would waive the rule for any
# change whose message merely discusses it -- the change that introduced this
# check waived itself that way. The shape is deliberately no looser than the
# one a tag parser accepts: waiving locally on a line that the surrounding
# tooling would not read as a tag is how an author ends up believing they are
# covered when they are not.
#
# The tag has to carry a reason, so it has to reach a non-space character. A
# bare `NO_UNIT_GUIDE=` would otherwise waive every file the change adds while
# recording nothing a reviewer can weigh, which is the opposite of what the tag
# exists for.
_NO_UNIT_GUIDE_TAG = re.compile(
    r'^(?:NO|SKIP)_UNIT_GUIDE=[ \t]*\S', re.MULTILINE
)

_PREFIX_VIOLATION_LINE = (
    "Error: New Python file '{path}' must have a '_' prefix.\n"
    'All new Python files in src/google/adk/ must be private by default.\n'
    'To expose a public interface, use __init__.py and list public symbols in'
    ' __all__.\n'
    'See .agents/skills/adk-style/references/visibility.md for details.'
)

_GUIDE_VIOLATION_LINE = (
    "Error: New Python file '{path}' requires a unit guide in docs/guides/.\n"
    "Expected guide at 'docs/guides/{expected}/index.md' or"
    " 'docs/guides/{expected}.md'.\n"
    'If a unit guide is not required for this file, explain why with a'
    " 'NO_UNIT_GUIDE=<reason>' tag in the commit message of the change that"
    ' adds it. Where no message can be read, as when the change is only'
    ' staged or the tree carries no version control, set the tag in the'
    " environment instead: NO_UNIT_GUIDE='<reason>' git commit ...\n"
    'See .agents/skills/adk-unit-guide/SKILL.md for details on creating unit'
    ' guides.'
)

# Subtrees that may exist in the working tree but are intentionally absent from
# the baseline tree or should not be checked for public/private conventions.
_IGNORED_PREFIXES = (
    'src/google/adk/internal/',
    'src/google/adk/v1/',
    'src/google/adk/platform/internal/',
)

# Directories directly under the package root whose contents are not library
# source. Matched against the first path component only: a nested directory
# that happens to carry one of these names still holds source to check.
_EXCLUDE_DIR_NAMES = (
    'tests',
    'open_source_workspace',
    'contributing',
)

# File and directory glob patterns exempt from the unit guide requirement.
_EXEMPT_GUIDE_PATTERNS = (
    '__init__.py',
    'cli/*',
    '*/cli/*',
    'utils/*',
    '*/utils/*',
    '*_utils.py',
    '*_helper.py',
    '*_helpers.py',
    '*_types.py',
    '*_errors.py',
    '*_exceptions.py',
    '*_constants.py',
)


def find_py_files(root: str) -> set[str]:
  """Returns root-relative paths of every *.py under <root>/src/google/adk.

  Each path includes the src/google/adk/ prefix (e.g.
  'src/google/adk/agents/foo.py'). Symlinks are followed so that a
  src/google/adk tree assembled from symlinked subdirectories is walked
  correctly.

  Args:
    root: The root directory of the repository.

  Returns:
    A set of root-relative paths of every *.py under
    <root>/src/google/adk.
  """
  package_root = os.path.join(root, _PACKAGE_RELPATH)
  if not os.path.isdir(package_root):
    return set()
  found: set[str] = set()
  for dirpath, _, filenames in os.walk(package_root, followlinks=True):
    for name in filenames:
      if name.endswith('.py'):
        abs_path = os.path.join(dirpath, name)
        rel = os.path.relpath(abs_path, root).replace(os.sep, '/')
        found.add(rel)
  return found


def _should_check(relpath: str) -> bool:
  """Returns False for paths under an ignored prefix."""
  relpath = relpath.replace(os.sep, '/')
  return not any(relpath.startswith(prefix) for prefix in _IGNORED_PREFIXES)


def added_py_files_from_baseline(new_root: str, baseline_root: str) -> set[str]:
  """Returns .py files present in new_root but not in baseline_root."""
  added = find_py_files(new_root) - find_py_files(baseline_root)
  return {path for path in added if _should_check(path)}


def _run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
  try:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()
  except (FileNotFoundError, OSError):
    return -1, ''


def _is_private_name(path: str) -> bool:
  """Whether `path`'s basename is private by the '_' prefix convention."""
  return os.path.basename(path).startswith('_')


def _package_relative(repo_relative: str) -> str | None:
  """Returns a path relative to the package, or None if it lies outside it.

  Args:
    repo_relative: A path as git reports it, e.g. `src/google/adk/a/b.py`.

  Returns:
    The package-relative form, or None when the path is not library source
    these rules cover.
  """
  prefix = _PACKAGE_RELPATH.replace(os.sep, '/') + '/'
  path = repo_relative.replace(os.sep, '/')
  if not path.startswith(prefix) or not path.endswith('.py'):
    return None
  rel = path[len(prefix) :]
  return rel if _keep_relative_path(rel) else None


def _rename_exposes_a_new_name(source: str, destination: str) -> bool:
  """Whether a rename produces a name neither rule has judged before.

  The two rules do not cover the same files, so "was the source already
  judged" has to be asked once per rule. A move out of `cli/` is judged by
  the prefix rule at both ends but meets the unit guide rule only on arrival,
  and a `.pyi` renamed to `.py` was never library source at all.

  Args:
    source: The rename's source, as git reports it.
    destination: The rename's destination, as git reports it.

  Returns:
    True when the destination carries a name that has not been held to a rule
    it is now subject to.
  """
  source_rel = _package_relative(source)
  if source_rel is None:
    # Never library source, so nothing has ever looked at this name.
    return True
  destination_rel = _package_relative(destination)
  if destination_rel is None:
    # Leaving the library; the destination is not ours to judge.
    return False

  # The prefix rule covers both ends, so it matters only when visibility
  # changes.
  if _is_private_name(source_rel) and not _is_private_name(destination_rel):
    return True
  # The guide rule does not cover every file, so leaving an exemption puts a
  # name under it for the first time.
  was_exempt = is_exempt_from_unit_guide(
      source_rel, os.path.basename(source_rel)
  )
  now_exempt = is_exempt_from_unit_guide(
      destination_rel, os.path.basename(destination_rel)
  )
  return was_exempt and not now_exempt


def _git_renamed_to_new_names(base_cmd: list[str], root: str) -> set[str]:
  """Returns rename destinations that newly expose a public name.

  A rename needs judging when it produces a name nothing has judged before.
  Relocating `runners.py`, or renaming it to `runner.py`, carries a public
  name that was accepted when the file was created; re-judging either would
  fail an ordinary refactor against the prefix rule, which has no waiver.
  Two cases do need it: renaming `_runners.py` to `runners.py`, which puts a
  module on the public surface, and moving a file in from `tests/` or
  anywhere else these rules never covered, whose name has never been held to
  them whatever it happens to be.

  Args:
    base_cmd: The git diff invocation to extend, e.g. `['git', 'diff']`.
    root: The root directory of the repository.

  Returns:
    The destination paths worth checking. Empty when git reports no renames.
  """
  _, out = _run_cmd(base_cmd + ['--name-status', _GIT_RENAME_FILTER], cwd=root)
  renamed: set[str] = set()
  for line in out.splitlines():
    # `R100\told/path\tnew/path`, with the similarity score on the status.
    parts = line.split('\t')
    if len(parts) != 3 or not parts[0].startswith('R'):
      continue
    _, source, destination = parts
    source, destination = source.strip(), destination.strip()
    if _rename_exposes_a_new_name(source, destination):
      renamed.add(destination)
  return renamed


def _git_added_paths(base_cmd: list[str], root: str) -> set[str]:
  """Returns the paths a git diff brings into the tree under a new name.

  Args:
    base_cmd: The git diff invocation to extend, e.g. `['git', 'diff']`.
    root: The root directory of the repository.

  Returns:
    Plain additions, plus renames that change the file's name.
  """
  _, out = _run_cmd(base_cmd + ['--name-only', _GIT_ADD_FILTER], cwd=root)
  added = {f for f in out.splitlines() if f.strip()}
  return added | _git_renamed_to_new_names(base_cmd, root)


def _git_is_mid_commit(root: str) -> bool | None:
  """Reports whether a change is staged and not yet committed.

  Both the added-file scan and the waiver scan branch on this, and they must
  branch on it together: the index and HEAD describe different changes, so
  reading files from one and the waiver from the other lets a tag written for
  the previous commit apply to this one.

  Args:
    root: The root directory of the repository.

  Returns:
    True when anything at all is staged. Asking whether any *addition* is
    staged would send a commit that adds nothing down the HEAD~1..HEAD path,
    where it would be judged on what the previous commit added. None when git
    could not say, as with an unreadable index: treating that as "nothing is
    staged" sent the scan to HEAD~1..HEAD, which reports what the previous
    commit added and passes a staged file nobody looked at.
  """
  code, staged_any = _run_cmd(
      ['git', 'diff', '--cached', '--name-only'], cwd=root
  )
  if code != 0:
    return None
  return bool(staged_any.strip())


def get_vcs_added_files(root: str = '.') -> set[str] | None:
  """Detects added files using local VCS (git, jj, hg, g4, p4).

  Args:
    root: The root directory of the repository.

  Returns:
    A set of added file paths if a supported VCS is detected, or None if no
    supported VCS was detected.
  """
  # 1. git
  if shutil.which('git'):
    code, _ = _run_cmd(['git', 'rev-parse', '--is-inside-work-tree'], cwd=root)
    if code == 0:
      mid_commit = _git_is_mid_commit(root)
      if mid_commit is None:
        print(
            'git is active but its index cannot be read, so whether this'
            ' change is staged or committed is unknown, and so is what it'
            ' adds.',
            file=sys.stderr,
        )
        return None
      if mid_commit:
        return _git_added_paths(['git', 'diff', '--cached'], root)
      range_code, head_diff = _run_cmd(
          ['git', 'diff', _GIT_HEAD_RANGE, '--name-only', _GIT_ADD_FILTER],
          cwd=root,
      )
      if range_code != 0:
        # HEAD~1 is unreachable, as in a depth-1 clone. The range resolved to
        # nothing rather than to an empty diff, so the added files are unknown
        # and saying "none" here would be a clean bill of health nobody earned.
        # Say why here: the caller only learns that nothing could be resolved,
        # and "no version control is active" would be the wrong diagnosis when
        # git is active and it is the range that failed.
        print(
            f'git is active but {_GIT_HEAD_RANGE} does not resolve, so what'
            ' this change adds cannot be read from it. A shallow clone does'
            ' this; fetch enough history for HEAD to have a parent.',
            file=sys.stderr,
        )
        return None
      added = {f for f in head_diff.splitlines() if f.strip()}
      return added | _git_renamed_to_new_names(
          ['git', 'diff', _GIT_HEAD_RANGE], root
      )

  # 2. jj
  if shutil.which('jj'):
    code, jj_root = _run_cmd(['jj', 'root'], cwd=root)
    if code == 0:
      _, out = _run_cmd(['jj', 'diff', '--summary'], cwd=root)
      added = set()
      for line in out.splitlines():
        if line.startswith('A '):
          parts = line.split(maxsplit=1)
          if len(parts) == 2:
            p = parts[1].strip()
            if jj_root and not os.path.isabs(p):
              p = os.path.join(jj_root, p)
            added.add(p)
      return added

  # 3. hg
  if shutil.which('hg'):
    code, hg_root = _run_cmd(['hg', 'root'], cwd=root)
    if code == 0:
      # A bare `hg status --added` reports only files added and not yet
      # committed, so it goes empty the moment the change is committed or
      # amended -- which is the usual state of a checkout by the time anyone
      # runs this. Diff against the last synced revision instead, so the file
      # set is the change's own content whether or not it is committed.
      # `_SYNCED_BASE` degrades to '.' in a checkout with no local commits,
      # where it reports the same thing a bare status does.
      code, out = _run_cmd(
          ['hg', 'status', '--added', '--no-status', '--rev', _SYNCED_BASE],
          cwd=root,
      )
      if code != 0:
        # A repository whose phases do not distinguish local work this way.
        _, out = _run_cmd(['hg', 'status', '--added', '--no-status'], cwd=root)
      return {
          os.path.join(hg_root, f.strip())
          if (hg_root and not os.path.isabs(f.strip()))
          else f.strip()
          for f in out.splitlines()
          if f.strip()
      }

  # 4. g4
  if shutil.which('g4'):
    code, _ = _run_cmd(['g4', 'info'], cwd=root)
    if code == 0:
      _, out = _run_cmd(['g4', 'opened'], cwd=root)
      added = set()
      for line in out.splitlines():
        if ' - add ' in line:
          depot_file = line.split(' - add ')[0].split('#')[0].strip()
          added.add(depot_file)
      return added

  # 5. p4
  if shutil.which('p4'):
    code, _ = _run_cmd(['p4', 'info'], cwd=root)
    if code == 0:
      _, out = _run_cmd(['p4', 'opened'], cwd=root)
      added = set()
      for line in out.splitlines():
        if ' - add ' in line:
          depot_file = line.split(' - add ')[0].split('#')[0].strip()
          added.add(depot_file)
      return added

  return None


def get_commit_message(root: str = '.') -> str:
  """Retrieves commit message or description from VCS.

  Args:
    root: The root directory of the repository.

  Returns:
    The message to search for a waiver tag, or '' when none can be read.
  """
  # 1. git
  if shutil.which('git'):
    code, _ = _run_cmd(['git', 'rev-parse', '--is-inside-work-tree'], cwd=root)
    if code == 0:
      if _git_is_mid_commit(root) is not False:
        # The change is staged, so the commit carrying it does not exist yet
        # and its message is nowhere to be read: a pre-commit hook runs before
        # git records what the author typed, and HEAD still describes the
        # previous change. Returning HEAD's message here is what let a waiver
        # written for an earlier commit silently cover this one. Waiving the
        # change being committed goes through the environment instead --
        # `NO_UNIT_GUIDE='<reason>' git commit ...` -- which
        # has_no_unit_guide_tag honours and the violation text advertises.
        # None lands here too: a waiver that cannot be attributed to a change
        # must not be applied to one.
        return ''
      _, msg = _run_cmd(['git', 'log', '-1', '--pretty=%B'], cwd=root)
      # On a pull request, HEAD is a merge commit whose own message is
      # generated by CI and can hold no waiver. The commits being merged are
      # the ones the contributor wrote, so read the same range the added-file
      # scan falls back to. Empty when HEAD~1 is unreachable.
      _, range_msg = _run_cmd(
          ['git', 'log', _GIT_HEAD_RANGE, '--pretty=%B'], cwd=root
      )
      if range_msg:
        msg = f'{msg}\n{range_msg}'
      # COMMIT_EDITMSG is deliberately not consulted. It was read here to
      # catch the message of the commit being made, which it never held: git
      # writes it only after the pre-commit hook has run, so during that hook
      # it carries the previous commit's message, or the message of an attempt
      # some hook rejected. Both are messages written for another change, and
      # neither can be told from a current one by inspection.
      return msg

  # 2. jj
  if shutil.which('jj'):
    code, _ = _run_cmd(['jj', 'root'], cwd=root)
    if code == 0:
      _, out = _run_cmd(
          ['jj', 'log', '-r', '@', '--no-graph', '-T', 'description'], cwd=root
      )
      return out

  # 3. hg
  if shutil.which('hg'):
    code, _ = _run_cmd(['hg', 'root'], cwd=root)
    if code == 0:
      # Every local commit, not just the tip. The added-file scan above spans
      # the whole range back to the last synced revision, so reading only the
      # tip's message would let one commit on top bury a waiver written in the
      # commit that actually adds the file -- and would let an unrelated tip
      # message waive the whole range.
      code, out = _run_cmd(
          ['hg', 'log', '-r', _LOCAL_COMMITS, '--template', '{desc}\n'],
          cwd=root,
      )
      if code != 0:
        _, out = _run_cmd(
            ['hg', 'log', '-r', '.', '--template', '{desc}'], cwd=root
        )
      return out

  # 4. g4
  if shutil.which('g4'):
    code, _ = _run_cmd(['g4', 'info'], cwd=root)
    if code == 0:
      code, out = _run_cmd(['g4', 'change', '-o'], cwd=root)
      if code == 0 and out:
        return out
      _, out = _run_cmd(['g4', 'describe'], cwd=root)
      return out

  # 5. p4
  if shutil.which('p4'):
    code, _ = _run_cmd(['p4', 'info'], cwd=root)
    if code == 0:
      _, out = _run_cmd(['p4', 'change', '-o'], cwd=root)
      return out

  return ''


def is_exempt_from_unit_guide(rel_path: str, filename: str) -> bool:
  """Returns True if the file matches exemption patterns for unit guides."""
  rel_path = rel_path.replace(os.sep, '/')
  for pattern in _EXEMPT_GUIDE_PATTERNS:
    if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(filename, pattern):
      return True
  return False


def has_no_unit_guide_tag(commit_msg: str) -> bool:
  """Checks if NO_UNIT_GUIDE / SKIP_UNIT_GUIDE is present in env or commit message.

  A reason is required in either channel, so a variable holding only whitespace
  waives nothing, the same way a bare tag in a message does not.
  """
  for name in ('NO_UNIT_GUIDE', 'SKIP_UNIT_GUIDE'):
    if os.environ.get(name, '').strip():
      return True
  return bool(_NO_UNIT_GUIDE_TAG.search(commit_msg))


def _depot_path_to_abs(depot_path: str, adk_real_root: str) -> str:
  """Locates a depot-style path inside the package being checked.

  A depot path names a file by its position in the repository the VCS serves,
  which shares no prefix with the checkout on disk. What the two do share is
  the package itself, so the split point is the last occurrence of the import
  path. Keying on that rather than on a repository prefix keeps any particular
  repository layout out of this script.

  Args:
    depot_path: A path of the form `//<repo>/<...>/<module>.py`.
    adk_real_root: Absolute, symlink-resolved path of the package root.

  Returns:
    The absolute path of the matching file, or the depot path resolved as-is
    when it does not run through the package. The caller drops anything that
    does not land inside the package.
  """
  clean_path = depot_path.lstrip('/')
  marker = f'{_PACKAGE_IMPORT_PATH}/'
  if marker in clean_path:
    rel_to_package = clean_path.rsplit(marker, 1)[1]
    return os.path.realpath(os.path.join(adk_real_root, rel_to_package))
  return os.path.realpath(clean_path)


def _subpackage_renames(package_dir: str) -> dict[str, str]:
  """Maps a subpackage's real directory to the name the source tree gives it.

  A subpackage can be exposed under a name of its own: `dependencies` points
  at `dependencies_external`. Which of the two names a path arrives wearing
  depends only on how it was detected -- git reports it relative to the
  checkout, while the Piper-shaped detectors report the real location -- so
  without this the same file demands its guide in two different directories.

  Args:
    package_dir: The checkout's own `src/google/adk`, symlinks unresolved.

  Returns:
    Real directory path -> source-tree name, for each renamed subpackage.
  """
  # The whole walk is guarded, not just the listing: is_dir() follows the
  # link, so a symlink loop or an unreadable target raises here rather than at
  # the scandir. Letting that escape would turn a checkout oddity into a
  # failed check.
  renames: dict[str, str] = {}
  try:
    for entry in os.scandir(package_dir):
      if not entry.is_symlink() or not entry.is_dir():
        continue
      target = os.path.realpath(entry.path)
      if os.path.basename(target) != entry.name:
        renames[target] = entry.name
  except OSError:
    return {}
  return renames


def _apply_subpackage_rename(
    rel_to_adk: str, abs_file: str, renames: dict[str, str]
) -> str:
  """Restores the source-tree name of a path that resolved through a symlink.

  Args:
    rel_to_adk: The package-relative path, possibly wearing the real name.
    abs_file: The same file, resolved.
    renames: The mapping from `_subpackage_renames`.

  Returns:
    `rel_to_adk` with its leading component put back to the name the source
    tree uses, or unchanged when no rename applies.
  """
  if not renames:
    return rel_to_adk
  first, sep, rest = rel_to_adk.partition('/')
  if not sep:
    return rel_to_adk
  # Match on the resolved directory rather than on the name, so that two
  # subpackages sharing a basename cannot be confused for one another.
  subpackage_real = abs_file[: -(len(rest) + 1)] if rest else abs_file
  renamed = renames.get(os.path.normpath(subpackage_real))
  return f'{renamed}/{rest}' if renamed else rel_to_adk


def _keep_relative_path(rel_to_adk: str) -> bool:
  """Whether a package-relative path names a file these rules apply to.

  Args:
    rel_to_adk: A path relative to the package root, e.g. `agents/_agent.py`.

  Returns:
    False for a path under an ignored prefix, or under a top-level directory
    that holds no library source.
  """
  full_rel = os.path.join('src', 'google', 'adk', rel_to_adk).replace(
      os.sep, '/'
  )
  if not _should_check(full_rel):
    return False
  # Anchored at the package root: a nested directory that happens to carry an
  # excluded name still holds source to check.
  return rel_to_adk.split('/')[0] not in _EXCLUDE_DIR_NAMES


def _normalize_and_filter_files(
    raw_files: set[str] | list[str], repo_root: str
) -> list[tuple[str, str, str]]:
  """Normalizes added files and filters to relevant Python source files.

  Handles both standard layout (src/google/adk/) and symlinked package
  structures where subpackages point to an upstream source tree.

  Args:
    raw_files: The set of raw file paths to normalize and filter.
    repo_root: The root directory of the repository.

  Returns:
    A list of tuples: (display_path, rel_to_adk_root, filename).
  """
  repo_root = os.path.abspath(repo_root)
  package_dir = os.path.join(repo_root, _PACKAGE_RELPATH)
  package_real_dir = (
      os.path.realpath(package_dir)
      if os.path.exists(package_dir)
      else package_dir
  )

  init_file = os.path.join(package_dir, '__init__.py')
  if os.path.exists(init_file):
    adk_real_root = os.path.dirname(os.path.realpath(init_file))
  else:
    adk_real_root = package_real_dir

  renames = _subpackage_renames(package_dir)

  results: list[tuple[str, str, str]] = []
  for raw_file in sorted(raw_files):
    if not raw_file or not raw_file.endswith('.py'):
      continue

    # Handle depot-style paths (e.g. //depot/.../agents/_agent.py), which the
    # Perforce-style branches of get_vcs_added_files report.
    if raw_file.startswith('//'):
      abs_file = _depot_path_to_abs(raw_file, adk_real_root)
    elif os.path.isabs(raw_file):
      abs_file = os.path.realpath(raw_file)
    else:
      abs_file = os.path.realpath(os.path.join(repo_root, raw_file))

    # Before resolving anything, see whether the path already sits under the
    # checkout's own src/google/adk. A subpackage exposed there under a name
    # different from its own -- `dependencies` for `dependencies_external` --
    # would otherwise resolve through the symlink and come back wearing the
    # name the source tree does not use, so the guide would be demanded at a
    # directory that does not exist in the exported repository. Keeping the
    # unresolved form makes the source-tree name win, which is the one both
    # this checkout and the exported repository agree on.
    lexical = os.path.abspath(os.path.join(repo_root, raw_file))
    if not raw_file.startswith('//') and lexical.startswith(
        package_dir + os.sep
    ):
      rel_to_adk = os.path.relpath(lexical, package_dir).replace(os.sep, '/')
      if _keep_relative_path(rel_to_adk):
        results.append((raw_file, rel_to_adk, os.path.basename(lexical)))
      continue

    # Check whether the file belongs to the package in either internal or
    # external layout.
    #
    # The checkout's own src/google/adk comes first, and must: internally it
    # sits *inside* the package it points into, so both prefixes match a file
    # under it and matching the outer one first mislabels the file. A path
    # there resolves out to the package only through a subpackage symlink, so
    # a file in a subpackage the checkout has no symlink for -- a subpackage
    # the change is adding -- stays put and relativizes against the package
    # root as `<checkout>/src/google/adk/<...>`. That starts with an excluded
    # directory name, so the file was dropped and a change adding a new
    # subpackage passed both rules without being examined.
    if abs_file.startswith(package_real_dir + os.sep):
      rel_to_adk = os.path.relpath(abs_file, package_real_dir).replace(
          os.sep, '/'
      )
    elif abs_file.startswith(adk_real_root + os.sep):
      rel_to_adk = os.path.relpath(abs_file, adk_real_root).replace(os.sep, '/')
    else:
      continue

    # These two branches reached the file through its real location, so a
    # renamed subpackage arrives under the name the source tree does not use.
    rel_to_adk = _apply_subpackage_rename(rel_to_adk, abs_file, renames)

    if not _keep_relative_path(rel_to_adk):
      continue

    filename = os.path.basename(abs_file)
    results.append((raw_file, rel_to_adk, filename))

  return results


def check_files(
    files_to_check: list[tuple[str, str, str]],
    repo_root: str,
    commit_msg: str = '',
    skip_unit_guide: bool = False,
    skip_prefix: bool = False,
    ignore_waiver: bool = False,
) -> tuple[list[str], list[str]]:
  """Validates newly added Python files against ADK conventions.

  Args:
    files_to_check: List of files to check, each as a tuple of (display_path,
      rel_to_adk_root, filename).
    repo_root: The root directory of the repository.
    commit_msg: The commit message of the change being checked.
    skip_unit_guide: Whether to skip unit guide checks.
    skip_prefix: Whether to skip the private-by-default prefix check.
    ignore_waiver: Whether to disregard a NO_UNIT_GUIDE tag in `commit_msg` or
      in the environment, for a caller that applies the waiver itself.

  Returns:
    A tuple of (prefix_violations, guide_violations).
  """
  prefix_violations: list[str] = []
  guide_violations: list[tuple[str, str]] = []  # (path, expected_guide_dir)

  docs_guides_dir = os.path.join(
      os.path.abspath(repo_root), _DOCS_GUIDES_RELPATH
  )
  skip_guide = skip_unit_guide or (
      not ignore_waiver and has_no_unit_guide_tag(commit_msg)
  )

  for display_path, rel_to_adk, filename in files_to_check:
    # 1. Private '_' prefix check
    if not skip_prefix and not filename.startswith('_'):
      prefix_violations.append(display_path)

    # 2. Unit guide check
    if not skip_guide and not is_exempt_from_unit_guide(rel_to_adk, filename):
      rel_dir = os.path.dirname(rel_to_adk)
      name_no_ext = filename[:-3]  # strip .py
      # One underscore, not all of them: '__thing.py' is the private form of
      # '_thing', so that is the guide name to look for.
      name_no_prefix = name_no_ext.removeprefix('_') or name_no_ext

      guide_found = False
      for cand_name in (name_no_prefix, name_no_ext):
        if rel_dir and rel_dir != '.':
          candidates = [
              os.path.join(docs_guides_dir, rel_dir, cand_name, 'index.md'),
              os.path.join(docs_guides_dir, rel_dir, f'{cand_name}.md'),
          ]
        else:
          candidates = [
              os.path.join(docs_guides_dir, cand_name, 'index.md'),
              os.path.join(docs_guides_dir, f'{cand_name}.md'),
          ]
        if any(os.path.isfile(c) for c in candidates):
          guide_found = True
          break

      if not guide_found:
        expected = (
            f'{rel_dir}/{name_no_prefix}'
            if (rel_dir and rel_dir != '.')
            else name_no_prefix
        )
        guide_violations.append((display_path, expected))

  rendered_prefix_errors = [
      _PREFIX_VIOLATION_LINE.format(path=p) for p in prefix_violations
  ]
  rendered_guide_errors = [
      _GUIDE_VIOLATION_LINE.format(path=p, expected=exp)
      for p, exp in guide_violations
  ]
  return rendered_prefix_errors, rendered_guide_errors


def _has_package_dir(root: str) -> bool:
  return os.path.isdir(os.path.join(root, _PACKAGE_RELPATH))


def _parse_args(argv: list[str]) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--baseline-dir',
      help=(
          'Baseline source tree to diff against (an origin/main checkout). If'
          ' omitted, detects added files via local VCS.'
      ),
  )
  parser.add_argument(
      '--new-dir',
      default='.',
      help='New source tree to check (default: current directory).',
  )
  parser.add_argument(
      '--added-files-from',
      help=(
          'File holding the added paths to check, one per line. Blank lines'
          ' and #-comments are ignored. Use instead of positional arguments'
          ' when the list may outgrow a command line.'
      ),
  )
  parser.add_argument(
      '--no-unit-guide',
      '--skip-unit-guide',
      action='store_true',
      dest='no_unit_guide',
      help='Skip unit guide requirement checks.',
  )
  parser.add_argument(
      '--no-waiver',
      action='store_true',
      dest='no_waiver',
      help=(
          'Ignore NO_UNIT_GUIDE / SKIP_UNIT_GUIDE from the commit message and'
          ' the environment. For a caller that applies the waiver itself and'
          ' must not have a stray tag in the surroundings suppress the rule.'
      ),
  )
  parser.add_argument(
      '--no-prefix-check',
      '--skip-prefix-check',
      action='store_true',
      dest='no_prefix_check',
      help="Skip the private-by-default '_' prefix check.",
  )
  parser.add_argument(
      'files',
      nargs='*',
      help='Explicit list of files to check (optional).',
  )
  return parser.parse_args(argv)


def read_added_files_list(path: str) -> set[str]:
  """Reads newline-delimited paths from `path`, ignoring blanks and comments."""
  with open(path, 'r', encoding='utf-8') as f:
    return {
        line.strip()
        for line in f
        if line.strip() and not line.lstrip().startswith('#')
    }


def main(argv: list[str]) -> int:
  args = _parse_args(argv)
  repo_root = args.new_dir

  if not _has_package_dir(repo_root):
    print(
        f'Error: new tree has no {_PACKAGE_RELPATH} directory: {repo_root}',
        file=sys.stderr,
    )
    return _EXIT_SETUP_ERROR

  # Only a VCS can supply a commit message, so this is '' when checking an
  # exported tree that has none. NO_UNIT_GUIDE comes from the environment
  # there instead -- see _GUIDE_VIOLATION_LINE.
  commit_msg = get_commit_message(repo_root)
  if args.added_files_from:
    if not os.path.isfile(args.added_files_from):
      print(
          f'Error: --added-files-from names no file: {args.added_files_from}',
          file=sys.stderr,
      )
      return _EXIT_SETUP_ERROR
    raw_added_files = read_added_files_list(args.added_files_from)
    raw_added_files.update(args.files)
  elif args.files:
    raw_added_files = set(args.files)
  elif args.baseline_dir:
    if not _has_package_dir(args.baseline_dir):
      print(
          'Error: baseline tree has no'
          f' {_PACKAGE_RELPATH} directory: {args.baseline_dir}',
          file=sys.stderr,
      )
      return _EXIT_SETUP_ERROR
    raw_added_files = added_py_files_from_baseline(repo_root, args.baseline_dir)
  else:
    vcs_added = get_vcs_added_files(repo_root)
    if vcs_added is None:
      print(
          'Could not determine the added files: no --baseline-dir or'
          ' --added-files-from was given, and no version control in'
          f' {os.path.abspath(repo_root)} could report them -- either none of'
          ' git/jj/hg/g4/p4 is active there, or the one that is could not'
          ' resolve what this change added (see above).\n'
          'This is not a clean bill of health -- nothing was checked. Pass'
          ' --baseline-dir or --added-files-from to say what to check.',
          file=sys.stderr,
      )
      return _EXIT_INDETERMINATE
    raw_added_files = vcs_added

  filtered_files = _normalize_and_filter_files(raw_added_files, repo_root)

  # A caller that names the files itself has already decided they are library
  # sources, so filtering every one of them away means the two disagree about
  # where the package is, not that there is nothing to check. Reporting that as
  # success is the failure this whole check exists to prevent, so say so
  # instead. The other modes legitimately filter everything away -- a change
  # that adds only tests, for one -- and are left alone.
  if args.added_files_from and raw_added_files and not filtered_files:
    print(
        'Error: none of the'
        f' {len(raw_added_files)} path(s) in {args.added_files_from} were'
        ' recognized as library sources under'
        f' {_PACKAGE_RELPATH}, so nothing was checked. This is a bug in how'
        ' the caller and this script locate the package, not a clean result.',
        file=sys.stderr,
    )
    return _EXIT_SETUP_ERROR

  prefix_errors, guide_errors = check_files(
      filtered_files,
      repo_root=repo_root,
      commit_msg=commit_msg,
      skip_unit_guide=args.no_unit_guide,
      skip_prefix=args.no_prefix_check,
      ignore_waiver=args.no_waiver,
  )

  for err in prefix_errors:
    print(err, file=sys.stderr)
  for err in guide_errors:
    print(err, file=sys.stderr)

  return _EXIT_VIOLATIONS if (prefix_errors or guide_errors) else _EXIT_OK


def run(argv: list[str]) -> int:
  """Runs main(), turning any crash into a setup error rather than a violation.

  An unhandled exception would exit 1, which is this script's code for "the
  rules were checked and the change breaks one" -- so a caller would report a
  violation, and offer whatever remedy it offers, for a check that never ran.
  Every other way of failing to check already reports itself as a setup error;
  a crash has to do the same.

  Args:
    argv: The argument list, without the program name.

  Returns:
    main()'s exit code, or the setup-error code if it raised.
  """
  try:
    return main(argv)
  except Exception:  # pylint: disable=broad-except
    traceback.print_exc()
    print(
        'Error: this check crashed, so the conventions were never verified.'
        ' That is a failure of the check itself, not of the change.',
        file=sys.stderr,
    )
    return _EXIT_SETUP_ERROR


if __name__ == '__main__':
  sys.exit(run(sys.argv[1:]))
