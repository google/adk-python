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

# The unit guide waiver, as it appears in a commit message: its own line, in
# `KEY=<reason>` form, flush left and with no space before the `=`. Matching
# the bare word anywhere in the text instead would waive the rule for any
# change whose message merely discusses it -- the change that introduced this
# check waived itself that way. The shape is deliberately no looser than the
# one a tag parser accepts: waiving locally on a line that the surrounding
# tooling would not read as a tag is how an author ends up believing they are
# covered when they are not.
_NO_UNIT_GUIDE_TAG = re.compile(r'^(?:NO|SKIP)_UNIT_GUIDE=', re.MULTILINE)

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
    " 'NO_UNIT_GUIDE=<reason>' tag in your commit message, or by setting"
    " NO_UNIT_GUIDE='<reason>' in the environment where this check runs.\n"
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
      _, staged = _run_cmd(
          ['git', 'diff', '--cached', '--name-only', '--diff-filter=A'],
          cwd=root,
      )
      if staged:
        return {f for f in staged.splitlines() if f.strip()}
      range_code, head_diff = _run_cmd(
          ['git', 'diff', _GIT_HEAD_RANGE, '--name-only', '--diff-filter=A'],
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
      if head_diff:
        return {f for f in head_diff.splitlines() if f.strip()}
      return set()

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
  """Retrieves commit message or description from VCS."""
  # 1. git
  if shutil.which('git'):
    code, _ = _run_cmd(['git', 'rev-parse', '--is-inside-work-tree'], cwd=root)
    if code == 0:
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
      _, git_dir = _run_cmd(['git', 'rev-parse', '--git-dir'], cwd=root)
      if git_dir:
        editmsg_path = (
            os.path.join(root, git_dir, 'COMMIT_EDITMSG')
            if not os.path.isabs(git_dir)
            else os.path.join(git_dir, 'COMMIT_EDITMSG')
        )
        if os.path.isfile(editmsg_path):
          try:
            with open(editmsg_path, 'r', encoding='utf-8') as f:
              msg = f'{msg} {f.read()}'
          except OSError:
            pass
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
  """Checks if NO_UNIT_GUIDE / SKIP_UNIT_GUIDE is present in env or commit message."""
  if os.environ.get('NO_UNIT_GUIDE') or os.environ.get('SKIP_UNIT_GUIDE'):
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

    # Check ignored prefixes and exclude directories
    full_rel = os.path.join('src', 'google', 'adk', rel_to_adk).replace(
        os.sep, '/'
    )
    if not _should_check(full_rel):
      continue

    # Check excluded subdirectories, anchored at the package root.
    if rel_to_adk.split('/')[0] in _EXCLUDE_DIR_NAMES:
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
