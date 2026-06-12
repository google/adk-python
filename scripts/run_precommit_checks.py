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

"""Runs the .pre-commit-config.yaml checks WITHOUT the pre-commit framework.

`pre-commit` requires a git repository, so it cannot run in environments such as
a piper checkout that has no .git directory. This script parses
.pre-commit-config.yaml and invokes each hook's underlying tool directly.

To minimize maintenance, the file lists, the global/per-hook exclude and
`files` patterns, and per-hook `args` are all read from the config. Only the
hook-id -> CLI mapping (and each hook's implicit file-type filter) is kept here,
in _HOOK_SPECS, since pre-commit derives those from each hook's repo definition
rather than from the config.

Usage:
  python scripts/run_precommit_checks.py [--check] [PATH ...]

  --check    Only verify; do not modify files, and exit non-zero if changes are
             needed (like CI). By default fixes are applied in place.
  PATH ...   Files/dirs to check, interpreted relative to the repo root (not the
             current directory). Defaults to the source trees (src, tests,
             contributing) plus pyproject.toml. The script can be run from any
             directory.

Install the tools first (matching .pre-commit-config.yaml):
  uv sync --extra dev
  # addlicense is a Go binary: go install github.com/google/addlicense@latest
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import os
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, '.pre-commit-config.yaml')

# Paths to check by default. Limited to these so the walk never touches .venv,
# build artifacts, or other non-source files (pre-commit avoids them by only
# seeing git-tracked files, which we cannot query in a no-git checkout).
_DEFAULT_TARGETS = ('src', 'tests', 'contributing', 'pyproject.toml')

_PY = r'\.py$'


@dataclass(frozen=True)
class HookSpec:
  """How to run a standard pre-commit hook id as a direct tool invocation.

  Attributes:
    check_cmd: Command that verifies formatting; a non-zero exit means changes
      are needed. The matched files are appended.
    fix_cmd: Command that applies fixes in place. None for check-only hooks.
    type_filter: Regex for the hook's implicit file-type restriction (the
      ``types`` default declared in the hook's own repo definition, which is not
      present in our config). None means it accepts every file.
    is_fixer: True for tools that always rewrite in place and have no check
      mode; check mode is emulated by diffing against a temporary copy.
    text_only: True for hooks that should skip binary files (pre-commit's
      ``types: [text]`` default, e.g. trailing-whitespace/end-of-file-fixer);
      detected by content so images, PDFs, etc. are never modified.
  """

  check_cmd: list[str]
  fix_cmd: list[str] | None = None
  type_filter: str | None = None
  is_fixer: bool = False
  text_only: bool = False


# hook id -> how to run it. `type_filter` mirrors each hook's `types:` default
# from its .pre-commit-hooks.yaml. The `local` hooks (addlicense,
# check-new-py-prefix) are handled by _LOCAL_HOOKS below instead.
_HOOK_SPECS: dict[str, HookSpec] = {
    'ruff': HookSpec(
        ['ruff', 'check', '--force-exclude'],
        ['ruff', 'check', '--fix', '--force-exclude'],
        _PY,
    ),
    'isort': HookSpec(['isort', '--check-only', '--diff'], ['isort'], _PY),
    'pyink': HookSpec(['pyink', '--check', '--diff'], ['pyink'], _PY),
    'pyproject-fmt': HookSpec(
        ['pyproject-fmt', '--check'],
        ['pyproject-fmt'],
        r'(^|/)pyproject\.toml$',
    ),
    'mdformat': HookSpec(['mdformat', '--check'], ['mdformat']),
    'check-yaml': HookSpec(['check-yaml'], type_filter=r'\.ya?ml$'),
    'end-of-file-fixer': HookSpec(
        ['end-of-file-fixer'], is_fixer=True, text_only=True
    ),
    'trailing-whitespace': HookSpec(
        ['trailing-whitespace-fixer'], is_fixer=True, text_only=True
    ),
}


@dataclass(frozen=True)
class Hook:
  """A single hook entry parsed from .pre-commit-config.yaml."""

  hook_id: str
  files: re.Pattern | None
  exclude: re.Pattern | None
  args: list[str]


# ----------------------------------------------------------------------------
# Config parsing and file selection
# ----------------------------------------------------------------------------


def load_config() -> tuple[list[Hook], re.Pattern | None]:
  """Returns (hooks, global_exclude) parsed from .pre-commit-config.yaml."""
  with open(_CONFIG_PATH, encoding='utf-8') as f:
    config = yaml.safe_load(f)

  def compile_opt(pattern: str | None) -> re.Pattern | None:
    return re.compile(pattern) if pattern else None

  hooks = [
      Hook(
          hook_id=hook['id'],
          files=compile_opt(hook.get('files')),
          exclude=compile_opt(hook.get('exclude')),
          args=hook.get('args', []),
      )
      for repo in config.get('repos', [])
      for hook in repo.get('hooks', [])
  ]
  return hooks, compile_opt(config.get('exclude'))


def collect_files(
    targets: list[str], global_exclude: re.Pattern | None
) -> list[str]:
  """Returns repo-relative file paths under targets, minus global excludes.

  Symlinks are never followed: a symlinked file is skipped and ``os.walk`` runs
  with ``followlinks=False``, so symlinked directories are not descended into.
  This avoids linting code outside the repo (e.g. a piper checkout where
  src/google/adk/a2a links to a parent folder) and the infinite recursion that
  would occur if such a link points to an ancestor of the repo.
  """
  files: set[str] = set()
  for target in targets:
    abs_target = os.path.join(_REPO_ROOT, target)
    if os.path.islink(abs_target):
      continue
    if os.path.isfile(abs_target):
      files.add(os.path.relpath(abs_target, _REPO_ROOT))
    elif os.path.isdir(abs_target):
      _walk_into(abs_target, files)

  result = sorted(files)
  if global_exclude is not None:
    result = [f for f in result if not global_exclude.search(f)]
  return result


def _walk_into(directory: str, files: set[str]) -> None:
  """Adds non-symlink files under `directory` (relative to repo) into `files`."""
  for root, dirs, names in os.walk(directory, followlinks=False):
    if '__pycache__' in root or '/.git' in root:
      continue
    # Prune symlinked subdirectories so they are never descended into.
    dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
    for name in names:
      path = os.path.join(root, name)
      if not os.path.islink(path):
        files.add(os.path.relpath(path, _REPO_ROOT))


def _is_binary(rel_path: str) -> bool:
  """Returns True if the file looks binary (contains a NUL byte).

  This mirrors how pre-commit decides text vs binary, so binary files (images,
  PDFs, compiled artifacts) are never modified by text fixers.
  """
  try:
    with open(os.path.join(_REPO_ROOT, rel_path), 'rb') as f:
      return b'\x00' in f.read(8192)
  except OSError:
    return True


def files_for_hook(
    hook: Hook, candidates: list[str], spec: HookSpec
) -> list[str]:
  """Returns the files a hook applies to, honoring config + implicit filters."""
  selected = candidates
  if hook.files is not None:
    selected = [f for f in selected if hook.files.search(f)]
  if hook.exclude is not None:
    selected = [f for f in selected if not hook.exclude.search(f)]
  if spec.type_filter is not None:
    type_re = re.compile(spec.type_filter)
    selected = [f for f in selected if type_re.search(f)]
  if spec.text_only:
    selected = [f for f in selected if not _is_binary(f)]
  return selected


# ----------------------------------------------------------------------------
# Running hooks
# ----------------------------------------------------------------------------


def _exec(cmd: list[str]) -> bool:
  """Runs a single command in the repo root, echoing output; True on exit 0."""
  proc = subprocess.run(
      cmd, cwd=_REPO_ROOT, check=False, capture_output=True, text=True
  )
  output = (proc.stdout or '') + (proc.stderr or '')
  if output.strip():
    print(output.rstrip())
  return proc.returncode == 0


def _run(prefix: list[str], files: list[str]) -> bool:
  """Runs `prefix` over `files`, batching to stay under the OS arg limit.

  A whole-repo run can pass thousands of paths, which overflows ARG_MAX. The
  files are split into chunks and the command is invoked once per chunk; the
  result is the AND of all chunks.
  """
  if not files:
    return _exec(prefix)
  ok = True
  for batch in _batched(files):
    ok = _exec(prefix + batch) and ok
  return ok


def _batched(files: list[str]) -> list[list[str]]:
  """Splits files into chunks small enough to fit a single command line."""
  # Stay well under ARG_MAX (bytes) with headroom for the command prefix and
  # the environment block; also cap the count as a simple safety net.
  try:
    arg_max = os.sysconf('SC_ARG_MAX')
  except (ValueError, OSError):
    arg_max = 256 * 1024
  budget = max(arg_max // 2, 64 * 1024)
  batches: list[list[str]] = []
  current: list[str] = []
  size = 0
  for f in files:
    item = len(f) + 1  # path length plus the separating NUL/space.
    if current and (size + item > budget or len(current) >= 1000):
      batches.append(current)
      current, size = [], 0
    current.append(f)
    size += item
  if current:
    batches.append(current)
  return batches


def _run_fixer_in_check_mode(tool: str, files: list[str]) -> bool:
  """Emulates check mode for in-place fixers by diffing against a copy."""
  ok = True
  with tempfile.TemporaryDirectory() as tmp:
    for f in files:
      original = os.path.join(_REPO_ROOT, f)
      copy = os.path.join(tmp, f.replace('/', '_'))
      shutil.copyfile(original, copy)
      subprocess.run([tool, copy], check=False, capture_output=True)
      if not _same_contents(original, copy):
        print(f'Would reformat: {f}')
        ok = False
  return ok


def _same_contents(a: str, b: str) -> bool:
  with open(a, 'rb') as fa, open(b, 'rb') as fb:
    return fa.read() == fb.read()


# Each runner returns True (ran, passed), False (ran, failed), or None
# (skipped / no matching files -- it already printed its own status line).
HookResult = bool | None


def run_standard_hook(
    hook: Hook, candidates: list[str], fix: bool
) -> HookResult:
  """Runs a hook backed by an entry in _HOOK_SPECS."""
  spec = _HOOK_SPECS[hook.hook_id]
  tool = spec.check_cmd[0]
  if not shutil.which(tool):
    print(f"SKIPPED: '{tool}' not installed")
    return None

  files = files_for_hook(hook, candidates, spec)
  if not files:
    print('no matching files')
    return None

  if spec.is_fixer and not fix:
    return _run_fixer_in_check_mode(tool, files)
  command = spec.fix_cmd if (fix and spec.fix_cmd) else spec.check_cmd
  # Drop `--fix` from the config args: check mode must not modify files, and
  # fix mode already gets `--fix` from the spec's fix_cmd (passing it twice is
  # an error, e.g. ruff rejects a repeated `--fix`).
  args = [a for a in hook.args if a != '--fix']
  return _run(command + args, files)


# --- local hooks (no upstream tool; bespoke handling) -----------------------


def run_addlicense(hook: Hook, candidates: list[str], fix: bool) -> HookResult:
  """Adds/checks Apache license headers (the `addlicense` Go binary)."""
  if not shutil.which('addlicense'):
    print("SKIPPED: 'addlicense' not installed")
    return None
  files = files_for_hook(hook, candidates, HookSpec(check_cmd=['addlicense']))
  if not files:
    print('no matching files')
    return None
  base = ['addlicense', '-c', 'Google LLC', '-l', 'apache']
  return _run(base if fix else base + ['-check'], files)


def skip_git_only_hook(
    hook: Hook, candidates: list[str], fix: bool
) -> HookResult:
  """Skips a hook that needs git (cannot run in a no-git checkout)."""
  del hook, candidates, fix  # Unused; signature matches the local protocol.
  print('SKIPPED: requires git (detects newly-added files via git diff).')
  print("Manually ensure new files under src/google/adk/ start with '_'.")
  return None


_LocalHookRunner = Callable[[Hook, list[str], bool], HookResult]
_LOCAL_HOOKS: dict[str, _LocalHookRunner] = {
    'addlicense': run_addlicense,
    'check-new-py-prefix': skip_git_only_hook,
}


def run_hook(hook: Hook, candidates: list[str], fix: bool) -> HookResult:
  """Runs one hook (standard or local), printing a header and result."""
  print(f'\n=== {hook.hook_id} ===')
  if hook.hook_id in _LOCAL_HOOKS:
    result = _LOCAL_HOOKS[hook.hook_id](hook, candidates, fix)
  elif hook.hook_id in _HOOK_SPECS:
    result = run_standard_hook(hook, candidates, fix)
  else:
    print('SKIPPED: unknown hook id (add it to _HOOK_SPECS).')
    return False
  if result is True:
    print('OK')
  return result


@dataclass
class Report:
  failures: list[str] = field(default_factory=list)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--check',
      action='store_true',
      help='only verify; do not modify files (default: apply fixes in place)',
  )
  parser.add_argument(
      'paths',
      nargs='*',
      help=(
          'files/dirs relative to the repo root '
          '(default: src, tests, contributing, pyproject.toml)'
      ),
  )
  ns = parser.parse_args()

  fix = not ns.check
  hooks, global_exclude = load_config()
  candidates = collect_files(ns.paths or list(_DEFAULT_TARGETS), global_exclude)

  report = Report()
  for hook in hooks:
    if run_hook(hook, candidates, fix) is False:
      report.failures.append(hook.hook_id)

  print('\n=== Summary ===')
  if report.failures:
    print('FAILED: ' + ', '.join(report.failures))
    if ns.check:
      print('Re-run without --check to auto-fix where possible.')
    return 1
  print('All checks passed.')
  return 0


if __name__ == '__main__':
  sys.exit(main())
