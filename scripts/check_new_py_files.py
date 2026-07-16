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

"""Checks that newly-added Python files under src/google/adk/ have a '_' prefix.

ADK is private-by-default: a newly-added Python file under src/google/adk/ must
have a '_'-prefixed basename. To make it public, add the symbol to the package
__init__.py / __all__ instead. See
.agents/skills/adk-style/references/visibility.md.

Newly-added files are detected by diffing the working tree against a baseline
source tree (e.g. an origin/main checkout), so it works in a checkout that has
no local git history:

  python scripts/check_new_py_files.py --baseline-dir /path/to/origin-main

Exit codes: 0 = ok, 1 = violation(s) found, 2 = usage/setup error.
"""

from __future__ import annotations

import argparse
import os
import sys

_PACKAGE_RELPATH = os.path.join('src', 'google', 'adk')

_VIOLATION_LINE = "Error: New Python file '{path}' must have a '_' prefix."

_GUIDANCE = (
    'All new Python files in src/google/adk/ must be private by default.\n'
    'To expose a public interface, use __init__.py and list public symbols'
    ' in __all__.\n'
    'See .agents/skills/adk-style/references/visibility.md for details.'
)

# Subtrees that may exist in the working tree but are intentionally absent from
# the baseline tree; ignore them so the diff does not report them as newly
# added.
_IGNORED_PREFIXES = (
    'src/google/adk/internal/',
    'src/google/adk/v1/',
    'src/google/adk/platform/internal/',
)


def find_py_files(root: str) -> set[str]:
  """Returns root-relative paths of every *.py under <root>/src/google/adk.

  Each path includes the src/google/adk/ prefix (e.g.
  'src/google/adk/agents/foo.py'). Symlinks are followed so that a src/google/adk
  tree assembled from symlinked subdirectories is walked correctly.
  """
  package_root = os.path.join(root, _PACKAGE_RELPATH)
  found: set[str] = set()
  for dirpath, _, filenames in os.walk(package_root, followlinks=True):
    for name in filenames:
      if name.endswith('.py'):
        abs_path = os.path.join(dirpath, name)
        found.add(os.path.relpath(abs_path, root))
  return found


def _should_check(relpath: str) -> bool:
  """Returns False for paths under an ignored prefix."""
  return not any(relpath.startswith(prefix) for prefix in _IGNORED_PREFIXES)


def added_py_files(new_root: str, baseline_root: str) -> set[str]:
  """Returns .py files present in new_root but not in baseline_root.

  Paths under _IGNORED_PREFIXES are skipped: they may exist in the working tree
  but are intentionally absent from the baseline, so a plain diff would
  otherwise report them as newly added.
  """
  added = find_py_files(new_root) - find_py_files(baseline_root)
  return {path for path in added if _should_check(path)}


def find_violations(added: set[str]) -> list[str]:
  """Returns the sorted added files whose basename does not start with '_'."""
  return sorted(
      path for path in added if not os.path.basename(path).startswith('_')
  )


def _has_package_dir(root: str) -> bool:
  return os.path.isdir(os.path.join(root, _PACKAGE_RELPATH))


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--baseline-dir',
      required=True,
      help='Baseline source tree to diff against (an origin/main checkout).',
  )
  parser.add_argument(
      '--new-dir',
      default='.',
      help='New source tree to check (default: current directory).',
  )
  return parser.parse_args(argv)


def main(argv: list[str]) -> int:
  args = _parse_args(argv)
  for label, root in (('baseline', args.baseline_dir), ('new', args.new_dir)):
    if not _has_package_dir(root):
      print(
          f'Error: {label} tree has no {_PACKAGE_RELPATH} directory: {root}',
          file=sys.stderr,
      )
      return 2

  violations = find_violations(added_py_files(args.new_dir, args.baseline_dir))
  for path in violations:
    print(_VIOLATION_LINE.format(path=path), file=sys.stderr)
  if violations:
    print(_GUIDANCE, file=sys.stderr)
  return 1 if violations else 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
