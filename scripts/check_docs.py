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

"""Checks the documentation for broken links and guides missing from the index.

Two checks, both of which run offline over the checkout:

1. Link validity: every relative link and image target in a markdown file
   under docs/ or contributing/ resolves to a file that exists, and every
   '#fragment' resolves to a heading in the file it points at. Only ATX
   ('# Title') headings are recognized. Links with a scheme (https:, mailto:)
   are left alone, so the check never reaches the network and never fails
   because a third-party site is down.
2. Index coverage: every guide under docs/guides/ is listed in
   docs/guides/README.md, which is the only table of contents that directory
   has. A guide missing from it is reachable only by a reader who already knows
   the path.

Usage:
  python scripts/check_docs.py [--root DIR]

Exit codes: 0 = ok, 1 = problems found, 2 = usage/setup error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
import sys

_DOCS_RELPATH = 'docs'
# Directories whose markdown documents the product for a reader. The repository
# root is left out: its README links to files that only exist once the package
# has been assembled for publication.
_CHECKED_RELPATHS = (_DOCS_RELPATH, 'contributing')
_GUIDES_RELPATH = os.path.join('docs', 'guides')
_INDEX_RELPATH = os.path.join(_GUIDES_RELPATH, 'README.md')

_EXIT_OK = 0
_EXIT_PROBLEMS = 1
_EXIT_SETUP_ERROR = 2

# [text](target), with an optional title, and the <...> form of the target.
_INLINE_LINK_RE = re.compile(
    r'!?\[(?:[^\]\\]|\\.)*\]\(\s*<?([^)<>\s]*)>?(?:\s+[\'"][^\'"]*[\'"])?\s*\)'
)
# [label]: target, the definition half of a reference-style link.
_LINK_DEFINITION_RE = re.compile(r'^ {0,3}\[[^\]]+\]:\s*<?([^\s>]+)>?')
# href= and src= on raw HTML, which markdown allows inline.
_HTML_ATTR_RE = re.compile(r'(?:href|src)\s*=\s*["\']([^"\']+)["\']', re.I)
_HEADING_RE = re.compile(r'^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$')
_HTML_ANCHOR_RE = re.compile(
    r'<a\s[^>]*(?:name|id)\s*=\s*["\']([^"\']+)["\']', re.I
)
_FENCE_RE = re.compile(r'^ {0,3}(`{3,}|~{3,})\s*([^`]*)$')
_SCHEME_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.\-]*:')
_INLINE_CODE_RE = re.compile(r'`[^`]*`')


@dataclass(frozen=True)
class Problem:
  """One thing wrong with the docs.

  Attributes:
    path: Repo-relative path of the file the problem is in.
    detail: What is wrong, phrased so the fix is obvious.
  """

  path: str
  detail: str


def markdown_files(directory: str) -> list[str]:
  """Returns every *.md under `directory`, sorted, as absolute paths."""
  found: list[str] = []
  for dirpath, dirnames, filenames in os.walk(directory):
    dirnames[:] = [d for d in dirnames if d not in ('.git', '__pycache__')]
    found.extend(
        os.path.join(dirpath, name)
        for name in filenames
        if name.endswith('.md')
    )
  return sorted(found)


def _lines_outside_code(text: str) -> list[tuple[int, str]]:
  """Returns (line number, line) for lines that are not inside a code fence.

  A fence opens on a line that is nothing but three or more backticks or
  tildes plus an info string, and closes on a marker of the same character
  that is at least as long. Requiring the whole line to match keeps a fence
  quoted mid-sentence, as prose about markdown does, from opening a block
  that swallows the rest of the file.
  """
  result: list[tuple[int, str]] = []
  fence: str | None = None
  for number, line in enumerate(text.splitlines(), start=1):
    match = _FENCE_RE.match(line)
    if fence is None:
      if match:
        fence = match.group(1)
        continue
      result.append((number, line))
    elif (
        match
        and match.group(1)[0] == fence[0]
        and len(match.group(1)) >= len(fence)
        and not match.group(2).strip()
    ):
      fence = None
  return result


def anchors_in(text: str) -> set[str]:
  """Returns the fragment ids a reader can link to in `text`."""
  found: set[str] = set()
  counts: dict[str, int] = {}
  for _, line in _lines_outside_code(text):
    match = _HEADING_RE.match(line)
    if match:
      base = slugify(match.group(2))
      seen = counts.get(base, 0)
      counts[base] = seen + 1
      found.add(base if seen == 0 else f'{base}-{seen}')
    found.update(anchor.group(1) for anchor in _HTML_ANCHOR_RE.finditer(line))
  return found


def slugify(heading: str) -> str:
  """Returns the fragment id a markdown renderer derives from a heading."""
  text = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', heading)
  text = re.sub(r'<[^>]+>', '', text)
  # Underscore is left in: a renderer keeps it, and ADK headings name
  # snake_case identifiers far more often than they use _emphasis_.
  text = re.sub(r'[`*~]', '', text).strip().lower()
  text = re.sub(r'[^\w\- ]', '', text)
  return text.replace(' ', '-')


def links_in(text: str) -> list[tuple[int, str]]:
  """Returns (line number, target) for every link and image target in `text`."""
  found: list[tuple[int, str]] = []
  for number, line in _lines_outside_code(text):
    definition = _LINK_DEFINITION_RE.match(line)
    if definition:
      found.append((number, definition.group(1)))
      continue
    without_code = _INLINE_CODE_RE.sub('', line)
    found.extend(
        (number, match.group(1))
        for match in _INLINE_LINK_RE.finditer(without_code)
    )
    found.extend(
        (number, match.group(1))
        for match in _HTML_ATTR_RE.finditer(without_code)
    )
  return found


def _is_external(target: str) -> bool:
  return bool(_SCHEME_RE.match(target)) or target.startswith('//')


def check_links(root: str, paths: list[str]) -> list[Problem]:
  """Returns a problem for every link in `paths` that does not resolve."""
  problems: list[Problem] = []
  anchor_cache: dict[str, set[str]] = {}

  def anchors_of(path: str) -> set[str]:
    if path not in anchor_cache:
      with open(path, encoding='utf-8') as f:
        anchor_cache[path] = anchors_in(f.read())
    return anchor_cache[path]

  for path in paths:
    display = os.path.relpath(path, root)
    with open(path, encoding='utf-8') as f:
      text = f.read()
    for number, target in links_in(text):
      if not target or _is_external(target):
        continue
      if target.startswith('#'):
        destination, fragment = path, target[1:]
      else:
        relative, _, fragment = target.partition('#')
        destination = os.path.normpath(
            os.path.join(os.path.dirname(path), relative)
        )
        if not os.path.exists(destination):
          problems.append(
              Problem(display, f'line {number}: {target} does not exist')
          )
          continue
      if fragment and destination.endswith('.md'):
        if fragment not in anchors_of(destination):
          problems.append(
              Problem(
                  display,
                  f'line {number}: {target} names no heading in the file it'
                  ' points at',
              )
          )
  return problems


def check_index_coverage(root: str) -> list[Problem]:
  """Returns a problem for every guide the guides index does not list."""
  index_path = os.path.join(root, _INDEX_RELPATH)
  guides_dir = os.path.join(root, _GUIDES_RELPATH)
  if not os.path.isfile(index_path):
    return [Problem(_INDEX_RELPATH, 'the guides index is missing')]

  with open(index_path, encoding='utf-8') as f:
    text = f.read()
  listed = {
      os.path.normpath(os.path.join(guides_dir, target.partition('#')[0]))
      for _, target in links_in(text)
      if target and not _is_external(target) and not target.startswith('#')
  }
  missing = [
      path
      for path in markdown_files(guides_dir)
      if path != index_path and path not in listed
  ]
  return [
      Problem(
          os.path.relpath(index_path, root),
          f'{os.path.relpath(path, guides_dir)} is not listed in the index',
      )
      for path in missing
  ]


def check(root: str) -> list[Problem]:
  """Runs every documentation check against the checkout at `root`."""
  files: list[str] = []
  for relpath in _CHECKED_RELPATHS:
    directory = os.path.join(root, relpath)
    if os.path.isdir(directory):
      files.extend(markdown_files(directory))
  return check_links(root, files) + check_index_coverage(root)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--root',
      default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
      help='repository root to check (default: the repo this script is in)',
  )
  namespace = parser.parse_args(argv)
  root = os.path.abspath(namespace.root)
  if not os.path.isdir(os.path.join(root, _DOCS_RELPATH)):
    print(f'No docs/ directory under {root}', file=sys.stderr)
    return _EXIT_SETUP_ERROR

  problems = check(root)
  for problem in problems:
    print(f'{problem.path}: {problem.detail}')
  if problems:
    print(
        f'\n{len(problems)} documentation problem(s). A link is written'
        ' relative to the file it is in, and a new guide needs a line in'
        f' {_INDEX_RELPATH}.'
    )
    return _EXIT_PROBLEMS
  return _EXIT_OK


if __name__ == '__main__':
  sys.exit(main())
