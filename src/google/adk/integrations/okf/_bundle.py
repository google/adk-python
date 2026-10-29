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

"""Bundle traversal and path resolution (SPEC §3, §6, §8). ADK-free.

One bad file must never hide the rest of a bundle, and one hostile path must never
escape it. Both properties are enforced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from ._concept import Concept
from ._concept import is_reserved
from ._concept import NonConformantError

_INDEX_ENTRY_RE = re.compile(
    r"^\s*[*-]\s+\[(?P<title>[^\]]+)\]\((?P<target>[^)]+)\)\s*-?\s*(?P<desc>.*)$"
)
_LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)]+)\)")


class UnsafeBundlePath(ValueError):
  """A resolved path escaped the bundle root.

  Callers must surface this, never swallow it: it is the boundary between the
  bundle and the host filesystem.
  """


@dataclass(frozen=True)
class IndexEntry:
  """One ``* [Title](target) - description`` line from an ``index.md`` (SPEC §8)."""

  title: str
  target: str
  description: str


class Bundle:
  """A directory tree of OKF concept documents.

  Loading never raises for a non-conformant file: violations are collected on
  :attr:`violations` so a consumer can report them all at once (§11 forbids
  rejecting a bundle for anything beyond rules 1-3).
  """

  def __init__(self, root: Path) -> None:
    self.root = root.resolve()
    self._concepts: list[Concept] = []
    self.violations: list[tuple[str, str]] = []
    for path in sorted(self.root.rglob("*.md")):
      if is_reserved(path):
        continue
      rel_path = path.relative_to(self.root).as_posix()
      # A symlink inside the bundle must not smuggle a host file in at
      # load time. `Bundle.resolve` refuses escapes for explicit reads;
      # this closes the same hole on the eager-loading path.
      try:
        resolved = path.resolve()
      except (OSError, RuntimeError) as exc:
        self.violations.append((rel_path, f"unreadable path: {exc}"))
        continue
      if resolved != self.root and self.root not in resolved.parents:
        self.violations.append((rel_path, "symlink escapes bundle root"))
        continue
      if not resolved.is_file():
        self.violations.append((rel_path, "not a regular file"))
        continue
      try:
        self._concepts.append(Concept.from_path(path, self.root))
      except (NonConformantError, OSError, UnicodeDecodeError) as exc:
        self.violations.append((rel_path, str(exc)))

  @classmethod
  def open(cls, root: str | Path) -> Bundle:
    """Load the bundle rooted at ``root``."""
    return cls(Path(root))

  # ------------------------------------------------------------------ concepts

  def concepts(self) -> list[Concept]:
    """Every concept document, in stable path order."""
    return list(self._concepts)

  def by_rel_path(self, rel_path: str) -> Concept | None:
    """Look up a concept by bundle-relative path, tolerating a leading ``/``."""
    want = rel_path.lstrip("/")
    return next((c for c in self._concepts if c.rel_path == want), None)

  def reserved_files(self) -> list[str]:
    """Bundle-relative paths of every reserved ``index.md``/``log.md``."""
    return [
        path.relative_to(self.root).as_posix()
        for path in sorted(self.root.rglob("*.md"))
        if is_reserved(path)
    ]

  # --------------------------------------------------------------------- paths

  def resolve(self, bundle_path: str) -> Path:
    """Resolve a §6.2 path value to a real path **inside** this bundle.

    Accepts the §6.2 grammar: a bundle-relative path beginning ``/`` or a
    relative path. Absolute URLs are not bundle paths and are rejected.

    Raises:
        UnsafeBundlePath: the target escapes the bundle root, including via a
            symlink that points outside it.
    """
    if "://" in bundle_path:
      raise UnsafeBundlePath(
          f"not a bundle path (absolute URL): {bundle_path!r}"
      )
    candidate = (self.root / bundle_path.lstrip("/")).resolve()
    if candidate != self.root and self.root not in candidate.parents:
      raise UnsafeBundlePath(f"escapes bundle root: {bundle_path!r}")
    return candidate

  # ------------------------------------------------------------------- indexes

  def index_entries(self) -> list[IndexEntry]:
    """Parse the bundle-root ``index.md`` (SPEC §8). Empty when there is none."""
    index = self.root / "index.md"
    if not index.is_file():
      return []
    entries: list[IndexEntry] = []
    for line in index.read_text(encoding="utf-8").splitlines():
      match = _INDEX_ENTRY_RE.match(line)
      if match:
        entries.append(
            IndexEntry(match["title"], match["target"], match["desc"].strip())
        )
    return entries

  def synthesize_index(self) -> str:
    """Build a bundle-root ``index.md`` body on the fly when none is present (§8 permits)."""
    lines = ["# Concepts", ""]
    for concept in sorted(self._concepts, key=lambda c: c.rel_path):
      description = f" - {concept.description}" if concept.description else ""
      lines.append(f"* [{concept.title}](/{concept.rel_path}){description}")
    return "\n".join(lines) + "\n"

  # --------------------------------------------------------------------- links

  def broken_links(self) -> list[tuple[str, str]]:
    """``(source_rel_path, target)`` for body links with no in-bundle target.

    Report-only: SPEC §6.1 requires consumers to tolerate broken links, which
    may simply represent knowledge not written yet.
    """
    known = {concept.rel_path for concept in self._concepts}
    broken: list[tuple[str, str]] = []
    for concept in self._concepts:
      for match in _LINK_RE.finditer(concept.body):
        target = match["target"]
        if "://" in target:
          continue
        if target.lstrip("/") not in known:
          broken.append((concept.rel_path, target))
    return broken
