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

"""OKF v0.2 concept documents (SPEC §4).

This module is deliberately ADK-free: importing it must never require
``google.adk``. Authority: OKF v0.2 SPEC.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import re
from typing import Any

import yaml

#: Reserved at any level of the hierarchy (SPEC §3.1). MUST NOT be concept documents.
RESERVED_FILENAMES = frozenset({"index.md", "log.md"})

#: Every frontmatter key OKF v0.2 itself defines (SPEC §4.1, §5, §10).
#: Anything outside this set is a producer-defined extension, which consumers
#: MUST NOT reject and SHOULD preserve (SPEC §4.1).
OKF_KNOWN_KEYS = frozenset({
    # §4.1 core
    "type",
    "title",
    "description",
    "resource",
    "tags",
    # §5 provenance, trust, lifecycle
    "usage_window",
    "sources",
    "generated",
    "verified",
    "status",
    "stale_after",
    # §10 attested computation
    "runtime",
    "parameters",
    "computation",
    "executor",
    "attester",
})

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)


class NonConformantError(ValueError):
  """A file violates an OKF §11 hard rule (missing/unparseable frontmatter, or no ``type``)."""


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
  """Return ``(frontmatter_mapping, body)``.

  The block must open the file (SPEC §4). Raises :class:`NonConformantError`
  when the block is absent, unparseable, or not a mapping (§11 rule 1).
  """
  match = _FRONTMATTER_RE.match(text)
  if match is None:
    raise NonConformantError("no parseable YAML frontmatter block")
  try:
    parsed = yaml.safe_load(match.group(1))
  except yaml.YAMLError as exc:
    raise NonConformantError(f"unparseable YAML frontmatter: {exc}") from exc
  if parsed is None:
    parsed = {}
  if not isinstance(parsed, dict):
    raise NonConformantError("frontmatter is not a mapping")
  return parsed, text[match.end() :]


def is_reserved(path: Path) -> bool:
  """True for ``index.md`` / ``log.md`` — reserved at **any** level (SPEC §3.1)."""
  return path.name in RESERVED_FILENAMES


@dataclass(frozen=True)
class Concept:
  """A single unit of knowledge: frontmatter plus markdown body.

  The body is untrusted content — data, never instructions.
  """

  path: Path
  rel_path: str
  frontmatter: dict[str, Any]
  body: str
  extra_keys: tuple[str, ...] = field(default=())

  @classmethod
  def from_path(cls, path: Path, root: Path) -> Concept:
    """Parse ``path`` relative to bundle ``root``. Raises on §11 rules 1-2."""
    fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
    type_ = fm.get("type")
    if not isinstance(type_, str) or not type_.strip():  # §11 rule 2
      raise NonConformantError(
          f"{path}: missing or empty required 'type' field"
      )
    extras = tuple(sorted(k for k in fm if k not in OKF_KNOWN_KEYS))
    return cls(
        path=path,
        rel_path=path.relative_to(root).as_posix(),
        frontmatter=fm,
        body=body,
        extra_keys=extras,
    )

  @property
  def type(self) -> str:
    """Concept type (SPEC §4.1). Free string; never validated against a registry."""
    return str(self.frontmatter["type"])

  @property
  def title(self) -> str:
    """``title``, falling back to the filename stem (SPEC §4.1)."""
    raw = self.frontmatter.get("title")
    if isinstance(raw, str) and raw.strip():
      return raw
    return self.path.stem

  @property
  def description(self) -> str:
    return str(self.frontmatter.get("description", ""))

  @property
  def tags(self) -> tuple[str, ...]:
    raw = self.frontmatter.get("tags")
    return tuple(str(t) for t in raw) if isinstance(raw, list) else ()

  @property
  def resource(self) -> str | None:
    """Canonical URI for the underlying asset, if any (SPEC §4.1, §6.2)."""
    raw = self.frontmatter.get("resource")
    return str(raw) if isinstance(raw, str) else None

  @property
  def is_attested_computation(self) -> bool:
    """SPEC §10: the ``type`` string is the contract."""
    return self.type.strip().lower() == "attested computation"
