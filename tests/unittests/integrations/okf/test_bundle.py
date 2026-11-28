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

"""Tests for bundle traversal, reserved-file partition, and path safety (SPEC §3/§6/§8)."""

from __future__ import annotations

from google.adk.integrations.okf._bundle import Bundle
from google.adk.integrations.okf._bundle import UnsafeBundlePath
import pytest


def _bundle(tmp_path):
  (tmp_path / "index.md").write_text(
      "# Index\n* [MRR](metrics/revenue.md) - rev\n", encoding="utf-8"
  )
  (tmp_path / "metrics").mkdir()
  (tmp_path / "metrics" / "revenue.md").write_text(
      "---\ntype: Metric\ntitle: MRR\n---\nbody\n", encoding="utf-8"
  )
  (tmp_path / "log.md").write_text(
      "# Dir Log\n\n## 2026-05-22\n* **Creation**: x\n", encoding="utf-8"
  )
  return tmp_path


def test_concepts_exclude_reserved_files(tmp_path):
  b = Bundle.open(_bundle(tmp_path))
  assert [c.rel_path for c in b.concepts()] == ["metrics/revenue.md"]
  assert sorted(b.reserved_files()) == ["index.md", "log.md"]


def test_index_md_entries_are_parsed(tmp_path):
  b = Bundle.open(_bundle(tmp_path))
  entries = b.index_entries()
  assert entries[0].title == "MRR" and entries[0].target == "metrics/revenue.md"


def test_broken_links_are_tolerated(tmp_path):
  """SPEC §6.1 MUST."""
  (tmp_path / "metrics").mkdir()
  (tmp_path / "metrics" / "a.md").write_text(
      "---\ntype: Metric\n---\nsee [gone](/metrics/nope.md)\n", encoding="utf-8"
  )
  b = Bundle.open(tmp_path)
  assert b.broken_links() == [
      ("metrics/a.md", "/metrics/nope.md")
  ]  # reported, not raised


def test_missing_index_is_tolerated_and_synthesizable(tmp_path):
  """SPEC §8."""
  (tmp_path / "a.md").write_text(
      "---\ntype: Metric\n---\nb\n", encoding="utf-8"
  )
  b = Bundle.open(tmp_path)
  assert b.index_entries() == []
  assert "a.md" in b.synthesize_index()


def test_resolve_inside_bundle_is_relative_to_root(tmp_path):
  (tmp_path / "tables").mkdir()
  (tmp_path / "tables" / "t.md").write_text(
      "---\ntype: Table\n---\nb\n", encoding="utf-8"
  )
  assert Bundle.open(tmp_path).resolve("/tables/t.md").name == "t.md"


def test_resolve_rejects_traversal(tmp_path):
  b = Bundle.open(tmp_path)
  with pytest.raises(UnsafeBundlePath):
    b.resolve("../../../../etc/passwd")


def test_absolute_path_is_bundle_relative_not_host_absolute(tmp_path):
  """§6.2: a path starting `/` is relative to the BUNDLE root, not the host root.

  This is a safety property, not just a parsing one: `/etc/passwd` must never
  reach the host's file.
  """
  b = Bundle.open(tmp_path)
  resolved = b.resolve("/etc/passwd")
  assert b.root in resolved.parents, "resolved outside the bundle root"
  assert not resolved.exists(), "must not resolve to the host's /etc/passwd"


def test_symlink_escaping_the_bundle_is_refused(tmp_path):
  """The real traversal threat: a symlink inside the bundle pointing outside it."""
  outside = tmp_path / "outside_secret.txt"
  outside.write_text("host secret", encoding="utf-8")
  root = tmp_path / "bundle"
  root.mkdir()
  (root / "a.md").write_text("---\ntype: Metric\n---\nb\n", encoding="utf-8")
  try:
    (root / "escape").symlink_to(outside)
  except OSError:  # pragma: no cover - platform without symlink support
    pytest.skip("symlinks unavailable on this platform")

  with pytest.raises(UnsafeBundlePath):
    Bundle.open(root).resolve("escape")


def test_non_conformant_file_is_recorded_not_raised(tmp_path):
  """One bad file must not hide the rest of the bundle."""
  (tmp_path / "good.md").write_text(
      "---\ntype: Metric\n---\nb\n", encoding="utf-8"
  )
  (tmp_path / "bad.md").write_text("# no frontmatter\n", encoding="utf-8")
  (tmp_path / "binary.md").write_bytes(b"---\ntype: Metric\n---\n\xff\xfe")
  b = Bundle.open(tmp_path)
  assert [c.rel_path for c in b.concepts()] == ["good.md"]
  assert [v[0] for v in b.violations] == ["bad.md", "binary.md"]


def test_symlinked_concept_escaping_the_bundle_is_not_loaded(tmp_path):
  """Eager loading must refuse a symlinked concept that points outside the root."""
  outside = tmp_path / "outside.md"
  outside.write_text(
      "---\ntype: Metric\ntitle: Host Secret\n---\nHOST SECRET\n",
      encoding="utf-8",
  )
  root = tmp_path / "bundle"
  root.mkdir()
  (root / "good.md").write_text("---\ntype: Metric\n---\nb\n", encoding="utf-8")
  try:
    (root / "leak.md").symlink_to(outside)
  except OSError:  # pragma: no cover - platform without symlink support
    pytest.skip("symlinks unavailable on this platform")

  b = Bundle.open(root)
  assert [c.rel_path for c in b.concepts()] == ["good.md"]
  assert b.violations == [("leak.md", "symlink escapes bundle root")]
  assert "HOST SECRET" not in "\n".join(c.body for c in b.concepts())


def test_by_rel_path_accepts_leading_slash(tmp_path):
  b = Bundle.open(_bundle(tmp_path))
  assert b.by_rel_path("/metrics/revenue.md") is not None
  assert b.by_rel_path("nope.md") is None
