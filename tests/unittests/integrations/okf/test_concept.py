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

"""Tests for OKF concept documents (SPEC §4)."""

from __future__ import annotations

from google.adk.integrations.okf._concept import Concept
from google.adk.integrations.okf._concept import NonConformantError
from google.adk.integrations.okf._concept import split_frontmatter
import pytest

VALID = "---\ntype: Metric\ntitle: MRR\n---\n# Definition\nbody\n"


def test_split_frontmatter_returns_mapping_and_body():
  fm, body = split_frontmatter(VALID)
  assert fm == {"type": "Metric", "title": "MRR"}
  assert body.strip() == "# Definition\nbody"


def test_concept_exposes_type_and_falls_back_to_filename_for_title(tmp_path):
  p = tmp_path / "revenue.md"
  p.write_text("---\ntype: Metric\n---\nbody\n", encoding="utf-8")
  c = Concept.from_path(p, root=tmp_path)
  assert c.type == "Metric"
  assert c.title == "revenue"  # §4.1: derive from filename when absent


def test_missing_frontmatter_raises(tmp_path):
  p = tmp_path / "bare.md"
  p.write_text("# no frontmatter\n", encoding="utf-8")
  with pytest.raises(NonConformantError):
    Concept.from_path(p, root=tmp_path)


def test_empty_type_is_non_conformant(tmp_path):
  p = tmp_path / "x.md"
  p.write_text("---\ntype: ''\n---\nbody\n", encoding="utf-8")
  with pytest.raises(NonConformantError):
    Concept.from_path(p, root=tmp_path)


def test_unknown_keys_are_preserved(tmp_path):
  p = tmp_path / "x.md"
  p.write_text("---\ntype: Metric\nnot: a-real-key\n---\nb\n", encoding="utf-8")
  assert Concept.from_path(p, root=tmp_path).frontmatter["not"] == "a-real-key"


def test_unparseable_yaml_is_non_conformant(tmp_path):
  p = tmp_path / "bad.md"
  p.write_text("---\ntype: [unclosed\n---\nb\n", encoding="utf-8")
  with pytest.raises(NonConformantError):
    Concept.from_path(p, root=tmp_path)


def test_tags_and_resource_accessors(tmp_path):
  p = tmp_path / "m.md"
  p.write_text(
      "---\ntype: Metric\ntags: [a, b]\nresource: dashboard://x\n---\nb\n",
      encoding="utf-8",
  )
  c = Concept.from_path(p, root=tmp_path)
  assert c.tags == ("a", "b")
  assert c.resource == "dashboard://x"


def test_reserved_filenames_are_detected_at_any_level(tmp_path):
  from google.adk.integrations.okf._concept import is_reserved

  assert is_reserved(tmp_path / "index.md")
  assert is_reserved(tmp_path / "deep" / "log.md")
  assert not is_reserved(tmp_path / "concept.md")
