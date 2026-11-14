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

"""L2 ``okf_read`` and L3 ``okf_resource``: structured errors, guarded paths."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from google.adk.integrations.okf import OkfToolset
from google.adk.integrations.okf._toolset import MAX_RESOURCE_BYTES
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "bundles"


def _tools(bundle_dir: str = "acme_retail") -> dict:
  ts = OkfToolset(bundle=str(FIXTURES / bundle_dir), tool_name_prefix="okf")
  return {tool.name: tool for tool in asyncio.run(ts.get_tools_with_prefix())}


def _toolset(root: Path) -> dict:
  ts = OkfToolset(bundle=str(root), tool_name_prefix="okf")
  return {tool.name: tool for tool in asyncio.run(ts.get_tools_with_prefix())}


def test_read_returns_body_and_frontmatter():
  out = _tools()["okf_read"].func(path="/metrics/revenue.md")
  assert out["type"] == "Metric"
  assert out["trust_tier"] == "human-reviewed"
  assert "body" in out and "# Definition" in out["body"]
  assert out["frontmatter"]["title"] == "Revenue"
  assert out["content_is_untrusted_data"] is True
  json.dumps(out)  # ADK transports tool results as JSON


def test_read_accepts_relative_paths_too():
  out = _tools()["okf_read"].func(path="metrics/revenue.md")
  assert out["rel_path"] == "metrics/revenue.md"


def test_read_missing_path_returns_structured_error_not_exception():
  out = _tools()["okf_read"].func(path="/nope.md")
  assert out["error"] == "not_found"


def test_read_traversal_returns_unsafe_path_not_exception():
  out = _tools()["okf_read"].func(path="../../../../../etc/passwd")
  assert out["error"] == "unsafe_path"


def test_read_host_absolute_path_stays_inside_the_bundle():
  """SPEC §6.2: a leading slash is bundle-relative, never host-absolute."""
  out = _tools()["okf_read"].func(path="/etc/passwd")
  assert out["error"] == "not_found"
  assert "root:" not in json.dumps(out)


def test_read_does_not_expose_reserved_files_as_concepts():
  out = _tools()["okf_read"].func(path="/index.md")
  assert out["error"] == "not_found"


def test_resource_reads_a_file_declared_by_a_concept():
  """The acme sample's attester lives at bundle-root attesters/ (see PROVENANCE)."""
  out = _tools()["okf_resource"].func(
      concept_path="/computations/revenue-ytd.md",
      file="attesters/sql_equality.py",
  )
  assert "def " in out["text"]
  assert out["bytes"] > 0
  assert out["truncated"] is False
  assert out["content_is_untrusted_data"] is True


def test_resource_supports_concept_relative_paths():
  out = _tools()["okf_resource"].func(
      concept_path="/computations/revenue-ytd.md",
      file="../attesters/sql_equality.py",
  )
  assert "def " in out["text"]


def test_resource_reads_a_references_file(tmp_path):
  """The §6.3 convention: a references/ file relative to the concept."""
  root = tmp_path / "bundle"
  (root / "computations" / "references" / "attesters").mkdir(parents=True)
  (root / "computations" / "revenue.md").write_text(
      "---\ntype: Attested Computation\n---\nbody\n", encoding="utf-8"
  )
  (
      root / "computations" / "references" / "attesters" / "revenue.py"
  ).write_text("print('attester')\n", encoding="utf-8")
  out = _toolset(root)["okf_resource"].func(
      concept_path="/computations/revenue.md",
      file="references/attesters/revenue.py",
  )
  assert out["text"] == "print('attester')\n"
  assert out["file"] == "computations/references/attesters/revenue.py"


def test_resource_rejects_traversal():
  out = _tools()["okf_resource"].func(
      concept_path="/metrics/revenue.md",
      file="../../../../../etc/passwd",
  )
  assert out["error"] == "unsafe_path"  # never the file contents


def test_resource_rejects_absolute_urls():
  out = _tools()["okf_resource"].func(
      concept_path="/metrics/revenue.md",
      file="https://example.com/secret.txt",
  )
  assert out["error"] == "unsafe_path"


def test_resource_missing_file_is_a_structured_error():
  out = _tools()["okf_resource"].func(
      concept_path="/metrics/revenue.md", file="references/nope.py"
  )
  assert out["error"] == "not_found"


def test_resource_requires_an_existing_concept():
  out = _tools()["okf_resource"].func(concept_path="/nope.md", file="x.py")
  assert out["error"] == "not_found"


def test_resource_host_absolute_path_stays_inside_the_bundle():
  out = _tools()["okf_resource"].func(
      concept_path="/metrics/revenue.md", file="/etc/passwd"
  )
  assert out["error"] == "not_found"


def test_resource_caps_reads_at_64_kib(tmp_path):
  root = tmp_path / "bundle"
  root.mkdir()
  (root / "a.md").write_text("---\ntype: Metric\n---\nbody\n", encoding="utf-8")
  (root / "big.txt").write_text(
      "x" * (MAX_RESOURCE_BYTES + 4096), encoding="utf-8"
  )

  out = _toolset(root)["okf_resource"].func(
      concept_path="/a.md", file="big.txt"
  )
  assert out["truncated"] is True
  assert out["bytes"] == MAX_RESOURCE_BYTES
  assert len(out["text"]) == MAX_RESOURCE_BYTES


def test_resource_rejects_empty_file_argument():
  out = _tools()["okf_resource"].func(
      concept_path="/metrics/revenue.md", file="  "
  )
  assert out["error"] == "invalid_argument"


@pytest.mark.parametrize(
    "path",
    ["", "..", "/", "index.md", "a/../../b", "https://example.com/x.md"],
)
def test_read_never_raises_for_hostile_paths(path):
  out = _tools()["okf_read"].func(path=path)
  assert isinstance(out, dict)
  assert "error" in out or "body" in out
