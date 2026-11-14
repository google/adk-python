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

"""``okf_search``: deterministic scan, filters shared with ``okf_index``."""

from __future__ import annotations

import asyncio
from pathlib import Path

from google.adk.integrations.okf import OkfToolset
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "bundles"


def _tools() -> dict:
  ts = OkfToolset(bundle=str(FIXTURES / "acme_retail"), tool_name_prefix="okf")
  return {tool.name: tool for tool in asyncio.run(ts.get_tools_with_prefix())}


def test_search_matches_body_text_case_insensitively():
  tools = _tools()
  lower = tools["okf_search"].func(query="revenue")
  upper = tools["okf_search"].func(query="REVENUE")
  assert lower["count"] >= 1
  assert lower["total"] == upper["total"]
  assert all("revenue" in json_lower(row) for row in lower["results"])


def json_lower(row: dict) -> str:
  return (row["rel_path"] + " " + row["title"] + " " + row["snippet"]).lower()


def test_search_reports_where_the_match_was_found():
  tools = _tools()
  body = tools["okf_search"].func(query="approved narrations")
  assert body["count"] == 1
  assert body["results"][0]["matched_in"] == ["body"]
  assert "approved narrations" in body["results"][0]["snippet"]

  frontmatter = tools["okf_search"].func(query="jsmith@acme")
  assert frontmatter["count"] >= 1
  assert all(
      "frontmatter" in row["matched_in"] for row in frontmatter["results"]
  )


def test_search_filters_to_human_reviewed_only():
  out = _tools()["okf_search"].func(query="", trust_tier="human-reviewed")
  assert out["count"] == 7  # 8 verified minus the deprecated legacy concept
  assert all(row["trust_tier"] == "human-reviewed" for row in out["results"])
  with_deprecated = _tools()["okf_search"].func(
      query="", trust_tier="human-reviewed", include_deprecated=True
  )
  assert with_deprecated["count"] == 8


def test_search_excludes_deprecated_by_default():
  out = _tools()["okf_search"].func(query="", include_deprecated=False)
  assert all(row["status"] != "deprecated" for row in out["results"])
  with_deprecated = _tools()["okf_search"].func(
      query="", include_deprecated=True
  )
  assert any(
      row["status"] == "deprecated" for row in with_deprecated["results"]
  )


def test_search_supports_regex():
  out = _tools()["okf_search"].func(query=r"gross[- ]margin", regex=True)
  assert out["count"] == 4
  invalid = _tools()["okf_search"].func(query="(", regex=True)
  assert invalid["error"] == "invalid_regex"


def test_search_empty_query_returns_everything_selected():
  out = _tools()["okf_search"].func(query="")
  assert out["count"] == 8
  assert out["total"] == 8
  assert out["truncated"] is False


def test_search_limit_is_reported():
  out = _tools()["okf_search"].func(query="", limit=2)
  assert out["count"] == 2
  assert out["total"] == 8
  assert out["truncated"] is True
  assert (
      _tools()["okf_search"].func(query="", limit=0)["error"]
      == "invalid_argument"
  )


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"directory": "metrics"},
        {"type": "Metric"},
        {"tags": ["finance"]},
        {"trust_tier": "human-reviewed"},
        {"include_deprecated": True},
        {"include_deprecated": True, "directory": "metrics", "type": "Metric"},
    ],
)
def test_search_and_index_share_one_filter_predicate(filters):
  """The drift guard: both tools must select exactly the same concept set."""
  tools = _tools()
  indexed = {
      row["rel_path"] for row in tools["okf_index"].func(**filters)["concepts"]
  }
  searched = {
      row["rel_path"]
      for row in tools["okf_search"].func(query="", **filters)["results"]
  }
  assert indexed == searched
