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

"""L1 ``okf_index``: frontmatter summaries only, plus the full tool surface."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path

from google.adk.integrations.okf import OkfToolset

FIXTURES = Path(__file__).parent / "fixtures" / "bundles"


def _tools(
    bundle_dir: str = "acme_retail", *, now: dt.datetime | None = None
) -> dict:
  ts = OkfToolset(
      bundle=str(FIXTURES / bundle_dir), tool_name_prefix="okf", now=now
  )
  return {tool.name: tool for tool in asyncio.run(ts.get_tools_with_prefix())}


def test_toolset_yields_expected_tool_names():
  """The prefix applies to every tool the toolset returns."""
  assert set(_tools()) == {
      "okf_index",
      "okf_read",
      "okf_resource",
      "okf_search",
  }


def test_index_returns_frontmatter_only_no_bodies():
  out = _tools()["okf_index"].func(directory="")
  assert out["count"] == 8  # acme_retail has 9 concepts, one deprecated
  first = out["concepts"][0]
  assert "body" not in first  # L1 stays cheap
  assert {"rel_path", "type", "title", "trust_tier", "status", "stale"} <= set(
      first
  )
  assert out["violations"] == []


def test_index_never_leaks_body_text():
  """A phrase that exists only in metrics/revenue.md's body must not appear."""
  out = _tools()["okf_index"].func(directory="")
  assert "30-day return window" not in json.dumps(out)


def test_index_can_filter_by_type():
  out = _tools()["okf_index"].func(directory="", type="Attested Computation")
  assert out["count"] == 2
  assert {row["type"] for row in out["concepts"]} == {"Attested Computation"}


def test_index_can_filter_by_directory():
  out = _tools()["okf_index"].func(directory="metrics")
  assert {row["rel_path"] for row in out["concepts"]} == {
      "metrics/gross-margin.md",
      "metrics/revenue.md",
  }
  assert _tools()["okf_index"].func(directory="/metrics/")["count"] == 2


def test_index_can_filter_by_tags_any_of():
  out = _tools()["okf_index"].func(directory="", tags=["headline-metric"])
  assert {row["rel_path"] for row in out["concepts"]} == {
      "metrics/gross-margin.md",
      "metrics/revenue.md",
  }
  assert _tools()["okf_index"].func(directory="", tags=["nope"])["count"] == 0


def test_index_hides_deprecated_by_default_and_can_include_them():
  tools = _tools()
  assert all(
      row["status"] != "deprecated"
      for row in tools["okf_index"].func()["concepts"]
  )
  with_deprecated = tools["okf_index"].func(include_deprecated=True)
  assert with_deprecated["count"] == 9
  assert any(
      row["status"] == "deprecated" for row in with_deprecated["concepts"]
  )


def test_index_filters_by_derived_trust_tier():
  tools = _tools()
  human = tools["okf_index"].func(trust_tier="human-reviewed")
  assert (
      human["count"] == 7
  )  # 8 verified concepts minus the deprecated legacy one
  assert all(row["trust_tier"] == "human-reviewed" for row in human["concepts"])
  assert tools["okf_index"].func(trust_tier="HUMAN-REVIEWED")["count"] == 7
  assert tools["okf_index"].func(trust_tier="machine-confirmed")["count"] == 0


def test_index_rejects_unknown_trust_tier_as_structured_error():
  out = _tools()["okf_index"].func(trust_tier="vibes")
  assert out["error"] == "invalid_argument"
  assert "trust_tier" in out["message"]


def test_index_can_exclude_stale_concepts():
  future = dt.datetime(2027, 6, 1, tzinfo=dt.timezone.utc)
  tools = _tools(now=future)
  fresh = tools["okf_index"].func(include_stale=False)
  assert [row["rel_path"] for row in fresh["concepts"]] == [
      "skills/run-on-bq.md"
  ]
  assert all(not row["stale"] for row in fresh["concepts"])
  assert tools["okf_index"].func(include_stale=True)["count"] == 8
