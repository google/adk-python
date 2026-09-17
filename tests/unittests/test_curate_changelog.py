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

"""Tests for the release changelog curation."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_SCRIPT = (
    pathlib.Path(__file__).parent.parent.parent
    / "scripts"
    / "curate_changelog.py"
)
_SPEC = importlib.util.spec_from_file_location("curate_changelog", _SCRIPT)
curate_changelog = importlib.util.module_from_spec(_SPEC)
sys.modules["curate_changelog"] = curate_changelog
_SPEC.loader.exec_module(curate_changelog)


def test_unwrap_joins_a_wrapped_paragraph():
  text = "A release about\ncorrectness and hardening."

  assert (
      curate_changelog._unwrap_lines(text)
      == "A release about correctness and hardening."
  )


def test_unwrap_joins_a_wrapped_list_item():
  text = "* **Tools**: a tool response now carries\nimages back to the model."

  assert curate_changelog._unwrap_lines(text) == (
      "* **Tools**: a tool response now carries images back to the model."
  )


def test_unwrap_keeps_separate_list_items_apart():
  text = "* first item\n* second item\n- third item"

  assert curate_changelog._unwrap_lines(text) == text


def test_unwrap_keeps_numbered_list_items_apart():
  text = "1. first step\n2. second step"

  assert curate_changelog._unwrap_lines(text) == text


def test_unwrap_keeps_blank_lines_and_headers_on_their_own_lines():
  text = "the theme.\n\n#### Breaking changes\n\n* **X**: migrate by doing Y."

  assert curate_changelog._unwrap_lines(text) == text


def test_unwrap_leaves_a_fenced_block_alone():
  text = "install it:\n\n```bash\nuv pip install google-adk\nadk web\n```"

  assert curate_changelog._unwrap_lines(text) == text


def test_unwrap_does_not_join_a_paragraph_onto_a_closing_fence():
  text = "```bash\nadk web\n```\nthen open the\nbrowser."

  assert curate_changelog._unwrap_lines(text) == (
      "```bash\nadk web\n```\nthen open the browser."
  )


def test_unwrap_does_not_join_a_paragraph_onto_a_table_row():
  text = "| a | b |\nnot a table cell."

  assert curate_changelog._unwrap_lines(text) == text


def test_build_block_unwraps_drafted_prose():
  drafted = "A release about\ncorrectness.\n\n* **Tools**: return\nmedia."

  block = curate_changelog._build_block(drafted)

  assert block == (
      "### Highlights\n\nA release about correctness.\n\n"
      "* **Tools**: return media.\n"
  )


def test_dedupe_key_strips_commit_hashes():
  line1 = (
      "* **eventarc:** add Eventarc Advanced toolset for ADK"
      " ([217a90a](https://github.com/google/adk-python/commit/217a90a2e6c9725aeaac3dffedca8d63c25037fd))"
  )
  line2 = (
      "* **eventarc:** add Eventarc Advanced toolset for ADK"
      " ([d4f157d](https://github.com/google/adk-python/commit/d4f157d2ed6fad21a6aa4c6e29e6133e3fe5db76))"
  )

  assert curate_changelog._dedupe_key(line1) == curate_changelog._dedupe_key(
      line2
  )
  assert (
      curate_changelog._dedupe_key(line1)
      == "* **eventarc:** add eventarc advanced toolset for adk"
  )


def test_dedupe_key_strips_bare_markdown_links():
  line1 = (
      "* Remove duplicate options from `adk deploy`"
      " [3fa2ea7](https://github.com/google/adk-python/commit/3fa2ea7cb923c9f8606d98b45a23bd58a7027436)"
  )
  line2 = (
      "* Remove duplicate options from `adk deploy`"
      " ([3fa2ea7](https://github.com/google/adk-python/commit/3fa2ea7cb923c9f8606d98b45a23bd58a7027436))"
  )

  assert curate_changelog._dedupe_key(line1) == curate_changelog._dedupe_key(
      line2
  )


def test_dedupe_key_strips_pr_links_and_closes():
  line1 = (
      "* add state_delta support to LiveRequest for live mode"
      " ([8219774](https://github.com/google/adk-python/commit/82197740a603e146ee35e0f18d2761a5c8f155d6)),"
      " closes [#4220](https://github.com/google/adk-python/issues/4220)"
  )
  line2 = (
      "* add state_delta support to LiveRequest for live mode"
      " ([abcdef1](https://github.com/google/adk-python/commit/abcdef10a603e146ee35e0f18d2761a5c8f155d6))"
  )
  line3 = "* add state_delta support to LiveRequest for live mode (#4220)"

  assert curate_changelog._dedupe_key(line1) == curate_changelog._dedupe_key(
      line2
  )
  assert curate_changelog._dedupe_key(line1) == curate_changelog._dedupe_key(
      line3
  )


def test_dedupe_key_strips_trailing_punctuation():
  line1 = (
      "* fix: Resolve scheduler leakage and make scheduler instantiation"
      " explicit. ([ce2e4ca](https://...))"
  )
  line2 = (
      "* fix: Resolve scheduler leakage and make scheduler instantiation"
      " explicit ([d4f0772](https://...))"
  )

  assert curate_changelog._dedupe_key(line1) == curate_changelog._dedupe_key(
      line2
  )


def test_normalize_body_deduplicates_duplicate_commits():
  lines = [
      "### Features\n",
      "\n",
      (
          "* **eventarc:** add Eventarc Advanced toolset for ADK"
          " ([217a90a](https://...))\n"
      ),
      (
          "* **eventarc:** add Eventarc Advanced toolset for ADK"
          " ([d4f157d](https://...))\n"
      ),
      "* unique feature ([1234567](https://...))\n",
      "\n",
      "### Bug Fixes\n",
      "\n",
      (
          "* Resolve scheduler leakage and make scheduler instantiation"
          " explicit ([ce2e4ca](https://...))\n"
      ),
      (
          "* Resolve scheduler leakage and make scheduler instantiation"
          " explicit. ([d4f0772](https://...))\n"
      ),
  ]

  deduped = curate_changelog._normalize_body(lines)
  assert deduped == [
      "### Features\n",
      "\n",
      (
          "* **eventarc:** add Eventarc Advanced toolset for ADK"
          " ([217a90a](https://...))\n"
      ),
      "* unique feature ([1234567](https://...))\n",
      "\n",
      "### Bug Fixes\n",
      "\n",
      (
          "* resolve scheduler leakage and make scheduler instantiation"
          " explicit ([ce2e4ca](https://...))\n"
      ),
  ]


def test_curate_deduplicates_and_adds_highlights():
  text = """# Changelog

## [2.7.0](https://github.com/google/adk-python/compare/v2.6.3...v2.7.0) (2026-08-27)

### Features

* **eventarc:** add Eventarc Advanced toolset for ADK ([217a90a](https://...))
* **eventarc:** add Eventarc Advanced toolset for ADK ([d4f157d](https://...))
* unique feature ([1234567](https://...))
"""

  updated = curate_changelog.curate(text, model="test-model", fold_threshold=12)
  assert (
      "* **eventarc:** add Eventarc Advanced toolset for ADK"
      " ([d4f157d](https://...))"
      not in updated
  )
  assert (
      "* **eventarc:** add Eventarc Advanced toolset for ADK"
      " ([217a90a](https://...))"
      in updated
  )
  assert "* unique feature ([1234567](https://...))" in updated
  assert "### Highlights" in updated


def test_curate_leaves_already_curated_section_unchanged():
  text = """# Changelog

## [2.7.0](https://github.com/google/adk-python/compare/v2.6.3...v2.7.0) (2026-08-27)

### Highlights

Theme of the release.

* **eventarc**: add Eventarc toolset. (217a90a)

### Features

* **eventarc:** add Eventarc Advanced toolset for ADK ([217a90a](https://...))
"""

  updated = curate_changelog.curate(text, model="test-model", fold_threshold=12)
  assert updated == text
