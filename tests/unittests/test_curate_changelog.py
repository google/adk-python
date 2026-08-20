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
