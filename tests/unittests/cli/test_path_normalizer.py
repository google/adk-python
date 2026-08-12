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

"""Tests for normalizing model-generated file path strings."""

from __future__ import annotations

from pathlib import Path

from google.adk.cli.built_in_agents.utils.path_normalizer import sanitize_generated_file_path
import pytest


@pytest.mark.parametrize(
    'raw, expected',
    [
        # Nothing to strip.
        ('tools/web.yaml', 'tools/web.yaml'),
        # Whole path wrapped in quotes, which would otherwise create a
        # directory literally named "'tools".
        ("'tools/web.yaml'", 'tools/web.yaml'),
        ('"tools/web.yaml"', 'tools/web.yaml'),
        ('`tools/web.yaml`', 'tools/web.yaml'),
        # Each segment quoted independently.
        ('"tools"/"web.yaml"', 'tools/web.yaml'),
        # Surrounding whitespace, including a stray newline.
        ('  agent.yaml\n', 'agent.yaml'),
        ('tools/ web.yaml', 'tools/web.yaml'),
        # Backslash separators are preserved as separators.
        ("'dir'\\'file.txt'", 'dir\\file.txt'),
        # A leading separator survives (empty first segment).
        ('/abs/path.txt', '/abs/path.txt'),
    ],
)
def test_sanitize_generated_file_path_strips_boundary_noise(raw, expected):
  assert sanitize_generated_file_path(raw) == expected


def test_sanitize_generated_file_path_keeps_interior_quotes():
  """Only segment boundaries are stripped, so real filenames survive."""
  assert sanitize_generated_file_path("my'file.yaml") == "my'file.yaml"
  assert sanitize_generated_file_path("a/b'c/d.yaml") == "a/b'c/d.yaml"


def test_sanitize_generated_file_path_falls_back_when_all_chars_stripped():
  """Stripping everything would yield an empty path, so keep the input."""
  assert sanitize_generated_file_path("'''") == "'''"
  assert sanitize_generated_file_path('  ""  ') == '""'


def test_sanitize_generated_file_path_returns_empty_for_blank_input():
  assert sanitize_generated_file_path('') == ''
  assert sanitize_generated_file_path('   \t\n') == ''


def test_sanitize_generated_file_path_coerces_non_strings():
  assert sanitize_generated_file_path(Path('tools/web.yaml')) == (
      'tools/web.yaml'
  )
