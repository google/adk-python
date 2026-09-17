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

from __future__ import annotations

from google.adk.evaluation._path_validation import validate_path_segment
import pytest


@pytest.mark.parametrize(
    "value", ["eval_set_1", "my-app", "App Name 1", "résumé", "a.b.c"]
)
def test_validate_path_segment_accepts_valid_value(value):
  validate_path_segment(value, "field")


def test_validate_path_segment_rejects_empty():
  with pytest.raises(ValueError, match="must not be empty"):
    validate_path_segment("", "field")


def test_validate_path_segment_rejects_null_byte():
  with pytest.raises(ValueError, match="must not contain null bytes"):
    validate_path_segment("foo\x00bar", "field")


@pytest.mark.parametrize("value", ["foo/bar", "foo\\bar", "/", "\\"])
def test_validate_path_segment_rejects_path_separators(value):
  with pytest.raises(ValueError, match="must not contain path separators"):
    validate_path_segment(value, "field")


@pytest.mark.parametrize("value", [".", ".."])
def test_validate_path_segment_rejects_traversal_segments(value):
  with pytest.raises(ValueError, match="must not contain traversal segments"):
    validate_path_segment(value, "field")


def test_validate_path_segment_includes_field_name_in_error():
  with pytest.raises(ValueError, match="eval_set_id"):
    validate_path_segment("", "eval_set_id")
