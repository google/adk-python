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

from google.adk.cli.api_server import _sanitize_html_artifact
from google.genai import types
import pytest


@pytest.mark.parametrize(
    "payload",
    (
        "<script>alert(1)</script>",
        '<img src=x onerror="alert(1)">',
        '<body onload="alert(1)">',
        '<iframe src="javascript:alert(1)"></iframe>',
        '<a href="javascript:alert(1)">click</a>',
        '<svg onload="alert(1)"></svg>',
    ),
)
def test_sanitize_html_artifact_blocks_xss(payload: str) -> None:
  artifact = types.Part.from_bytes(data=payload.encode(), mime_type="text/html")

  sanitized = _sanitize_html_artifact(artifact).inline_data.data.decode()

  assert "alert" not in sanitized
  assert "script" not in sanitized
  assert "onerror" not in sanitized
  assert "onload" not in sanitized
  assert "iframe" not in sanitized
  assert "javascript:" not in sanitized


def test_sanitize_html_artifact_preserves_safe_html() -> None:
  artifact = types.Part.from_bytes(
      data=b"<h1>Title</h1><p>Hello <strong>world</strong></p>",
      mime_type="text/html",
  )

  sanitized = _sanitize_html_artifact(artifact).inline_data.data.decode()

  assert "<h1>Title</h1>" in sanitized
  assert "<strong>world</strong>" in sanitized


def test_sanitize_html_artifact_ignores_other_mime_types() -> None:
  artifact = types.Part.from_bytes(data=b"unchanged", mime_type="text/plain")

  assert _sanitize_html_artifact(artifact) is artifact
  assert artifact.inline_data.data == b"unchanged"
