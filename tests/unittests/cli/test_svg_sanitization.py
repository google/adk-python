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

from google.adk.cli.api_server import _sanitize_svg_artifact
from google.genai import types
import pytest


@pytest.mark.parametrize(
    "payload",
    (
        '<svg><foreignObject><img onerror="alert(1)"/></foreignObject></svg>',
        "<svg><script>alert(1)</script></svg>",
        '<svg onanimationend="alert(1)"><rect/></svg>',
        '<svg><a href="javascript:alert(1)">click</a></svg>',
    ),
)
def test_sanitize_svg_artifact_blocks_xss(payload: str) -> None:
  artifact = types.Part.from_bytes(
      data=payload.encode(), mime_type="image/svg+xml"
  )

  sanitized = _sanitize_svg_artifact(artifact).inline_data.data.decode()

  assert "alert" not in sanitized
  assert "foreignObject" not in sanitized
  assert "script" not in sanitized
  assert "onanimationend" not in sanitized
  assert "javascript:" not in sanitized


def test_sanitize_svg_artifact_preserves_safe_svg() -> None:
  artifact = types.Part.from_bytes(
      data=(
          b'<svg viewBox="0 0 10 10"><defs><linearGradient id="g">'
          b'</linearGradient></defs><rect width="10" height="10" fill="red"/>'
          b"</svg>"
      ),
      mime_type="image/svg+xml",
  )

  sanitized = _sanitize_svg_artifact(artifact).inline_data.data.decode()

  assert "linearGradient" in sanitized
  assert 'viewBox="0 0 10 10"' in sanitized
  assert '<rect width="10" height="10" fill="red"></rect>' in sanitized


def test_sanitize_svg_artifact_ignores_other_mime_types() -> None:
  artifact = types.Part.from_bytes(data=b"unchanged", mime_type="image/png")

  assert _sanitize_svg_artifact(artifact) is artifact
  assert artifact.inline_data.data == b"unchanged"
