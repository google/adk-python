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

"""Tests for _telemetry_context.py."""

from __future__ import annotations

import google.adk
from google.adk.utils._telemetry_context import _get_telemetry_surface
from google.adk.utils._telemetry_context import _is_visual_builder
from google.adk.utils._telemetry_context import _surface_user_agent
from google.adk.utils._telemetry_context import _telemetry_surface


def test_surface_user_agent_default_unset():
  """Returns None when neither telemetry context variable is set."""
  assert _get_telemetry_surface() is None
  assert _surface_user_agent() is None


def test_surface_user_agent_telemetry_surface():
  """Formats the active _telemetry_surface with the ADK version."""
  token = _telemetry_surface.set("my-surface")
  try:
    assert _get_telemetry_surface() == "my-surface"
    assert (
        _surface_user_agent()
        == f"google-adk-my-surface/{google.adk.__version__}"
    )
  finally:
    _telemetry_surface.reset(token)


def test_surface_user_agent_visual_builder():
  """Maps _is_visual_builder=True to the versioned visual-builder token."""
  token = _is_visual_builder.set(True)
  try:
    assert _get_telemetry_surface() == "visual-builder"
    assert (
        _surface_user_agent()
        == f"google-adk-visual-builder/{google.adk.__version__}"
    )
  finally:
    _is_visual_builder.reset(token)


def test_surface_user_agent_telemetry_surface_takes_precedence():
  """Prefers _telemetry_surface when both context variables are set."""
  vb_token = _is_visual_builder.set(True)
  surface_token = _telemetry_surface.set("my-surface")
  try:
    assert _get_telemetry_surface() == "my-surface"
    assert (
        _surface_user_agent()
        == f"google-adk-my-surface/{google.adk.__version__}"
    )
  finally:
    _telemetry_surface.reset(surface_token)
    _is_visual_builder.reset(vb_token)
