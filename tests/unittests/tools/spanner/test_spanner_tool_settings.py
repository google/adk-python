# Copyright 2025 Google LLC
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

from google.adk.tools.spanner.settings import SpannerToolSettings
import pytest


def test_spanner_tool_settings_experimental_warning():
  """Test SpannerToolSettings experimental warning."""
  with pytest.warns(
      UserWarning,
      match="Tool settings defaults may have breaking change in the future.",
  ):
    SpannerToolSettings()
