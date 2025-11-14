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


from google.adk.models.anthropic_llm import Claude
from google.adk.models.google_llm import Gemini
from google.adk.utils.output_schema_utils import can_use_output_schema_with_tools
import pytest


@pytest.mark.parametrize(
    "model, env_value, expected",
    [
        ("gemini-2.5-pro", "1", True),
        ("gemini-2.5-pro", "0", False),
        ("gemini-2.5-pro", None, False),
        (Gemini(model="gemini-2.5-pro"), "1", True),
        (Gemini(model="gemini-2.5-pro"), "0", False),
        (Gemini(model="gemini-2.5-pro"), None, False),
        ("gemini-2.0-flash", "1", True),
        ("gemini-2.0-flash", "0", False),
        ("gemini-2.0-flash", None, False),
        ("gemini-1.5-pro", "1", False),
        ("gemini-1.5-pro", "0", False),
        ("gemini-1.5-pro", None, False),
        (Claude(model="claude-3.7-sonnet"), "1", False),
        (Claude(model="claude-3.7-sonnet"), "0", False),
        (Claude(model="claude-3.7-sonnet"), None, False),
    ],
)
def test_can_use_output_schema_with_tools(
    monkeypatch, model, env_value, expected
):
  """Test can_use_output_schema_with_tools."""
  if env_value is not None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", env_value)
  else:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
  assert can_use_output_schema_with_tools(model) == expected
