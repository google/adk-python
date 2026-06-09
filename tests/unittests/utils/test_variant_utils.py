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

"""Tests for variant_utils."""

import warnings

from google.adk.utils import variant_utils
from google.adk.utils.variant_utils import GoogleLLMVariant


def test_get_google_llm_variant_enterprise(monkeypatch):
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', 'true')
  assert variant_utils.get_google_llm_variant() == GoogleLLMVariant.VERTEX_AI


def test_get_google_llm_variant_vertexai_fallback(monkeypatch):
  monkeypatch.delenv('GOOGLE_GENAI_USE_ENTERPRISE', raising=False)
  monkeypatch.setenv('GOOGLE_GENAI_USE_VERTEXAI', 'true')
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    result = variant_utils.get_google_llm_variant()
    assert result == GoogleLLMVariant.VERTEX_AI
    assert len(w) == 1
    assert issubclass(w[-1].category, DeprecationWarning)
    assert 'GOOGLE_GENAI_USE_VERTEXAI is deprecated' in str(w[-1].message)


def test_get_google_llm_variant_default(monkeypatch):
  monkeypatch.delenv('GOOGLE_GENAI_USE_ENTERPRISE', raising=False)
  monkeypatch.delenv('GOOGLE_GENAI_USE_VERTEXAI', raising=False)
  assert variant_utils.get_google_llm_variant() == GoogleLLMVariant.GEMINI_API
