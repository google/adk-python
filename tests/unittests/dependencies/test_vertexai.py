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

import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest


def test_vertexai_dependency_resolves_rag() -> None:
  """google.adk.dependencies.vertexai resolves rag and rejects unknown names."""
  module = importlib.import_module('google.adk.dependencies.vertexai')
  assert module.rag is importlib.import_module('vertexai.preview.rag')
  with pytest.raises(AttributeError):
    _ = module.nonexistent_attr


def test_internal_vertexai_dependency_defers_preview_rag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The internal Vertex AI dependency shim loads preview.rag on demand."""
  # Runs only in the internal monorepo checkout; dependencies_internal/ is in
  # PIPER_EXCLUDES and is not exported to GitHub, where this test skips.
  module_path = (
      Path(__file__).resolve().parents[3] / 'dependencies_internal/vertexai.py'
  )
  if not module_path.is_file():
    pytest.skip('Internal Vertex AI dependency shim is not present.')

  aiplatform_mod = types.ModuleType('google.cloud.aiplatform')
  vertexai_mod = types.ModuleType('google.cloud.aiplatform.vertexai')
  preview_mod = types.ModuleType('google.cloud.aiplatform.vertexai.preview')
  example_stores_mod = types.ModuleType(
      'google.cloud.aiplatform.vertexai.preview.example_stores'
  )
  setattr(aiplatform_mod, 'vertexai', vertexai_mod)
  setattr(vertexai_mod, 'preview', preview_mod)
  setattr(preview_mod, 'example_stores', example_stores_mod)

  monkeypatch.setitem(sys.modules, 'google.cloud.aiplatform', aiplatform_mod)
  monkeypatch.setitem(
      sys.modules, 'google.cloud.aiplatform.vertexai', vertexai_mod
  )
  monkeypatch.setitem(
      sys.modules, 'google.cloud.aiplatform.vertexai.preview', preview_mod
  )
  monkeypatch.setitem(
      sys.modules,
      'google.cloud.aiplatform.vertexai.preview.example_stores',
      example_stores_mod,
  )
  monkeypatch.delitem(
      sys.modules,
      'google.cloud.aiplatform.vertexai.preview.rag',
      raising=False,
  )

  spec = importlib.util.spec_from_file_location(
      '_test_internal_vertexai_shim', module_path
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  assert 'rag' not in module.__dict__
  assert 'google.cloud.aiplatform.vertexai.preview.rag' not in sys.modules

  rag_mod = types.ModuleType('google.cloud.aiplatform.vertexai.preview.rag')
  setattr(preview_mod, 'rag', rag_mod)
  monkeypatch.setitem(
      sys.modules, 'google.cloud.aiplatform.vertexai.preview.rag', rag_mod
  )
  assert module.rag is rag_mod
  with pytest.raises(AttributeError):
    _ = module.nonexistent_attr
