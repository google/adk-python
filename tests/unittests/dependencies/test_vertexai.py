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
import subprocess
import sys
import warnings


def test_importing_eval_facade_does_not_warn_about_preview_rag():
  """Importing the Vertex AI eval facade does not emit the preview.rag deprecation."""
  script = """
import warnings

warnings.simplefilter('always', UserWarning)
with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always', UserWarning)
  import google.adk.evaluation.vertex_ai_eval_facade

rag_warnings = [w for w in caught if 'preview.rag' in str(w.message)]
raise SystemExit(1 if rag_warnings else 0)
"""
  result = subprocess.run(
      [sys.executable, '-c', script],
      capture_output=True,
      text=True,
      check=False,
  )
  assert result.returncode == 0, result.stderr or result.stdout


def test_vertexai_dependency_does_not_load_preview_rag():
  """Importing google.adk.dependencies.vertexai does not load vertexai.preview.rag."""
  sys.modules.pop('google.adk.dependencies.vertexai', None)
  sys.modules.pop('vertexai.preview.rag', None)

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', UserWarning)
    module = importlib.import_module('google.adk.dependencies.vertexai')
    importlib.reload(module)

  rag_warnings = [
      warning for warning in caught if 'preview.rag' in str(warning.message)
  ]
  assert rag_warnings == []
  assert 'vertexai.preview.rag' not in sys.modules
  assert 'rag' not in module.__dict__
