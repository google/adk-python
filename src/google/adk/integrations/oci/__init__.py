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

"""OCI Generative AI integration.

Model providers for Google Gemini and other models hosted on Oracle Cloud
Infrastructure (OCI) Generative AI. Install with: pip install google-adk[oci]
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
  from ._oci_genai_llm import OCIGenAILlm

_lazy_imports = {
    "OCIGenAILlm": "._oci_genai_llm",
}


def __getattr__(name: str) -> typing.Any:
  if name in _lazy_imports:
    import importlib

    module = importlib.import_module(_lazy_imports[name], __name__)
    return getattr(module, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return list(_lazy_imports.keys())
