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

"""Firestore integrations for ADK.

This module provides session and memory services backed by Google Cloud
Firestore. They require the optional ``google-cloud-firestore`` package.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
  from .firestore_memory_service import FirestoreMemoryService
  from .firestore_session_service import FirestoreSessionService

# Map attribute names to relative module paths.
_lazy_imports = {
    "FirestoreMemoryService": ".firestore_memory_service",
    "FirestoreSessionService": ".firestore_session_service",
}

__all__ = [
    "FirestoreMemoryService",
    "FirestoreSessionService",
]


def __getattr__(name: str) -> typing.Any:
  if name in _lazy_imports:
    import importlib

    module_path = _lazy_imports[name]
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return list(_lazy_imports.keys())
