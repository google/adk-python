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
from typing import TYPE_CHECKING

from .base_artifact_service import BaseArtifactService

if TYPE_CHECKING:
  from .file_artifact_service import FileArtifactService
  from .gcs_artifact_service import GcsArtifactService
  from .in_memory_artifact_service import InMemoryArtifactService

__all__ = [
    'BaseArtifactService',
    'FileArtifactService',
    'GcsArtifactService',
    'InMemoryArtifactService',
]

_LAZY_MEMBERS: dict[str, str] = {
    'FileArtifactService': 'file_artifact_service',
    'GcsArtifactService': 'gcs_artifact_service',
    'InMemoryArtifactService': 'in_memory_artifact_service',
}


def __getattr__(name: str):
  if name in _LAZY_MEMBERS:
    module = importlib.import_module(f'{__name__}.{_LAZY_MEMBERS[name]}')
    return vars(module)[name]
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
