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

"""Fsspec integrations for ADK.

Artifact storage backed by ``fsspec``-compatible filesystems. Classes that
require ``fsspec`` are imported lazily so this package can be imported without
installing ``google-adk[fsspec]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...utils import _lazy

if TYPE_CHECKING:
  from ._fsspec_artifact_service import FsspecArtifactService

_LAZY_MEMBERS: dict[str, str] = {
    "FsspecArtifactService": "._fsspec_artifact_service",
}
__all__ = [
    "FsspecArtifactService",
]

__getattr__, __dir__ = _lazy.accessors(globals(), _LAZY_MEMBERS)
