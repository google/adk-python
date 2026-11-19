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

"""Discovery Engine integration.

This module provides a client for verifying model answers against facts with
the Discovery Engine CheckGrounding API.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
  from ._check_grounding_client import CheckGroundingClient
  from ._types import FactChunk
  from ._types import GroundingLimits
  from ._types import GroundingResult
  from ._types import GroundingStatus

# Map attribute names to relative module paths.
_lazy_imports = {
    'CheckGroundingClient': '._check_grounding_client',
    'FactChunk': '._types',
    'GroundingLimits': '._types',
    'GroundingResult': '._types',
    'GroundingStatus': '._types',
}

__all__ = [
    'CheckGroundingClient',
    'FactChunk',
    'GroundingLimits',
    'GroundingResult',
    'GroundingStatus',
]


def __getattr__(name: str) -> typing.Any:
  if name in _lazy_imports:
    import importlib

    module_path = _lazy_imports[name]
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
  return list(_lazy_imports.keys())
