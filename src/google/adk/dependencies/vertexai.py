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

from typing import Any
from typing import TYPE_CHECKING

import vertexai as vertexai

if TYPE_CHECKING:
  from vertexai.preview import rag as rag


def __getattr__(name: str) -> Any:
  if name == 'rag':
    # Load on demand: vertexai.preview.rag is deprecated and warns on import,
    # so evaluation and other Vertex AI helpers must not pull it in.
    from vertexai.preview import rag

    return rag
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
