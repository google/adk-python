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

from __future__ import annotations

import asyncio
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from .live_request_queue import LiveRequestQueue


class ActiveStreamingTool(BaseModel):
  """Manages streaming tool related resources during invocation."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  task: Optional[asyncio.Task] = None
  """The active task of this streaming tool."""

  stream: Optional[LiveRequestQueue] = None
  """The active (input) streams of this streaming tool."""
