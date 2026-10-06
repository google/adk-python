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

"""Registry entry for a tool call running in the background of a Live session."""

from __future__ import annotations

import asyncio
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from .live_request_queue import LiveRequestQueue


class ActiveBackgroundTool(BaseModel):
  """One tool call running in the background of a live invocation.

  Streaming tools (async generators) and non-blocking tools are both tracked
  with this model, one entry per function call.
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  tool_name: str
  """The name of the tool that was called."""

  function_call_id: str
  """The id of the function call this entry runs."""

  task: asyncio.Task[Any]
  """The task running the tool call."""

  input_stream: Optional[LiveRequestQueue] = None
  """The input stream the tool reads live requests from.

  Set only for tools that declare an ``input_stream: LiveRequestQueue``
  parameter.
  """

  is_generator: bool = False
  """Whether the tool is an async generator that yields several responses."""
