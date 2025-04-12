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

import json
from typing import AsyncGenerator

from pydantic import Field
import requests
from typing_extensions import override

from ..events.event import Event
from .base_agent import BaseAgent
from .invocation_context import InvocationContext


class RemoteAgent(BaseAgent):
  """Experimental, do not use."""

  url: str

  sub_agents: list[BaseAgent] = Field(
      default_factory=list, init=False, frozen=True
  )
  """Sub-agent is disabled in RemoteAgent."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    data = {
        'invocation_id': ctx.invocation_id,
        'session': ctx.session.model_dump(exclude_none=True),
    }
    events = requests.post(self.url, data=json.dumps(data), timeout=120)
    events.raise_for_status()
    for event in events.json():
      e = Event.model_validate(event)
      e.author = self.name
      yield e
