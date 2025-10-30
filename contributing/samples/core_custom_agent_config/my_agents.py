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

from keyword import kwlist
from typing import Any
from typing import AsyncGenerator
from typing import ClassVar
from typing import Dict
from typing import Type

from google.adk.agents import BaseAgent
from google.adk.agents.base_agent_config import BaseAgentConfig
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types
from pydantic import ConfigDict
from typing_extensions import override


class MyCustomAgentConfig(BaseAgentConfig):
  model_config = ConfigDict(
      extra="forbid",
  )
  agent_class: str = "core_cutom_agent_config.my_agents.MyCustomAgent"
  my_field: str = ""


class MyCustomAgent(BaseAgent):
  my_field: str = ""

  config_type: ClassVar[type[BaseAgentConfig]] = MyCustomAgentConfig

  @override
  @classmethod
  def _parse_config(
      cls: Type[MyCustomAgent],
      config: MyCustomAgentConfig,
      config_abs_path: str,
      kwargs: Dict[str, Any],
  ) -> Dict[str, Any]:
    if config.my_field:
      kwargs["my_field"] = config.my_field
    return kwargs

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=types.ModelContent(
            parts=[
                types.Part(
                    text=f"I feel good! value in my_field: `{self.my_field}`"
                )
            ]
        ),
    )
