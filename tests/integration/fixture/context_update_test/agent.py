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

from typing import List
from typing import Union

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel


def update_fc(
    data_one: str,
    data_two: Union[int, float, str],
    data_three: list[str],
    data_four: List[Union[int, float, str]],
    tool_context: ToolContext,
):
  """Simply ask to update these variables in the context"""
  tool_context.actions.update_state("data_one", data_one)
  tool_context.actions.update_state("data_two", data_two)
  tool_context.actions.update_state("data_three", data_three)
  tool_context.actions.update_state("data_four", data_four)


root_agent = Agent(
    model="gemini-1.5-flash",
    name="root_agent",
    instruction="Call tools",
    flow="auto",
    tools=[update_fc],
)
