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

from .tool_context import ToolContext


def transfer_to_agent(agent_name: str, tool_context: ToolContext) -> None:
  """Transfer the question to another agent.

  This tool hands off control to another agent when it's more suitable to
  answer the user's question according to the agent's description.

  Args:
    agent_name: the agent name to transfer to.
  """
  tool_context.actions.transfer_to_agent = agent_name
