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

from .tool_context import ToolContext


# TODO: make this internal, since user doesn't need to use this tool directly.
def transfer_to_agent(agent_name: str, tool_context: ToolContext):
  """Transfer the current request to another agent.

  This tool hands off control to a different named agent.  Any additional
  context (e.g. user query, temperature settings) should be carried in the
  LLM prompt or tool_context, not passed as extra parameters here.

  Args:
    agent_name: The identifier of the agent to transfer to (e.g. "math_agent").
    tool_context: The current ToolContext, whose `actions.transfer_to_agent`
      field will be set.

  Returns:
    None. Side-effect: `tool_context.actions.transfer_to_agent` is set.
  """
  tool_context.actions.transfer_to_agent = agent_name
