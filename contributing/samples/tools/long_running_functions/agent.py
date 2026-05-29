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

from google.adk import Agent
from google.adk.tools.long_running_tool import LongRunningFunctionTool


def export_data(export_type: str) -> dict[str, str]:
  """Exports user data.

  Args:
      export_type: The type of data to export (e.g., 'csv', 'json').

  Returns:
      A dict with the status.
  """
  # In a real application, this would kick off a background job.
  # Here we just return a status.
  return {
      "status": "in-progress",
      "progress": "0%",
      "message": f"Exporting {export_type} data. This may take some time.",
  }


root_agent = Agent(
    name="long_running_functions",
    instruction="""
    You are an assistant that can export user data.
    When the user asks to export data, call the `export_data` tool.
    """,
    tools=[LongRunningFunctionTool(func=export_data)],
)
