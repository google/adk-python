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

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.plugins import LoggingPlugin
from google.adk.plugins import ReflectAndRetryToolPlugin

APP_NAME = "basic"
USER_ID = "test_user"


def guess_number_tool(query: int) -> dict[str, Any]:
  """A tool that guesses a number.

  Args:
    query: The number to guess.

  Returns:
    A dictionary containing the status and result of the tool execution.
  """
  target_number = 3
  if query == target_number:
    return {"status": "success", "result": "Number is valid."}

  if abs(query - target_number) <= 2:
    return {"status": "error", "error_message": "Number is almost valid."}

  if query > target_number:
    raise ValueError("Number is too large.")

  if query < target_number:
    raise ValueError("Number is too small.")

  raise ValueError("Number is invalid.")


class CustomRetryPlugin(ReflectAndRetryToolPlugin):

  async def extract_error_from_result(
      self, *, tool, tool_args, tool_context, result
  ):
    return result if result.get("status") == "error" else None


# Sample query: "guess a number between 1 and 50"
root_agent = LlmAgent(
    name="hello_world",
    description="Helpful agent",
    instruction="""Your goal is to guess a secret positive integer by using the
    `guess_number_tool`.
    The tool will provide feedback on each guess.
    Your objective is to keep guessing until guess_number_tool returns
    'status: success'.
    Start by guessing 50, and use the tool's feedback to adjust your guesses
    and find the target number.""",
    model="gemini-2.5-flash",
    tools=[guess_number_tool],
)


app = App(
    name=APP_NAME,
    root_agent=root_agent,
    plugins=[
        CustomRetryPlugin(
            max_retries=20, throw_exception_if_retry_exceeded=False
        ),
        LoggingPlugin(),
    ],
)
