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
from google.genai import types


def concat_number_and_string(num: int, s: str) -> str:
  """Concatenate a number and a string.

  Args:
    num: The number to concatenate.
    s: The string to concatenate.

  Returns:
    The concatenated string.
  """
  return str(num) + ': ' + s


def write_document(document: str) -> dict[str, str]:
  """Write a document."""
  return {'status': 'ok'}


root_agent = Agent(
    model='gemini-3-pro-preview',
    name='hello_world_stream_fc_args',
    description='Demo agent showcasing streaming function call arguments.',
    instruction="""
      You are a helpful assistant.
      You can use the `concat_number_and_string` tool to concatenate a number and a string.
      You should always call the concat_number_and_string tool to concatenate a number and a string.
      You should never concatenate on your own.

      You can use the `write_document` tool to write a document.
      You should always call the write_document tool to write a document.
      You should never write a document on your own.
    """,
    tools=[
        concat_number_and_string,
        write_document,
    ],
    generate_content_config=types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True,
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                stream_function_call_arguments=True,
            ),
        ),
    ),
)
