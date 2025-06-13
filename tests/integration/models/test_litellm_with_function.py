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

from google.adk.models import LlmRequest
from google.adk.models import LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import litellm
import pytest

litellm.add_function_to_prompt = True

_TEST_MODEL_NAME = "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas"


_SYSTEM_PROMPT = """
You are a helpful assistant, and call tools optionally.
If call tools, the tool format should be in json, and the tool arguments should be parsed from users inputs.
"""


_FUNCTIONS = [{
    "name": "get_weather",
    "description": "Get the weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city, e.g. San Francisco",
            },
        },
        "required": ["city"],
    },
}]


def get_weather(city: str) -> str:
  """Simulates a web search. Use it get information on weather.

  Args:
      city: A string containing the location to get weather information for.

  Returns:
      A string with the simulated weather information for the queried city.
  """
  if "sf" in city.lower() or "san francisco" in city.lower():
    return "It's 70 degrees and foggy."
  return "It's 80 degrees and sunny."


@pytest.fixture
def oss_llm_with_function():
  return LiteLlm(model=_TEST_MODEL_NAME, functions=_FUNCTIONS)


@pytest.fixture
def llm_request():
  return LlmRequest(
      model=_TEST_MODEL_NAME,
      contents=[
          Content(
              role="user",
              parts=[
                  Part.from_text(text="What is the weather in San Francisco?")
              ],
          )
      ],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction=_SYSTEM_PROMPT,
      ),
  )


# Note that, this test disabled streaming because streaming is not supported
# properly in the current test model for now.
@pytest.mark.asyncio
async def test_generate_content_asyn_with_function(
    oss_llm_with_function, llm_request
):
  responses = [
      resp
      async for resp in oss_llm_with_function.generate_content_async(
          llm_request, stream=False
      )
  ]
  function_call = responses[0].content.parts[0].function_call
  assert function_call.name == "get_weather"
  assert function_call.args["city"] == "San Francisco"
