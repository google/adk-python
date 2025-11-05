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

from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest

_TEST_MODEL_NAME = "vertex_ai/meta/llama-3.1-405b-instruct-maas"

_SYSTEM_PROMPT = """You are a helpful assistant."""


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
def oss_llm():
  return LiteLlm(model=_TEST_MODEL_NAME)


@pytest.fixture
def llm_request():
  return LlmRequest(
      model=_TEST_MODEL_NAME,
      contents=[Content(role="user", parts=[Part.from_text(text="hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction=_SYSTEM_PROMPT,
      ),
  )


@pytest.fixture
def llm_request_with_tools():
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
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name="get_weather",
                          description="Get the weather in a given location",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "city": types.Schema(
                                      type=types.Type.STRING,
                                      description=(
                                          "The city to get the weather for."
                                      ),
                                  ),
                              },
                              required=["city"],
                          ),
                      )
                  ]
              )
          ],
      ),
  )


@pytest.mark.asyncio
async def test_generate_content_async(oss_llm, llm_request):
  async for response in oss_llm.generate_content_async(llm_request):
    assert isinstance(response, LlmResponse)
    assert response.content.parts[0].text


@pytest.mark.asyncio
async def test_generate_content_async(oss_llm, llm_request):
  responses = [
      resp
      async for resp in oss_llm.generate_content_async(
          llm_request, stream=False
      )
  ]
  part = responses[0].content.parts[0]
  assert len(part.text) > 0


@pytest.mark.asyncio
async def test_generate_content_async_with_tools(
    oss_llm, llm_request_with_tools
):
  responses = [
      resp
      async for resp in oss_llm.generate_content_async(
          llm_request_with_tools, stream=False
      )
  ]
  function_call = responses[0].content.parts[0].function_call
  assert function_call.name == "get_weather"
  assert function_call.args["city"] == "San Francisco"


@pytest.mark.asyncio
async def test_generate_content_async_stream(oss_llm, llm_request):
  responses = [
      resp
      async for resp in oss_llm.generate_content_async(llm_request, stream=True)
  ]
  text = ""
  for i in range(len(responses) - 1):
    assert responses[i].partial is True
    assert responses[i].content.parts[0].text
    text += responses[i].content.parts[0].text

  # Last message should be accumulated text
  assert responses[-1].content.parts[0].text == text
  assert not responses[-1].partial


@pytest.mark.asyncio
async def test_generate_content_async_stream_with_tools(
    oss_llm, llm_request_with_tools
):
  responses = [
      resp
      async for resp in oss_llm.generate_content_async(
          llm_request_with_tools, stream=True
      )
  ]
  function_call = responses[-1].content.parts[0].function_call
  assert function_call.name == "get_weather"
  assert function_call.args["city"] == "San Francisco"
