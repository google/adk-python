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

from google.adk.models.gemma_llm import Gemma
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest


@pytest.fixture
def llm_request():
  return LlmRequest(
      model="gemma-3-4b-it",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )


@pytest.fixture
def llm_request_with_duplicate_instruction():
  return LlmRequest(
      model="gemma-3-1b-it",
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="Talk like a pirate.")],
          ),
          Content(role="user", parts=[Part.from_text(text="Hello")]),
      ],
      config=types.GenerateContentConfig(
          response_modalities=[types.Modality.TEXT],
          system_instruction="Talk like a pirate.",
      ),
  )


@pytest.fixture
def llm_request_with_tools():
  return LlmRequest(
      model="gemma-3-1b-it",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name="search_web",
                          description="Search the web for a query.",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "query": types.Schema(type=types.Type.STRING)
                              },
                              required=["query"],
                          ),
                      ),
                      types.FunctionDeclaration(
                          name="get_current_time",
                          description="Gets the current time.",
                          parameters=types.Schema(
                              type=types.Type.OBJECT, properties={}
                          ),
                      ),
                  ]
              )
          ],
      ),
  )


@pytest.mark.asyncio
async def test_not_gemma_model():
  llm = Gemma()
  llm_request_bad_model = LlmRequest(
      model="not-a-gemma-model",
  )
  with pytest.raises(AssertionError, match=r".*model.*"):
    async for _ in llm.generate_content_async(llm_request_bad_model):
      pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_request",
    ["llm_request", "llm_request_with_duplicate_instruction"],
    indirect=True,
)
async def test_preprocess_request(llm_request):
  llm = Gemma()
  want_content_text = llm_request.config.system_instruction

  await llm._preprocess_request(llm_request)

  # system instruction should be cleared
  assert not llm_request.config.system_instruction
  # should be two content bits now (deduped, if needed)
  assert len(llm_request.contents) == 2
  # first message in contents should be "user": <original sys instruction>
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts[0].text == want_content_text


@pytest.mark.asyncio
async def test_preprocess_request_with_tools(llm_request_with_tools):

  gemma = Gemma()
  await gemma._preprocess_request(llm_request_with_tools)

  assert not llm_request_with_tools.config.tools

  # The original user content should now be the second item
  assert llm_request_with_tools.contents[1].role == "user"
  assert llm_request_with_tools.contents[1].parts[0].text == "Hello"

  sys_instruct_text = llm_request_with_tools.contents[0].parts[0].text
  assert sys_instruct_text is not None
  assert "You have access to the following functions" in sys_instruct_text
  assert (
      """{"description":"Search the web for a query.","name":"search_web","""
      in sys_instruct_text
  )
  assert (
      """{"description":"Gets the current time.","name":"get_current_time","parameters":{"properties":{}"""
      in sys_instruct_text
  )


@pytest.mark.asyncio
async def test_preprocess_request_with_function_response():
  # Simulate an LlmRequest with a function response
  func_response_data = types.FunctionResponse(
      name="search_web", response={"results": [{"title": "ADK"}]}
  )
  llm_request = LlmRequest(
      model="gemma-3-1b-it",
      contents=[
          types.Content(
              role="model",
              parts=[types.Part(function_response=func_response_data)],
          )
      ],
      config=types.GenerateContentConfig(),
  )

  gemma = Gemma()
  await gemma._preprocess_request(llm_request)

  # Assertions: function response converted to user role text content
  assert llm_request.contents
  assert len(llm_request.contents) == 1
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts
  assert (
      llm_request.contents[0].parts[0].text
      == 'Invoking tool `search_web` produced: `{"results": [{"title":'
      ' "ADK"}]}`.'
  )
  assert llm_request.contents[0].parts[0].function_response is None
  assert llm_request.contents[0].parts[0].function_call is None


@pytest.mark.asyncio
async def test_preprocess_request_with_function_call():
  func_call_data = types.FunctionCall(name="get_current_time", args={})
  llm_request = LlmRequest(
      model="gemma-3-1b-it",
      contents=[
          types.Content(
              role="user", parts=[types.Part(function_call=func_call_data)]
          )
      ],
  )

  gemma = Gemma()
  await gemma._preprocess_request(llm_request)

  assert len(llm_request.contents) == 1
  assert llm_request.contents[0].role == "model"
  expected_text = func_call_data.model_dump_json(exclude_none=True)
  assert llm_request.contents[0].parts
  got_part = llm_request.contents[0].parts[0]
  assert got_part.text == expected_text
  assert got_part.function_call is None
  assert got_part.function_response is None


@pytest.mark.asyncio
async def test_preprocess_request_with_mixed_content():
  func_call = types.FunctionCall(name="get_weather", args={"city": "London"})
  func_response = types.FunctionResponse(
      name="get_weather", response={"temp": "15C"}
  )

  llm_request = LlmRequest(
      model="gemma-3-1b-it",
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="Hello!")]
          ),
          types.Content(
              role="model", parts=[types.Part(function_call=func_call)]
          ),
          types.Content(
              role="some_function",
              parts=[types.Part(function_response=func_response)],
          ),
          types.Content(
              role="user", parts=[types.Part.from_text(text="How are you?")]
          ),
      ],
  )

  gemma = Gemma()
  await gemma._preprocess_request(llm_request)

  # Assertions
  assert len(llm_request.contents) == 4

  # First part: original user text
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts
  assert llm_request.contents[0].parts[0].text == "Hello!"

  # Second part: function call converted to model text
  assert llm_request.contents[1].role == "model"
  assert llm_request.contents[1].parts
  assert llm_request.contents[1].parts[0].text == func_call.model_dump_json(
      exclude_none=True
  )

  # Third part: function response converted to user text
  assert llm_request.contents[2].role == "user"
  assert llm_request.contents[2].parts
  assert (
      llm_request.contents[2].parts[0].text
      == 'Invoking tool `get_weather` produced: `{"temp": "15C"}`.'
  )

  # Fourth part: original user text
  assert llm_request.contents[3].role == "user"
  assert llm_request.contents[3].parts
  assert llm_request.contents[3].parts[0].text == "How are you?"


def test_process_response():
  # Simulate a response from Gemma that should be converted to a FunctionCall
  json_function_call_str = (
      '{"name": "search_web", "parameters": {"query": "latest news"}}'
  )
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=json_function_call_str)]
      )
  )

  gemma = Gemma()
  gemma._extract_function_calls_from_response(llm_response=llm_response)

  # Assert that the content was transformed into a FunctionCall
  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "search_web"
  assert part.function_call.args == {"query": "latest news"}
  # Assert that the entire part matches the expected structure
  expected_function_call = types.FunctionCall(
      name="search_web", args={"query": "latest news"}
  )
  expected_part = Part(function_call=expected_function_call)
  assert part == expected_part
  assert part.text is None  # Ensure text part is cleared


def test_process_response_invalid_json_text():
  # Simulate a response with plain text that is not JSON
  original_text = "This is a regular text response."
  llm_response = LlmResponse(
      content=Content(role="model", parts=[Part.from_text(text=original_text)])
  )

  gemma = Gemma()
  gemma._extract_function_calls_from_response(llm_response=llm_response)

  # Assert that the content remains unchanged
  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  assert llm_response.content.parts[0].text == original_text
  assert llm_response.content.parts[0].function_call is None


def test_process_response_malformed_json():
  # Simulate a response with valid JSON but not in the function call format
  malformed_json_str = '{"not_a_function": "value", "another_field": 123}'
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=malformed_json_str)]
      )
  )
  gemma = Gemma()
  gemma._extract_function_calls_from_response(llm_response=llm_response)

  # Assert that the content remains unchanged because it doesn't match the expected schema
  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  assert llm_response.content.parts[0].text == malformed_json_str
  assert llm_response.content.parts[0].function_call is None


def test_process_response_empty_content_or_multiple_parts():
  gemma = Gemma()

  # Test case 1: LlmResponse with no content
  llm_response_no_content = LlmResponse(content=None)
  gemma._extract_function_calls_from_response(
      llm_response=llm_response_no_content
  )
  assert llm_response_no_content.content is None

  # Test case 2: LlmResponse with empty parts list
  llm_response_empty_parts = LlmResponse(
      content=Content(role="model", parts=[])
  )
  gemma._extract_function_calls_from_response(
      llm_response=llm_response_empty_parts
  )
  assert llm_response_empty_parts.content
  assert not llm_response_empty_parts.content.parts

  # Test case 3: LlmResponse with multiple parts
  llm_response_multiple_parts = LlmResponse(
      content=Content(
          role="model",
          parts=[
              Part.from_text(text="part one"),
              Part.from_text(text="part two"),
          ],
      )
  )
  original_parts = list(
      llm_response_multiple_parts.content.parts
  )  # Copy for comparison
  gemma._extract_function_calls_from_response(
      llm_response=llm_response_multiple_parts
  )
  assert llm_response_multiple_parts.content
  assert (
      llm_response_multiple_parts.content.parts == original_parts
  )  # Should remain unchanged

  # Test case 4: LlmResponse with one part, but empty text
  llm_response_empty_text_part = LlmResponse(
      content=Content(role="model", parts=[Part.from_text(text="")])
  )
  gemma._extract_function_calls_from_response(
      llm_response=llm_response_empty_text_part
  )
  assert llm_response_empty_text_part.content
  assert llm_response_empty_text_part.content.parts
  assert llm_response_empty_text_part.content.parts[0].text == ""
  assert llm_response_empty_text_part.content.parts[0].function_call is None


def test_process_response_with_markdown_json_block():
  # Simulate a response from Gemma with a JSON function call in a markdown block
  json_function_call_str = """
```json
{"name": "search_web", "parameters": {"query": "latest news"}}
```"""
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=json_function_call_str)]
      )
  )

  gemma = Gemma()
  gemma._extract_function_calls_from_response(llm_response)

  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "search_web"
  assert part.function_call.args == {"query": "latest news"}
  assert part.text is None


def test_process_response_with_markdown_tool_code_block():
  # Simulate a response from Gemma with a JSON function call in a 'tool_code' markdown block
  json_function_call_str = """
Some text before.
```tool_code
{"name": "get_current_time", "parameters": {}}
```
And some text after."""
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=json_function_call_str)]
      )
  )

  gemma = Gemma()
  gemma._extract_function_calls_from_response(llm_response)

  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "get_current_time"
  assert part.function_call.args == {}
  assert part.text is None


def test_process_response_with_embedded_json():
  # Simulate a response with valid JSON embedded in text
  embedded_json_str = (
      'Please call the tool: {"name": "search_web", "parameters": {"query":'
      ' "new features"}} thanks!'
  )
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=embedded_json_str)]
      )
  )

  gemma = Gemma()
  gemma._extract_function_calls_from_response(llm_response)

  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "search_web"
  assert part.function_call.args == {"query": "new features"}
  assert part.text is None


def test_process_response_flexible_parsing():
  # Test with "function" and "args" keys as supported by GemmaFunctionCallModel
  flexible_json_str = '{"function": "do_something", "args": {"value": 123}}'
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=flexible_json_str)]
      )
  )

  gemma = Gemma()
  gemma._extract_function_calls_from_response(llm_response)

  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "do_something"
  assert part.function_call.args == {"value": 123}
  assert part.text is None


def test_process_response_last_json_object():
  # Simulate a response with multiple JSON objects, ensuring the last valid one is picked
  multiple_json_str = (
      'I thought about {"name": "first_call", "parameters": {"a": 1}} but then'
      ' decided to call: {"name": "second_call", "parameters": {"b": 2}}'
  )
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=multiple_json_str)]
      )
  )

  gemma = Gemma()
  gemma._extract_function_calls_from_response(llm_response)

  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "second_call"
  assert part.function_call.args == {"b": 2}
  assert part.text is None


# Tests for Gemma3Ollama (only run when LiteLLM is installed)
try:
  from google.adk.models.gemma_llm import Gemma3Ollama

  def test_gemma3_ollama_supported_models():
    assert Gemma3Ollama.supported_models() == [r"ollama/gemma3.*"]

  @pytest.mark.parametrize(
      "model_arg,expected_model",
      [
          (None, "ollama/gemma3:12b"),
          ("ollama/gemma3:27b", "ollama/gemma3:27b"),
      ],
  )
  def test_gemma3_ollama_model(model_arg, expected_model):
    model = (
        Gemma3Ollama() if model_arg is None else Gemma3Ollama(model=model_arg)
    )
    assert model.model == expected_model

except ImportError:
  # LiteLLM not installed, skip Gemma3Ollama tests
  pass
