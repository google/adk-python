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

import json
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.ollama_llm import Ollama
from google.genai import types
import pytest

#
# -----------------------------------
# Helpers
# -----------------------------------
#


def mock_response_ok(text="Hello world", tool_calls=None):
  """Create a typical Ollama /api/chat response."""
  message = {"content": text}
  if tool_calls:
    message["tool_calls"] = tool_calls
  return {"message": message}


#
# -----------------------------------
# Test: model extraction
# -----------------------------------
#


def test_extract_model_name_basic():
  o = Ollama(model="ollama/mistral")
  assert o._extract_model_name("ollama/mistral") == "mistral"


def test_extract_model_name_chat_prefix():
  o = Ollama(model="ollama_chat/llama3.1")
  assert o._extract_model_name("ollama_chat/llama3.1") == "llama3.1"


def test_extract_model_name_no_prefix():
  o = Ollama(model="mistral")
  assert o._extract_model_name("mistral") == "mistral"


#
# -----------------------------------
# Test: message conversion
# -----------------------------------
#


def test_convert_messages_basic():
  o = Ollama()
  req = LlmRequest(contents=[types.Content(role="user", parts=[types.Part.from_text("Hi")])])
  msgs = o._convert_messages(req)
  assert msgs[0]["role"] == "user"
  assert msgs[0]["content"] == "Hi"


def test_convert_messages_with_system():
  o = Ollama()
  req = LlmRequest(
      contents=[types.Content(role="user", parts=[types.Part.from_text("X")])],
      config=types.GenerateContentConfig(system_instruction="SYS"),
  )
  msgs = o._convert_messages(req)
  assert msgs[0]["role"] == "system"
  assert msgs[0]["content"] == "SYS"
  assert msgs[1]["content"] == "X"


#
# -----------------------------------
# Test: content → text
# -----------------------------------
#


def test_content_to_text_basic():
  o = Ollama()
  content = types.Content(role="user", parts=[types.Part.from_text("ABC")])
  assert o._content_to_text(content) == "ABC"


def test_content_to_text_function_call():
  o = Ollama()
  part = types.Part.from_function_call(name="add", args={"x": 1, "y": 2})
  part.function_call.id = "call123"

  content = types.Content(role="assistant", parts=[part])
  txt = o._content_to_text(content)

  assert "[tool_call name=add]" in txt
  assert '"x": 1' in txt


def test_content_to_text_tool_response():
  o = Ollama()
  part = types.Part.from_function_response(name="add", response={"z": 5})
  content = types.Content(role="tool", parts=[part])
  txt = o._content_to_text(content)

  assert "[tool_response name=add]" in txt
  assert '"z": 5' in txt


#
# -----------------------------------
# Test: tool conversion
# -----------------------------------
#


def test_convert_tools_basic():
  o = Ollama()
  req = LlmRequest(
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name="add",
                          description="Add numbers",
                          parameters=types.Schema(
                              type=types.Type.OBJECT, properties={"x": types.Schema(type=types.Type.NUMBER)}
                          ),
                      )
                  ]
              )
          ]
      )
  )
  tools = o._convert_tools(req)
  assert tools[0]["function"]["name"] == "add"
  assert tools[0]["function"]["parameters"]["type"] == "object"


#
# -----------------------------------
# Test: POST wrapper
# -----------------------------------
#


def test_post_chat_success(monkeypatch):
  fake_response = {"message": {"content": "OK"}}

  def fake_urlopen(req, timeout=0):
    class Resp:

      def read(self):
        return json.dumps(fake_response).encode("utf-8")

    return Resp()

  monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

  o = Ollama()
  resp = o._post_chat({"model": "x"})
  assert resp["message"]["content"] == "OK"


#
# -----------------------------------
# Test: _to_llm_response
# -----------------------------------
#


def test_to_llm_response_text():
  o = Ollama()
  resp = mock_response_ok("Hi")

  out = o._to_llm_response(resp)
  assert isinstance(out, LlmResponse)
  assert out.content.parts[0].text == "Hi"


def test_to_llm_response_tool_call():
  o = Ollama()
  tool_call = {"id": "abc", "function": {"name": "add", "arguments": '{"x": 1}'}}

  resp = mock_response_ok(tool_calls=[tool_call])
  out = o._to_llm_response(resp)

  fc = out.content.parts[0].function_call
  assert fc.name == "add"
  assert fc.args == {"x": 1}
  assert fc.id == "abc"


def test_to_llm_response_tool_call_bad_json():
  o = Ollama()
  tool_call = {"id": "zzz", "function": {"name": "add", "arguments": "{BAD_JSON"}}

  resp = mock_response_ok(tool_calls=[tool_call])
  out = o._to_llm_response(resp)

  fc = out.content.parts[0].function_call
  assert fc.args == {}  # BAD JSON → fallback to {}


def test_to_llm_response_usage_metadata():
  o = Ollama()
  resp = mock_response_ok("Hi")
  resp["prompt_eval_count"] = 10
  resp["eval_count"] = 5

  out = o._to_llm_response(resp)

  assert out.usage_metadata is not None
  assert out.usage_metadata.prompt_token_count == 10
  assert out.usage_metadata.candidates_token_count == 5
  assert out.usage_metadata.total_token_count == 15


#
# -----------------------------------
# async: generate_content_async
# -----------------------------------
#


@pytest.mark.asyncio
async def test_generate_content_async_basic(monkeypatch):
  resp = mock_response_ok("Hello!")

  async def fake_thread(fn, *args):
    return resp

  monkeypatch.setattr("asyncio.to_thread", fake_thread)

  o = Ollama(model="ollama/mistral")
  req = LlmRequest(contents=[types.Content(role="user", parts=[types.Part.from_text("Hi")])])

  results = [r async for r in o.generate_content_async(req)]
  assert results[0].content.parts[0].text == "Hello!"


@pytest.mark.asyncio
async def test_generate_content_async_error(monkeypatch):
  async def fake_thread(fn, *args):
    raise RuntimeError("boom")

  monkeypatch.setattr("asyncio.to_thread", fake_thread)

  o = Ollama()
  req = LlmRequest(contents=[types.Content(role="user", parts=[])])

  results = [r async for r in o.generate_content_async(req)]
  assert results[0].error_code == "OLLAMA_ERROR"


#
# -----------------------------------
# Test: model override
# -----------------------------------
#


@pytest.mark.asyncio
async def test_model_override(monkeypatch):
  resp = mock_response_ok("Hello")
  resp["model"] = "override"

  async def fake_thread(fn, *args):
    payload = args[0]
    assert payload["model"] == "override"  # important
    return resp

  monkeypatch.setattr("asyncio.to_thread", fake_thread)

  o = Ollama(model="default")
  req = LlmRequest(model="override", contents=[types.Content(role="user", parts=[types.Part.from_text("X")])])

  out = [r async for r in o.generate_content_async(req)][0]
  assert out.model_version == "override"
