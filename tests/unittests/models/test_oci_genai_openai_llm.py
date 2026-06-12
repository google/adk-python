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

"""Unit tests for OCIGenAIOpenAILlm (OpenAI-compat v1 transport).

These cover URL/auth construction and message conversion without making
network calls. Live integration tests live alongside the native-SDK
provider in tests/integration/models/.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from google.adk.models.oci_genai_openai_llm import _content_to_openai_messages
from google.adk.models.oci_genai_openai_llm import _openai_response_to_llm_response
from google.adk.models.oci_genai_openai_llm import _tools_to_openai
from google.adk.models.oci_genai_openai_llm import OCIGenAIOpenAILlm
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest

# ---------------------------------------------------------------------------
# URL + auth construction
# ---------------------------------------------------------------------------


def test_default_base_url_uses_v1_path():
  llm = OCIGenAIOpenAILlm(
      model="google.gemini-2.5-flash",
      auth_type="BEARER_TOKEN",
      api_key="fake",
      compartment_id="ocid1.compartment.oc1..xxx",
  )
  assert llm._resolve_base_url() == (
      "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      "/20231130/actions/v1"
  )


def test_explicit_service_endpoint_wins(monkeypatch):
  custom = "https://oci-proxy.example.com/v1"
  llm = OCIGenAIOpenAILlm(
      model="google.gemini-2.5-flash",
      service_endpoint=custom,
      auth_type="BEARER_TOKEN",
      api_key="fake",
      compartment_id="ocid1.compartment.oc1..xxx",
  )
  assert llm._resolve_base_url() == custom


def test_env_service_endpoint_used_when_not_set(monkeypatch):
  monkeypatch.setenv("OCI_SERVICE_ENDPOINT", "https://env-set.example.com/v1")
  llm = OCIGenAIOpenAILlm(
      model="google.gemini-2.5-flash",
      auth_type="BEARER_TOKEN",
      api_key="fake",
      compartment_id="ocid1.compartment.oc1..xxx",
  )
  assert llm._resolve_base_url() == "https://env-set.example.com/v1"


def test_region_changes_default_base_url():
  llm = OCIGenAIOpenAILlm(
      model="google.gemini-2.5-flash",
      region="eu-frankfurt-1",
      auth_type="BEARER_TOKEN",
      api_key="fake",
      compartment_id="ocid1.compartment.oc1..xxx",
  )
  assert "eu-frankfurt-1" in llm._resolve_base_url()
  assert llm._resolve_base_url().endswith("/20231130/actions/v1")


# ---------------------------------------------------------------------------
# BEARER_TOKEN validation
# ---------------------------------------------------------------------------


def test_bearer_token_requires_api_key(monkeypatch):
  monkeypatch.delenv("OCI_GENAI_API_KEY", raising=False)
  llm = OCIGenAIOpenAILlm(
      model="google.gemini-2.5-flash",
      auth_type="BEARER_TOKEN",
      compartment_id="ocid1.compartment.oc1..xxx",
  )
  with pytest.raises(ValueError, match="BEARER_TOKEN.*api_key"):
    llm._build_client()


def test_bearer_token_picks_up_env_api_key(monkeypatch):
  monkeypatch.setenv("OCI_GENAI_API_KEY", "env-bearer-key")
  llm = OCIGenAIOpenAILlm(
      model="google.gemini-2.5-flash",
      auth_type="BEARER_TOKEN",
      compartment_id="ocid1.compartment.oc1..xxx",
  )
  client = llm._build_client()
  assert client.api_key == "env-bearer-key"
  assert "20231130/actions/v1" in str(client.base_url)
  assert (
      client.default_headers["opc-compartment-id"]
      == "ocid1.compartment.oc1..xxx"
  )


def test_bearer_token_requires_compartment(monkeypatch):
  monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
  llm = OCIGenAIOpenAILlm(
      model="google.gemini-2.5-flash",
      auth_type="BEARER_TOKEN",
      api_key="fake",
  )
  with pytest.raises(ValueError, match="compartment_id"):
    llm._build_client()


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def test_content_to_openai_messages_text():
  contents = [
      Content(role="user", parts=[Part(text="hi"), Part(text=" there")]),
  ]
  msgs = _content_to_openai_messages(contents, system_instruction="be brief")
  assert msgs[0] == {"role": "system", "content": "be brief"}
  assert msgs[1] == {"role": "user", "content": "hi there"}


def test_content_to_openai_messages_model_role_is_assistant():
  contents = [Content(role="model", parts=[Part(text="howdy")])]
  msgs = _content_to_openai_messages(contents, system_instruction=None)
  assert msgs[0]["role"] == "assistant"


def test_content_to_openai_messages_function_call():
  fc_part = Part.from_function_call(name="lookup", args={"q": "weather"})
  contents = [Content(role="model", parts=[fc_part])]
  msgs = _content_to_openai_messages(contents, system_instruction=None)
  assert msgs[0]["role"] == "assistant"
  assert msgs[0]["tool_calls"][0]["function"]["name"] == "lookup"
  assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"]) == {
      "q": "weather"
  }


def test_content_to_openai_messages_function_response_becomes_tool_role():
  fr_part = Part.from_function_response(name="lookup", response={"temp": 72})
  contents = [Content(role="user", parts=[fr_part])]
  msgs = _content_to_openai_messages(contents, system_instruction=None)
  assert msgs[0]["role"] == "tool"
  assert json.loads(msgs[0]["content"]) == {"temp": 72}


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def test_tools_to_openai_none_returns_none():
  assert _tools_to_openai(None) is None
  assert _tools_to_openai([]) is None


def test_tools_to_openai_emits_function_schema():
  decl = types.FunctionDeclaration(
      name="get_weather",
      description="Get current weather",
      parameters=types.Schema(
          type=types.Type.OBJECT,
          properties={"city": types.Schema(type=types.Type.STRING)},
      ),
  )
  out = _tools_to_openai([types.Tool(function_declarations=[decl])])
  assert out is not None
  assert out[0]["type"] == "function"
  assert out[0]["function"]["name"] == "get_weather"
  assert out[0]["function"]["description"] == "Get current weather"
  assert out[0]["function"]["parameters"]["type"] == "OBJECT"


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


def _fake_openai_response(
    text: str | None, tool_calls: list[dict] | None = None
):
  """Build a minimal openai-like ChatCompletion stub."""
  tcs = []
  for tc in tool_calls or []:
    tcs.append(
        SimpleNamespace(
            id=tc.get("id", "tc-1"),
            function=SimpleNamespace(
                name=tc["name"], arguments=tc["arguments"]
            ),
        )
    )
  choice = SimpleNamespace(
      message=SimpleNamespace(content=text, tool_calls=tcs or None),
  )
  usage = SimpleNamespace(
      prompt_tokens=10, completion_tokens=5, total_tokens=15
  )
  return SimpleNamespace(choices=[choice], usage=usage)


def test_response_to_llm_response_text_only():
  resp = _openai_response_to_llm_response(_fake_openai_response("hello"))
  assert resp.content is not None
  assert resp.content.parts[0].text == "hello"
  assert resp.usage_metadata.total_token_count == 15


def test_response_to_llm_response_tool_call():
  resp = _openai_response_to_llm_response(
      _fake_openai_response(
          None,
          tool_calls=[
              {"name": "lookup", "arguments": '{"q":"weather"}', "id": "x"}
          ],
      )
  )
  fc = resp.content.parts[0].function_call
  assert fc is not None
  assert fc.name == "lookup"
  assert fc.args == {"q": "weather"}


def test_response_to_llm_response_tool_call_bad_json_falls_back_to_empty():
  resp = _openai_response_to_llm_response(
      _fake_openai_response(
          None,
          tool_calls=[{"name": "lookup", "arguments": "not-json", "id": "x"}],
      )
  )
  assert resp.content.parts[0].function_call.args == {}


# ---------------------------------------------------------------------------
# Package-surface import
# ---------------------------------------------------------------------------


def test_lazy_import_from_package():
  """`from google.adk.models import OCIGenAIOpenAILlm` resolves lazily."""
  from google.adk.models import OCIGenAIOpenAILlm as imported

  assert imported is OCIGenAIOpenAILlm


def test_supported_models_is_empty():
  """OCIGenAIOpenAILlm does not auto-route; users instantiate explicitly."""
  assert OCIGenAIOpenAILlm.supported_models() == []
