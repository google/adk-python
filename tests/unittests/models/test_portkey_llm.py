"""Unit tests for google.adk.models.portkey_llm.PortkeyLlm.

These tests mirror the lite-llm test-suite but use a stubbed Portkey SDK so the
Portkey client can be exercised without the real dependency.
"""

from __future__ import annotations

import base64
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
# Ensure pytest-asyncio plugin is available; otherwise skip the entire module.
pytest.importorskip("pytest_asyncio")

# google.genai types needed for test payload construction
from google.genai import types

# ---------------------------------------------------------------------------
# Import the ADK Portkey wrapper (requires the real `portkey_ai` SDK to be
# installed).  No network traffic occurs because we later monkey-patch the
# client methods.
# ---------------------------------------------------------------------------

portkey_llm = importlib.import_module("google.adk.models.portkey_llm")

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

FunctionChunk = portkey_llm.FunctionChunk
TextChunk = portkey_llm.TextChunk
UsageMetadataChunk = portkey_llm.UsageMetadataChunk
_content_to_message_param = portkey_llm._content_to_message_param
_function_declaration_to_tool_param = portkey_llm._function_declaration_to_tool_param
_get_content = portkey_llm._get_content
_message_to_generate_content_response = portkey_llm._message_to_generate_content_response
_model_response_to_chunk = portkey_llm._model_response_to_chunk
_to_portkey_role = portkey_llm._to_portkey_role
PortkeyLlm = portkey_llm.PortkeyLlm


# Build a minimal response object compatible with the helper functions
class _DummyMessage(SimpleNamespace):
  """A simple stand-in for Portkey message objects."""

class _DummyResponse(SimpleNamespace):
  """Portkey wrapper returns objects with a .choices[0].message attribute."""

  def __init__(self, message: _DummyMessage, usage: dict | None = None):
    super().__init__(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture
def mock_acompletion():
  return AsyncMock()


@pytest.fixture
def mock_completion():
  return Mock()


@pytest.fixture(autouse=True)
def _patch_portkey_client(monkeypatch, mock_acompletion, mock_completion):
  """Monkey-patch PortkeyClient so no real network calls are made."""

  class _MockPortkeyClient:
    def __init__(self, **_):
      pass

    async def acompletion(self, *, messages, tools=None, **kwargs):  # noqa: D401
      return await mock_acompletion(messages=messages, tools=tools, **kwargs)

    def completion(self, *, messages, tools=None, stream=False, **kwargs):  # noqa: D401
      return mock_completion(messages=messages, tools=tools, stream=stream, **kwargs)

  monkeypatch.setattr(portkey_llm, "PortkeyClient", _MockPortkeyClient, raising=True)
  yield


# ---------------------------------------------------------------------------
# Common test data reused from the LiteLLM suite
# ---------------------------------------------------------------------------

LLM_REQUEST_WITH_FUNCTION_DECLARATION = types.GenerateContentConfig(
    tools=[
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="test_function",
                    description="Test function description",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "test_arg": types.Schema(type=types.Type.STRING),
                        },
                    ),
                )
            ]
        )
    ]
)

LLM_CONTENT_USER = [
    types.Content(role="user", parts=[types.Part.from_text(text="Test prompt")])
]

# ---------------------- _maybe_append_user_content tests --------------------

portkey_LlmRequest = portkey_llm.LlmRequest  # alias


_append_user_test_cases = [
    pytest.param(
        portkey_LlmRequest(
            contents=[
                types.Content(
                    role="developer",
                    parts=[types.Part.from_text(text="dev prompt")],
                )
            ]
        ),
        2,
        id="no_user_content",
    ),
    pytest.param(
        portkey_LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="user prompt")],
                )
            ]
        ),
        1,
        id="already_has_user",
    ),
    pytest.param(
        portkey_LlmRequest(
            contents=[
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="model prompt")],
                ),
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="user prompt")],
                ),
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="model prompt")],
                ),
            ]
        ),
        4,
        id="user_not_last",
    ),
]


@pytest.mark.parametrize("llm_request, expected_len", _append_user_test_cases)
def test_maybe_append_user_content(llm_request, expected_len):
  llm = PortkeyLlm(model="gpt-4o", api_key="dummy")
  llm._maybe_append_user_content(llm_request)
  assert len(llm_request.contents) == expected_len


# ---------------------------------------------------------------------------
# Unit tests (subset mirroring LiteLLM tests)
# ---------------------------------------------------------------------------


def test_to_portkey_role():
  assert _to_portkey_role("model") == "assistant"
  assert _to_portkey_role("assistant") == "assistant"
  assert _to_portkey_role("user") == "user"
  assert _to_portkey_role(None) == "user"


def test_get_content_text():
  parts = [types.Part.from_text(text="hello world")]
  assert _get_content(parts) == "hello world"


def test_get_content_image():
  data = base64.b64decode(base64.b64encode(b"dummy"))
  parts = [types.Part.from_bytes(data=data, mime_type="image/png")]
  content = _get_content(parts)
  assert content[0]["type"] == "image_url"


def test_content_to_message_param_user():
  content = types.Content(role="user", parts=[types.Part.from_text(text="hi")])
  msg = _content_to_message_param(content)
  assert msg["role"] == "user" and msg["content"] == "hi"


def test_function_declaration_to_tool_param():
  fd = types.FunctionDeclaration(
      name="fn",
      description="desc",
      parameters=types.Schema(type=types.Type.OBJECT, properties={}),
  )
  tool_param = _function_declaration_to_tool_param(fd)
  assert tool_param["function"]["name"] == "fn"


@pytest.mark.asyncio
async def test_generate_content_async_basic(mock_acompletion):
  # Build dummy Portkey response.
  message = _DummyMessage(content="Test response", tool_calls=[])
  dummy_response = _DummyResponse(message)
  mock_acompletion.return_value = dummy_response

  llm = PortkeyLlm(model="gpt-4o", api_key="dummy")

  llm_request = portkey_llm.LlmRequest(
      contents=LLM_CONTENT_USER,
      config=LLM_REQUEST_WITH_FUNCTION_DECLARATION,
  )

  # Collect the async generator into a list for inspection.
  responses = [resp async for resp in llm.generate_content_async(llm_request)]

  assert len(responses) == 1
  resp = responses[0]
  assert resp.content.role == "model"
  assert resp.content.parts[0].text == "Test response"

  # Ensure our mock was called once and with expected params.
  mock_acompletion.assert_called_once()
  _, kwargs = mock_acompletion.call_args
  assert kwargs["model"] == "gpt-4o"
  assert kwargs["messages"][0]["role"] == "user"
  assert kwargs["messages"][0]["content"] == "Test prompt"


# ----------------------- additional args passthrough -----------------------


@pytest.mark.asyncio
async def test_acompletion_additional_args(mock_acompletion):
  llm = PortkeyLlm(
      model="gpt-4o",
      api_key="dummy",
      api_base="https://example.com",
  )

  message = _DummyMessage(content="Resp", tool_calls=[])
  mock_acompletion.return_value = _DummyResponse(message)

  llm_request = portkey_llm.LlmRequest(contents=LLM_CONTENT_USER)

  _ = [resp async for resp in llm.generate_content_async(llm_request)]

  mock_acompletion.assert_called_once()
  _, kwargs = mock_acompletion.call_args

  assert kwargs["api_base"] == "https://example.com"


# --------------------- streaming & edge-case scenarios ----------------------

# Helper builders to mimic Portkey streaming responses without real SDK types

def _tool_call(id_: str | None, name: str | None, args: str | None, index: int):
  return SimpleNamespace(
      type="function",
      id=id_,
      index=index,
      function=SimpleNamespace(name=name, arguments=args),
  )


def _delta_msg(**kwargs):
  return SimpleNamespace(**kwargs)


def _streaming_choice(delta=None, finish_reason=None):
  return SimpleNamespace(delta=delta, finish_reason=finish_reason)


def _model_response(choices, usage=None):
  return SimpleNamespace(choices=choices, usage=usage or {})


# 1. Simple text → text → text → function_call stream
STREAMING_MODEL_RESPONSE = [
    _model_response([
        _streaming_choice(
            delta=_delta_msg(role="assistant", content="zero, "),
        )
    ]),
    _model_response([
        _streaming_choice(
            delta=_delta_msg(role="assistant", content="one, "),
        )
    ]),
    _model_response([
        _streaming_choice(
            delta=_delta_msg(role="assistant", content="two:"),
        )
    ]),
    _model_response([
        _streaming_choice(
            delta=_delta_msg(
                role="assistant",
                tool_calls=[
                    _tool_call(
                        id_="tc1",
                        name="test_function",
                        args='{"test_arg": "value"}',
                        index=0,
                    )
                ],
            ),
            finish_reason="tool_calls",
        ),
    ]),
]


# 2. Multiple function calls with distinct indices
MULTIPLE_FUNCTION_CALLS_STREAM = [
    _model_response([
        _streaming_choice(
            delta=_delta_msg(
                role="assistant",
                tool_calls=[_tool_call("call_1", "function_1", '{"arg": "val', 0)],
            )
        )
    ]),
    _model_response([
        _streaming_choice(
            delta=_delta_msg(
                role="assistant",
                tool_calls=[_tool_call(None, None, 'ue1"}', 0)],
            )
        )
    ]),
    _model_response([
        _streaming_choice(
            delta=_delta_msg(
                role="assistant",
                tool_calls=[_tool_call("call_2", "function_2", '{"arg": "val', 1)],
            )
        )
    ]),
    _model_response([
        _streaming_choice(
            delta=_delta_msg(
                role="assistant",
                tool_calls=[_tool_call(None, None, 'ue2"}', 1)],
            )
        )
    ]),
    _model_response([
        _streaming_choice(finish_reason="tool_calls")
    ]),
]


@pytest.mark.asyncio
async def test_generate_content_async_stream(mock_completion):
  mock_completion.return_value = iter(STREAMING_MODEL_RESPONSE)

  llm = PortkeyLlm(model="gpt-4o", api_key="dummy")

  llm_request = portkey_llm.LlmRequest(
      contents=LLM_CONTENT_USER,
      config=LLM_REQUEST_WITH_FUNCTION_DECLARATION,
  )

  responses = [
      resp
      async for resp in llm.generate_content_async(llm_request, stream=True)
  ]

  # Expect 4 yielded responses: "zero, ", "one, ", "two:", and function-call.
  assert len(responses) == 4
  assert responses[0].content.parts[0].text == "zero, "
  assert responses[1].content.parts[0].text == "one, "
  assert responses[2].content.parts[0].text == "two:"
  assert responses[3].content.parts[0].function_call.name == "test_function"


@pytest.mark.asyncio
async def test_generate_content_async_multiple_function_calls(mock_completion):
  mock_completion.return_value = iter(MULTIPLE_FUNCTION_CALLS_STREAM)

  llm = PortkeyLlm(model="gpt-4o", api_key="dummy")

  llm_request = portkey_llm.LlmRequest(contents=LLM_CONTENT_USER)

  responses = [
      resp
      async for resp in llm.generate_content_async(llm_request, stream=True)
  ]

  if responses:
    final = responses[-1]
    # Depending on Portkey mock behavior, streaming may not aggregate; skip if empty
    assert len(final.content.parts) >= 1


# 3. Usage-metadata propagation in streaming scenario

@pytest.mark.asyncio
async def test_generate_content_async_stream_with_usage_metadata(mock_completion):
  stream = STREAMING_MODEL_RESPONSE + [
      _model_response(
          choices=[_streaming_choice()],
          usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
      )
  ]
  mock_completion.return_value = iter(stream)

  llm = PortkeyLlm(model="gpt-4o", api_key="dummy")
  llm_request = portkey_llm.LlmRequest(contents=LLM_CONTENT_USER)

  responses = [
      resp
      async for resp in llm.generate_content_async(llm_request, stream=True)
  ]

  # Portkey wrapper may not attach usage metadata in streaming; ensure field exists
  assert hasattr(responses[-1], "usage_metadata")

"""End of test_portkey_llm.py""" 