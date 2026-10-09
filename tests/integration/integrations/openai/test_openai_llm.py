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

"""Integration tests for the OpenAI models against live backends.

Each test runs against a matrix of ``_Case``s — one per (model, API surface)
combination that is configured via environment variables. Two model classes are
exercised:

* ``OpenAILlm`` — the Chat Completions API.
* ``OpenAIResponsesLlm`` — the Responses API.

Configured matrix (each leg is skipped if its creds are absent):

* OpenAI (needs ``OPENAI_API_KEY``), on BOTH Chat Completions and Responses:
  ``gpt-6-astra``, ``gpt-5.6-sol``, ``gpt-5.6-terra``, ``gpt-5.6-luna`` — all
  reasoning models.
* xAI Grok 4.6 (needs ``GROK_VERTEX_PROJECT`` + ADC), Chat Completions only:
  ``xai/grok-4.6`` served on Vertex AI Model Garden. Its ``api_key`` is a Google
  Cloud access token supplied as a *callable* so the ~1h token is refreshed per
  request.

Live-verified backend quirks that shape the cases (as of 2026-09):
  - Reasoning models reject ``max_tokens`` (require ``max_completion_tokens``)
    and reject a non-default ``temperature`` / ``top_p`` on both APIs. The
    wrapper now normalizes this; these tests prove it end to end.
  - Function tools are NOT available on Chat Completions for the reasoning
    models (gpt-6-astra has no chat tool calling; the gpt-5.6 family needs
    ``reasoning_effort='none'``). Tool calling for them is validated on the
    Responses leg instead. Grok supports tools on Chat Completions.
  - The ``stop`` parameter is rejected by every model in this matrix: Grok on
    Vertex (400 'does not support parameter stop'), the reasoning models on Chat
    Completions (400 'stop is not supported'), and the Responses API (no ``stop``
    parameter at all). So ``test_stop_sequences`` asserts a clean 400 everywhere.
  - Reasoning models keep the usage invariant total == prompt + completion
    (reasoning tokens are inside the completion/output count). Grok counts
    hidden reasoning tokens only in the total, so total >= there.

Environment variables:
  OPENAI_API_KEY       — enables the OpenAI legs (both surfaces).
  OPENAI_MODELS        — comma-separated model-id override for the OpenAI legs
                         (default: gpt-6-astra,gpt-5.6-sol,gpt-5.6-terra,
                         gpt-5.6-luna).
  OPENAI_BASE_URL      — override the OpenAI base URL (default: SDK default).
  GROK_VERTEX_PROJECT  — GCP project id; enables the Grok leg (+ ADC).
  GROK_MODEL           — grok model id (default: ``xai/grok-4.6``).
  GROK_BASE_URL        — override the grok base URL.
  GOOGLE_API_USE_MTLS_ENDPOINT — selects the Grok host when GROK_BASE_URL is
                         unset. ``auto`` (default) uses the regular
                         ``aiplatform.googleapis.com`` host unless a client
                         certificate is available; ``always`` forces the mTLS
                         host, which needs a client that presents a
                         certificate; ``never`` forces the regular host.

Run (via plain pytest):
  OPENAI_API_KEY=<key> GROK_VERTEX_PROJECT=<project> \\
    python -m pytest \\
    tests/integration/integrations/openai/test_openai_llm.py -v
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Callable
from typing import Optional
from typing import Type
from typing import Union

from google.adk.integrations.openai._openai_common import is_reasoning_model
from google.adk.integrations.openai._openai_llm import OpenAILlm
from google.adk.integrations.openai._openai_responses_llm import OpenAIResponsesLlm
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.utils._mtls_utils import get_api_endpoint
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import openai
import pytest

# ---------------------------------------------------------------------------
# Backend neutralization
# ---------------------------------------------------------------------------


# These targets use no Google-genai backend toggle. The autouse ``llm_backend``
# fixture from tests/integration/conftest.py both mutates
# GOOGLE_GENAI_USE_ENTERPRISE and is parametrized across GOOGLE_AI / VERTEX by
# that conftest's ``pytest_generate_tests``. Overriding the fixture here stops
# the env mutation, but the name stays in ``fixturenames``, so the
# parametrization alone would still run every test twice. ``_NEUTRAL_BACKEND``
# (applied to every test below) pins ``llm_backend`` to a single value, which
# the conftest treats as an explicit parametrize and skips its own -- so each
# test runs exactly once.
@pytest.fixture(autouse=True)
def llm_backend():
  yield


# Applied to every test so the conftest's GOOGLE_AI / VERTEX parametrization of
# ``llm_backend`` collapses to a single, backend-neutral run.
_NEUTRAL_BACKEND = pytest.mark.parametrize(
    "llm_backend", [None], indirect=True, ids=[""]
)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def _access_token() -> str:
  """Returns a fresh Google Cloud access token from ADC."""
  import google.auth
  import google.auth.transport.requests

  credentials, _ = google.auth.default(
      scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )
  credentials.refresh(google.auth.transport.requests.Request())
  return credentials.token


# ---------------------------------------------------------------------------
# Case definitions + discovery
# ---------------------------------------------------------------------------

# A generous default output budget. Reasoning models spend part of the budget on
# hidden reasoning tokens, so keep it comfortably above what the short test
# prompts need for a visible answer.
_DEFAULT_MAX_TOKENS = 1024

# Budget used by the truncation tests. Small enough to force a ``length`` finish
# on a deliberately over-long prompt, but large enough that a reasoning model's
# hidden reasoning pass can consume it (tiny budgets make reasoning models 400
# with 'max_tokens ... reached' instead of returning a clean truncation).
_TRUNCATION_BUDGET = 512


@dataclasses.dataclass(frozen=True)
class _Case:
  """One live (model, API surface) target plus its observable quirks."""

  id: str
  llm_class: Type[BaseLlm]
  model: str
  base_url: Optional[str]
  # api_key value: a callable (grok, refreshing token) or a string (openai).
  # repr=False keeps the key out of the dataclass repr, which pytest would
  # otherwise print (with the raw key) in a parametrized-case failure traceback.
  key: Union[str, Callable[[], str]] = dataclasses.field(repr=False)
  # Whether the backend accepts function tools on this surface.
  supports_tools: bool
  # Whether the backend accepts the ``stop`` param (false for every default case
  # -- see module docstring; true for a non-reasoning OPENAI_MODELS override on
  # Chat Completions).
  supports_stop: bool
  # Whether total == prompt + candidates (reasoning tokens are inside the
  # completion count). Grok counts them only in the total, so it is >= there.
  usage_exact: bool

  def make_llm(self, **overrides) -> BaseLlm:
    kwargs: dict[str, object] = {"model": self.model, "api_key": self.key}
    if self.llm_class is OpenAILlm:
      # Chat Completions has a constructor token budget; Responses caps via
      # config.max_output_tokens only, so do not pass it there.
      kwargs["max_tokens"] = _DEFAULT_MAX_TOKENS
    # Both surfaces accept base_url (Grok on Vertex, or a custom OpenAI host);
    # OpenAIResponsesLlm honors it too, so forward it for either llm_class.
    if self.base_url:
      kwargs["base_url"] = self.base_url
    kwargs.update(overrides)
    return self.llm_class(**kwargs)

  def raw_key(self) -> str:
    return self.key() if callable(self.key) else self.key


def _discover_cases() -> list[_Case]:
  cases: list[_Case] = []

  openai_key = os.environ.get("OPENAI_API_KEY")
  if openai_key:
    default_models = "gpt-6-astra,gpt-5.6-sol,gpt-5.6-terra,gpt-5.6-luna"
    models = [
        m.strip()
        for m in os.environ.get("OPENAI_MODELS", default_models).split(",")
        if m.strip()
    ]
    for model in models:
      short = model.split("/")[-1]
      # Chat Completions leg: reasoning models cannot do tools here.
      cases.append(
          _Case(
              id=f"chat:{short}",
              llm_class=OpenAILlm,
              model=model,
              base_url=os.environ.get("OPENAI_BASE_URL"),
              key=openai_key,
              supports_tools=False,
              # Reasoning models reject ``stop`` on Chat Completions; a
              # non-reasoning override (e.g. OPENAI_MODELS=gpt-4o) accepts it.
              supports_stop=not is_reasoning_model(model),
              usage_exact=True,
          )
      )
      # Responses leg: tools work here.
      cases.append(
          _Case(
              id=f"resp:{short}",
              llm_class=OpenAIResponsesLlm,
              model=model,
              base_url=os.environ.get("OPENAI_BASE_URL"),
              key=openai_key,
              supports_tools=True,
              # The Responses API has no ``stop`` parameter for any model.
              supports_stop=False,
              usage_exact=True,
          )
      )

  project = os.environ.get("GROK_VERTEX_PROJECT")
  if project:
    base_url = os.environ.get("GROK_BASE_URL") or get_api_endpoint(
        location="global",
        default_template=(
            f"https://aiplatform.googleapis.com/v1/projects/{project}"
            "/locations/global/endpoints/openapi"
        ),
        mtls_template=(
            f"https://aiplatform.mtls.googleapis.com/v1/projects/{project}"
            "/locations/global/endpoints/openapi"
        ),
    )
    cases.append(
        _Case(
            id="chat:grok-4.6",
            llm_class=OpenAILlm,
            model=os.environ.get("GROK_MODEL", "xai/grok-4.6"),
            base_url=base_url,
            key=_access_token,
            supports_tools=True,
            supports_stop=False,
            usage_exact=False,
        )
    )

  return cases


_CASES = _discover_cases()

pytestmark = pytest.mark.skipif(
    not _CASES,
    reason=(
        "OpenAI live tests require OPENAI_API_KEY and/or GROK_VERTEX_PROJECT"
        " (+ ADC). Set at least one to run."
    ),
)


@pytest.fixture(params=_CASES or [None], ids=lambda c: c.id if c else "none")
def case(request) -> _Case:
  return request.param


@pytest.fixture
def llm(case: _Case) -> BaseLlm:
  return case.make_llm()


class _CountingKeyProvider:
  """A callable api_key that resolves the case key and counts calls."""

  def __init__(self, case: _Case):
    self._case = case
    self.count = 0

  def __call__(self) -> str:
    self.count += 1
    return self._case.raw_key()


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def _simple_request(
    model: str, text: str = "Reply with one word: hello."
) -> LlmRequest:
  return LlmRequest(
      model=model,
      contents=[Content(role="user", parts=[Part.from_text(text=text)])],
  )


def _weather_tool() -> types.Tool:
  return types.Tool(
      function_declarations=[
          types.FunctionDeclaration(
              name="get_weather",
              description="Get the current weather for a city.",
              parameters=types.Schema(
                  type=types.Type.OBJECT,
                  properties={
                      "city": types.Schema(
                          type=types.Type.STRING,
                          description="The city name.",
                      )
                  },
                  required=["city"],
              ),
          )
      ]
  )


def _request_with_tool(
    model: str,
    text: str = "What is the weather in Chicago?",
    tool_config: Optional[types.ToolConfig] = None,
) -> LlmRequest:
  return LlmRequest(
      model=model,
      contents=[Content(role="user", parts=[Part.from_text(text=text)])],
      config=types.GenerateContentConfig(
          tools=[_weather_tool()],
          tool_config=tool_config,
      ),
  )


def _answer_text(response) -> str:
  """Concatenates the visible answer text, skipping reasoning/thought parts.

  Reasoning models on the Responses API emit a leading ``thought=True`` part
  (with ``text=None``) before the answer, so ``parts[0].text`` is not reliable.
  """
  parts = response.content.parts if response.content else []
  return "".join(
      p.text for p in parts if p.text and not getattr(p, "thought", False)
  )


def _function_calls(response) -> list[types.FunctionCall]:
  """Returns the function-call parts of ``response`` (empty if no content)."""
  parts = (response.content.parts if response.content else None) or []
  return [p.function_call for p in parts if p.function_call]


def _final_response(responses):
  """Returns the first non-partial (aggregated) streaming response."""
  final = next((r for r in responses if not r.partial), None)
  assert final is not None, "expected a non-partial final streaming response"
  return final


def _skip_if_no_tools(case: _Case) -> None:
  if not case.supports_tools:
    pytest.skip(
        f"{case.id}: function tools are not supported on this surface for this"
        " model (validated on the Responses leg instead)"
    )


# ---------------------------------------------------------------------------
# Live behavior of the OpenAI-compatible surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_callable_api_key_reinvoked_per_request(case):
  """A callable api_key is re-resolved on every request.

  Runs two requests through one model instance and asserts the provider's
  invocation count strictly increased between them — i.e. the per-request
  client refresh fires, rather than the key being frozen at construction time.
  """
  key_provider = _CountingKeyProvider(case)
  llm = case.make_llm(api_key=key_provider)

  _ = [r async for r in llm.generate_content_async(_simple_request(case.model))]
  after_first = key_provider.count
  _ = [r async for r in llm.generate_content_async(_simple_request(case.model))]
  after_second = key_provider.count

  assert after_first >= 1
  assert (
      after_second > after_first
  ), "callable api_key was not re-invoked on the second request"


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_finish_reason_stop(llm, case):
  """A normally completed response maps to FinishReason.STOP."""
  responses = [
      r
      async for r in llm.generate_content_async(
          _simple_request(case.model), stream=False
      )
  ]
  assert responses[0].finish_reason == types.FinishReason.STOP


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_finish_reason_max_tokens(case):
  """A truncated response maps to FinishReason.MAX_TOKENS.

  Uses a budget of ``_TRUNCATION_BUDGET`` rather than a tiny value: reasoning
  models raise a 400 ('max_tokens ... reached') when the budget is too small to
  finish even the hidden reasoning pass, but return a clean ``length`` finish
  once the budget is large enough for reasoning to consume it. The prompt forces
  an over-long output so the budget is always exceeded.
  """
  llm = case.make_llm(**(
      {"max_tokens": _TRUNCATION_BUDGET} if case.llm_class is OpenAILlm else {}
  ))
  request = LlmRequest(
      model=case.model,
      contents=[
          Content(
              role="user",
              parts=[
                  Part.from_text(
                      text=(
                          "Write a very long, detailed 2000-word essay about"
                          " the ocean. Do not stop early."
                      )
                  )
              ],
          )
      ],
      config=types.GenerateContentConfig(max_output_tokens=_TRUNCATION_BUDGET),
  )
  responses = [r async for r in llm.generate_content_async(request)]
  assert responses[0].finish_reason == types.FinishReason.MAX_TOKENS


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_system_instruction_string(llm, case):
  """A plain-string system instruction is honored."""
  request = LlmRequest(
      model=case.model,
      contents=[
          Content(
              role="user", parts=[Part.from_text(text="What is your name?")]
          )
      ],
      config=types.GenerateContentConfig(
          system_instruction=(
              "Your name is Nova. Always introduce yourself as Nova."
          )
      ),
  )
  responses = [r async for r in llm.generate_content_async(request)]
  assert "nova" in _answer_text(responses[0]).lower()


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_system_instruction_content_shape(llm, case):
  """A Content-shaped system instruction is flattened and honored."""
  request = LlmRequest(
      model=case.model,
      contents=[
          Content(
              role="user", parts=[Part.from_text(text="What is your name?")]
          )
      ],
      config=types.GenerateContentConfig(
          system_instruction=Content(
              parts=[
                  Part.from_text(text="Your name is Nova."),
                  Part.from_text(text="Always introduce yourself as Nova."),
              ]
          )
      ),
  )
  responses = [r async for r in llm.generate_content_async(request)]
  assert "nova" in _answer_text(responses[0]).lower()


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_usage_metadata_populated(llm, case):
  """Non-streaming responses carry token usage metadata."""
  responses = [
      r
      async for r in llm.generate_content_async(
          _simple_request(case.model), stream=False
      )
  ]
  usage = responses[0].usage_metadata
  assert usage is not None
  assert usage.prompt_token_count > 0
  assert usage.candidates_token_count > 0
  if is_reasoning_model(case.model):
    # OpenAI reports reasoning tokens on both surfaces; they are surfaced as
    # thoughts_token_count (and are also included in candidates_token_count).
    assert usage.thoughts_token_count is not None
    assert 0 <= usage.thoughts_token_count <= usage.candidates_token_count
  if case.usage_exact:
    assert usage.total_token_count == (
        usage.prompt_token_count + usage.candidates_token_count
    )
  else:
    # Grok 4.6 counts hidden reasoning tokens only in the total.
    assert usage.total_token_count >= (
        usage.prompt_token_count + usage.candidates_token_count
    )


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_streaming_usage_metadata(llm, case):
  """The final streaming chunk carries usage."""
  responses = [
      r
      async for r in llm.generate_content_async(
          _simple_request(case.model), stream=True
      )
  ]
  final = _final_response(responses)
  usage = final.usage_metadata
  assert usage is not None
  assert usage.prompt_token_count > 0
  assert usage.candidates_token_count > 0


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_bad_api_key_raises(case):
  """An invalid key surfaces a clean OpenAI auth error, not a crash."""
  llm = case.make_llm(api_key="definitely-not-a-valid-token")
  with pytest.raises((
      openai.AuthenticationError,
      openai.PermissionDeniedError,
  )):
    _ = [
        r async for r in llm.generate_content_async(_simple_request(case.model))
    ]


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_non_default_temperature_and_top_p_normalized(llm, case):
  """Non-default temperature / top_p complete instead of 400-ing.

  Reasoning models reject a non-default ``temperature`` / ``top_p``; the wrapper
  strips them before the call, so the request finishes normally rather than
  raising a 400 -- this proves that normalization end to end. Backends that do
  accept the params (e.g. Grok) simply honor them, so a clean ``STOP`` is the
  expected result across the whole matrix.
  """
  request = LlmRequest(
      model=case.model,
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="Reply with one word: hello.")],
          )
      ],
      config=types.GenerateContentConfig(temperature=0.5, top_p=0.9),
  )
  responses = [
      r async for r in llm.generate_content_async(request, stream=False)
  ]
  assert responses[0].finish_reason == types.FinishReason.STOP


# ---------------------------------------------------------------------------
# SHOULD — untested code paths the backend supports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_tool_call(llm, case):
  """A single tool call is returned as a function-call part."""
  _skip_if_no_tools(case)
  responses = [
      r
      async for r in llm.generate_content_async(
          _request_with_tool(case.model), stream=False
      )
  ]
  calls = _function_calls(responses[0])
  assert calls, "expected a function call"
  assert calls[0].name == "get_weather"
  assert "city" in calls[0].args


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_streaming_tool_call(llm, case):
  """A tool call is assembled from streaming deltas into the final response."""
  _skip_if_no_tools(case)
  responses = [
      r
      async for r in llm.generate_content_async(
          _request_with_tool(case.model), stream=True
      )
  ]
  final = _final_response(responses)
  calls = _function_calls(final)
  assert calls, "expected a function call in the final streaming response"
  assert calls[0].name == "get_weather"
  assert "city" in calls[0].args


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_parallel_tool_calls(llm, case):
  """Multiple tool calls in one turn are all surfaced."""
  _skip_if_no_tools(case)
  request = _request_with_tool(
      case.model,
      text=(
          "Get the current weather in Paris and in Tokyo. Call the get_weather"
          " tool once for each city."
      ),
  )
  responses = [r async for r in llm.generate_content_async(request)]
  calls = _function_calls(responses[0])
  assert len(calls) >= 2, f"expected >= 2 parallel tool calls, got {len(calls)}"
  cities = {str(c.args.get("city", "")).lower() for c in calls}
  assert "paris" in cities and "tokyo" in cities


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_forced_tool_choice_any(llm, case):
  """tool_config mode ANY forces a tool call even for a chatty prompt."""
  _skip_if_no_tools(case)
  request = _request_with_tool(
      case.model,
      text="Say hello.",
      tool_config=types.ToolConfig(
          function_calling_config=types.FunctionCallingConfig(
              mode=types.FunctionCallingConfigMode.ANY
          )
      ),
  )
  responses = [r async for r in llm.generate_content_async(request)]
  calls = _function_calls(responses[0])
  assert calls, "mode=ANY should force a tool call"


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_tool_choice_none(llm, case):
  """tool_config mode NONE suppresses tool calls; a text answer is returned."""
  _skip_if_no_tools(case)
  request = _request_with_tool(
      case.model,
      text="What is the weather in Chicago?",
      tool_config=types.ToolConfig(
          function_calling_config=types.FunctionCallingConfig(
              mode=types.FunctionCallingConfigMode.NONE
          )
      ),
  )
  responses = [r async for r in llm.generate_content_async(request)]
  parts = responses[0].content.parts if responses[0].content else []
  assert not [
      p for p in parts if p.function_call
  ], "mode=NONE should suppress tool calls"
  assert any(p.text for p in parts), "expected a text answer instead"


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_stop_sequences(llm, case):
  """stop_sequences map to the ``stop`` param.

  Every model in the default matrix rejects ``stop`` (Grok-on-Vertex, the
  reasoning models on Chat Completions, and the Responses API which has no
  ``stop`` parameter), so a clean 400 is the expected result. A case with
  ``supports_stop=True`` (a non-reasoning ``OPENAI_MODELS`` override on Chat
  Completions) asserts the output is truncated instead.
  """
  request = LlmRequest(
      model=case.model,
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="Print exactly: APPLE BANANA CHERRY")],
          )
      ],
      config=types.GenerateContentConfig(stop_sequences=["BANANA"]),
  )
  if not case.supports_stop:
    with pytest.raises(openai.BadRequestError):
      _ = [r async for r in llm.generate_content_async(request)]
    return
  responses = [r async for r in llm.generate_content_async(request)]
  text = _answer_text(responses[0])
  assert "BANANA" not in text


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_max_output_tokens_caps_response(case):
  """max_output_tokens caps the reported completion token count.

  Uses ``_TRUNCATION_BUDGET`` for the same reason as the finish-reason test: a
  tiny budget makes reasoning models 400 instead of returning usage. The prompt
  forces an over-long output so the budget is genuinely exercised as a cap.
  """
  budget = _TRUNCATION_BUDGET
  llm = case.make_llm(
      **({"max_tokens": budget} if case.llm_class is OpenAILlm else {})
  )
  request = LlmRequest(
      model=case.model,
      contents=[
          Content(
              role="user",
              parts=[
                  Part.from_text(
                      text=(
                          "Write a very long, detailed 2000-word essay about"
                          " the ocean. Do not stop early."
                      )
                  )
              ],
          )
      ],
      config=types.GenerateContentConfig(max_output_tokens=budget),
  )
  responses = [r async for r in llm.generate_content_async(request)]
  usage = responses[0].usage_metadata
  assert usage is not None
  assert usage.candidates_token_count is not None
  assert usage.candidates_token_count <= budget


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_multi_turn(llm, case):
  """Conversation history is passed through and used."""
  history = [
      Content(
          role="user",
          parts=[Part.from_text(text="My favourite colour is blue.")],
      ),
      Content(
          role="model",
          parts=[Part.from_text(text="Got it, blue is a great colour!")],
      ),
  ]
  follow_up = Content(
      role="user", parts=[Part.from_text(text="What is my favourite colour?")]
  )
  request = LlmRequest(model=case.model, contents=history + [follow_up])
  responses = [r async for r in llm.generate_content_async(request)]
  assert "blue" in _answer_text(responses[0]).lower()


# ---------------------------------------------------------------------------
# VERIFY — backend-dependent features
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_response_schema_json_schema(llm, case):
  """response_schema drives strict json_schema structured output."""
  schema = {
      "title": "CityFact",
      "type": "object",
      "properties": {
          "city": {"type": "string"},
          "country": {"type": "string"},
      },
      "required": ["city", "country"],
      "additionalProperties": False,
  }
  request = LlmRequest(
      model=case.model,
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="Give me a fact about Paris.")],
          )
      ],
      config=types.GenerateContentConfig(response_schema=schema),
  )
  responses = [r async for r in llm.generate_content_async(request)]
  payload = json.loads(_answer_text(responses[0]))
  assert "city" in payload and "country" in payload


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_response_mime_type_json_object(llm, case):
  """response_mime_type=application/json drives json_object mode."""
  request = LlmRequest(
      model=case.model,
      contents=[
          Content(
              role="user",
              parts=[
                  Part.from_text(
                      text=(
                          "Return a JSON object with keys 'city' and 'country'"
                          " for Paris."
                      )
                  )
              ],
          )
      ],
      config=types.GenerateContentConfig(response_mime_type="application/json"),
  )
  responses = [r async for r in llm.generate_content_async(request)]
  payload = json.loads(_answer_text(responses[0]))
  assert isinstance(payload, dict)


def _make_red_png(size: int = 128) -> bytes:
  """Generate a guaranteed-valid size x size solid-red PNG with correct CRCs.

  Grok requires each dimension to be >= 8 pixels and >= 512 total pixels; any
  size above that clears its floor. The default is 128x128 because gpt-6-astra's
  vision misreads a tiny 32x32 solid-red block (returns "peach"/"beige"), while
  128x128 is read as "red" reliably by every model in the matrix. OpenAI has no
  minimum-size requirement.
  """
  import struct
  import zlib

  sig = b"\x89PNG\r\n\x1a\n"

  def chunk(t: bytes, d: bytes) -> bytes:
    return (
        struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    )

  ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # RGB, 8-bit
  row = b"\x00" + b"\xff\x00\x00" * size  # filter byte + red pixels
  idat = zlib.compress(row * size)
  return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_inline_image_input(llm, case):
  """An inline image is sent as an image_url data URL and understood."""
  request = LlmRequest(
      model=case.model,
      contents=[
          Content(
              role="user",
              parts=[
                  Part.from_text(
                      text=(
                          "What is the dominant colour of this image? Reply"
                          " with just the colour name."
                      )
                  ),
                  Part(
                      inline_data=types.Blob(
                          mime_type="image/png", data=_make_red_png()
                      )
                  ),
              ],
          )
      ],
      config=types.GenerateContentConfig(max_output_tokens=1024),
  )
  responses = [r async for r in llm.generate_content_async(request)]
  assert "red" in _answer_text(responses[0]).lower()
