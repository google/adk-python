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

"""Live integration tests for OpenAI reasoning-effort tiers.

For every configured OpenAI reasoning model, this suite drives the model on
BOTH API surfaces (Chat Completions via ``OpenAILlm`` and Responses via
``OpenAIResponsesLlm``) across EVERY effort tier the model supports, using
``OpenAIGenerateContentConfig(effort=...)``. The set of tiers per model comes
from ``_openai_common.supported_efforts`` -- the same table the wrappers use to
validate requests -- so a live 400 here means the table is wrong and must be
corrected.

Each leg asserts the request is accepted (no error_code) and either returns a
visible answer (finish_reason STOP) or hits the output budget (MAX_TOKENS);
higher tiers spend more of the budget on hidden reasoning tokens.

Environment variables:
  OPENAI_API_KEY   — enables the suite (required).
  OPENAI_MODELS    — comma-separated model-id override
                     (default: gpt-6-astra,gpt-5.6-sol,gpt-5.6-terra,
                     gpt-5.6-luna).
  OPENAI_BASE_URL  — override the OpenAI base URL (default: SDK default).

Run (plain pytest):
  OPENAI_API_KEY=<key> python -m pytest \\
    tests/integration/integrations/openai/test_openai_reasoning.py -v
"""

from __future__ import annotations

import dataclasses
import os

from google.adk.integrations.openai import OpenAIGenerateContentConfig
from google.adk.integrations.openai import OpenAILlm
from google.adk.integrations.openai import OpenAIResponsesLlm
from google.adk.integrations.openai._openai_common import supported_efforts
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest


# The integration conftest's autouse ``llm_backend`` fixture both mutates
# GOOGLE_GENAI_USE_ENTERPRISE and is parametrized across GOOGLE_AI / VERTEX by
# its ``pytest_generate_tests``. Overriding the fixture stops the env mutation,
# but the name stays in ``fixturenames``, so the parametrization alone would
# still double every case (and every live call). ``_NEUTRAL_BACKEND`` (applied
# to every test) pins ``llm_backend`` to a single value, which the conftest
# treats as an explicit parametrize and skips its own -- so each case runs once.
@pytest.fixture(autouse=True)
def llm_backend():
  yield


_NEUTRAL_BACKEND = pytest.mark.parametrize(
    "llm_backend", [None], indirect=True, ids=[""]
)


# Reasoning tiers can burn a large chunk of the output budget on hidden
# reasoning tokens (``max`` most of all), so keep the answer budget generous to
# leave room for a visible reply.
_MAX_OUTPUT_TOKENS = 4096

_DEFAULT_MODELS = "gpt-6-astra,gpt-5.6-sol,gpt-5.6-terra,gpt-5.6-luna"


@dataclasses.dataclass(frozen=True)
class _EffortCase:
  """One (model, API surface, effort tier) live target."""

  id: str
  llm_class: type[BaseLlm]
  model: str
  base_url: str | None
  # repr=False keeps the key out of the dataclass repr, which pytest would
  # otherwise print (with the raw key) in a parametrized-case failure traceback.
  key: str = dataclasses.field(repr=False)
  effort: str

  def make_llm(self) -> BaseLlm:
    # The output budget is applied per request via ``config.max_output_tokens``
    # in ``_request`` (for both surfaces), which overrides the Chat Completions
    # constructor ``max_tokens``, so there is nothing to set here.
    kwargs: dict[str, object] = {"model": self.model, "api_key": self.key}
    if self.base_url:
      kwargs["base_url"] = self.base_url
    return self.llm_class(**kwargs)


def _discover_cases() -> list[_EffortCase]:
  key = os.environ.get("OPENAI_API_KEY")
  if not key:
    return []
  base_url = os.environ.get("OPENAI_BASE_URL")
  models = [
      m.strip()
      for m in os.environ.get("OPENAI_MODELS", _DEFAULT_MODELS).split(",")
      if m.strip()
  ]
  cases: list[_EffortCase] = []
  for model in models:
    short = model.split("/")[-1]
    # The accepted tier set differs per surface (e.g. ``max`` is Responses-only
    # for the advanced models), so query supported_efforts per API.
    for surface, api, cls in (
        ("chat", "chat", OpenAILlm),
        ("resp", "responses", OpenAIResponsesLlm),
    ):
      for effort in sorted(supported_efforts(model, api)):
        cases.append(
            _EffortCase(
                id=f"{surface}:{short}:{effort}",
                llm_class=cls,
                model=model,
                base_url=base_url,
                key=key,
                effort=effort,
            )
        )
  return cases


_CASES = _discover_cases()

pytestmark = pytest.mark.skipif(
    not _CASES,
    reason=(
        "No OpenAI reasoning cases discovered. Set OPENAI_API_KEY, and point"
        " OPENAI_MODELS at model ids with known effort tiers --"
        " supported_efforts is empty for a non-OpenAI model id, so a custom"
        " OPENAI_BASE_URL run yields no cases even with the key set."
    ),
)


@pytest.fixture(params=_CASES or [None], ids=lambda c: c.id if c else "none")
def effort_case(request) -> _EffortCase:
  return request.param


def _request(model: str, effort: str) -> LlmRequest:
  return LlmRequest(
      model=model,
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="Reply with one word: hello.")],
          )
      ],
      config=OpenAIGenerateContentConfig(
          effort=effort, max_output_tokens=_MAX_OUTPUT_TOKENS
      ),
  )


def _answer_text(response) -> str:
  """Concatenates visible answer text, skipping reasoning/thought parts."""
  parts = response.content.parts if response.content else []
  return "".join(p.text for p in parts if p.text and not p.thought)


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_effort_tier_is_accepted(effort_case: _EffortCase):
  """Every supported tier is accepted live and yields a usable response."""
  llm = effort_case.make_llm()
  responses = [
      r
      async for r in llm.generate_content_async(
          _request(effort_case.model, effort_case.effort), stream=False
      )
  ]

  assert responses, f"{effort_case.id}: no response returned"
  final = responses[-1]

  # The tier was accepted (a rejected effort raises a 400 out of this call), and
  # it terminated normally: either a clean stop, or the output budget was
  # exhausted by reasoning tokens. Budget exhaustion with no visible parts is
  # surfaced as finish_reason MAX_TOKENS *with* error_code MAX_TOKENS, so only a
  # STOP is required to be error-free with visible text.
  assert final.finish_reason in (
      types.FinishReason.STOP,
      types.FinishReason.MAX_TOKENS,
  ), (
      f"{effort_case.id}: unexpected finish_reason {final.finish_reason}"
      f" (error {final.error_code}: {final.error_message})"
  )
  if final.finish_reason == types.FinishReason.STOP:
    assert final.error_code is None, (
        f"{effort_case.id}: STOP but error {final.error_code}:"
        f" {final.error_message}"
    )
    assert _answer_text(
        final
    ).strip(), f"{effort_case.id}: STOP but no visible answer text"


@pytest.mark.asyncio
@_NEUTRAL_BACKEND
async def test_effort_tier_streams(effort_case: _EffortCase):
  """Every supported tier is accepted on the streaming path too."""
  llm = effort_case.make_llm()
  responses = [
      r
      async for r in llm.generate_content_async(
          _request(effort_case.model, effort_case.effort), stream=True
      )
  ]

  assert responses, f"{effort_case.id}: streaming produced no events"

  # The Responses accumulator can emit a failure chunk mid-stream and still
  # close with a hardcoded STOP, so the whole stream -- not just the final
  # chunk -- must be error-free. Budget exhaustion is the one allowed error
  # (error_code MAX_TOKENS); any other error_code is a real failure.
  bad = [
      r
      for r in responses
      if r.error_code is not None
      and r.error_code != types.FinishReason.MAX_TOKENS
  ]
  assert not bad, (
      f"{effort_case.id}: stream carried an error:"
      f" {[(r.error_code, r.error_message) for r in bad]}"
  )

  final = responses[-1]

  # The tier was accepted on the streaming path (a rejected effort raises a
  # 400): the closing chunk carries a normal terminal finish reason.
  assert final.finish_reason in (
      types.FinishReason.STOP,
      types.FinishReason.MAX_TOKENS,
  ), (
      f"{effort_case.id}: unexpected finish_reason {final.finish_reason}"
      f" (error {final.error_code}: {final.error_message})"
  )
  if final.finish_reason == types.FinishReason.STOP:
    assert final.error_code is None, (
        f"{effort_case.id}: STOP but error {final.error_code}:"
        f" {final.error_message}"
    )
    # The closing chunk may be a usage-only event with no parts, so look for
    # visible answer text anywhere in the stream.
    assert any(
        _answer_text(r).strip() for r in responses
    ), f"{effort_case.id}: STOP but the stream carried no visible answer text"
