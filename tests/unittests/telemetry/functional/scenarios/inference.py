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

"""The model a scenario runs, and the instrumentation watching it.

``inference_under_test`` hands out the model to run with, its instrumentation
already active: a ``MockModel`` ADK instruments itself, or a real ``Gemini``
over a mocked-out SDK wrapped by opentelemetry-instrumentation-google-genai.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Iterator
from contextlib import contextmanager
import copy

from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from google.genai.models import AsyncModels
from google.genai.types import Candidate
from google.genai.types import Content
from google.genai.types import FinishReason
from google.genai.types import GenerateContentResponse
from opentelemetry.instrumentation._semconv import _OpenTelemetrySemanticConventionStability
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
import pytest
from typing_extensions import assert_never

from ....testing_utils import MockModel
from .._divergences import InferenceInstrumentation
from .conversation import MODEL_NAME
from .conversation import StreamedTurn
from .conversation import TOOL_CALLING_TURNS
from .conversation import Turn
from .telemetry_setup import TelemetryProviders


def mock_test_model(
    *,
    turns: tuple[Turn, ...] = TOOL_CALLING_TURNS,
    model_exception: Exception | None = None,
) -> MockModel:
  """The canned conversation as a ``MockModel``, for the ADK-native path.

  With ``model_exception`` the model raises instead of responding: leave the
  responses empty so the mock never yields.
  """
  model = MockModel.create(
      responses=(
          []
          if model_exception is not None
          else [
              LlmResponse(
                  content=Content(role="model", parts=[copy.deepcopy(part)]),
                  finish_reason=FinishReason.STOP,
                  usage_metadata=usage,
              )
              for part, usage in turns
          ]
      ),
      error=model_exception,
  )
  model.model = MODEL_NAME
  return model


def gemini_test_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    turns: tuple[Turn, ...] = TOOL_CALLING_TURNS,
    model_exception: Exception | None = None,
) -> Gemini:
  """The canned conversation as a real ``Gemini`` over a mocked-out SDK.

  ``AsyncModels.generate_content`` returns the canned responses instead of
  calling the API, so the model is real, the SDK call path is real, and no
  request leaves the process.

  With ``model_exception`` the SDK raises it instead of responding,
  exercising the inference-failure telemetry path.
  """
  responses = iter([
      GenerateContentResponse(
          candidates=[
              Candidate(
                  content=Content(role="model", parts=[copy.deepcopy(part)]),
                  finish_reason=FinishReason.STOP,
              )
          ],
          usage_metadata=usage,
      )
      for part, usage in turns
  ])

  async def mock_generate_content(
      self: AsyncModels, **kwargs: object
  ) -> GenerateContentResponse:
    # The canned responses don't depend on the request; the request is
    # asserted through the telemetry the instrumentor derives from it.
    del self, kwargs
    if model_exception is not None:
      raise model_exception
    return next(responses)

  monkeypatch.setattr(AsyncModels, "generate_content", mock_generate_content)

  # ``Gemini`` builds a real ``google.genai.Client``, which opens no
  # connection -- but without a key it would look for application default
  # credentials, so pin one to keep the test off the developer's environment.
  monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key-for-tests")

  return Gemini(model=MODEL_NAME)


def streaming_gemini_test_model(
    monkeypatch: pytest.MonkeyPatch,
    turns: tuple[StreamedTurn, ...],
) -> Gemini:
  """The streamed conversation as a real `Gemini` over a mocked-out SDK.

  Mocks the streaming SDK entrypoint, so it is the one the instrumentor wraps
  and the one the chunks are counted through. Only the chunk that ends a turn
  reports why generation stopped; the ones before it leave the field at its
  proto3 zero value, as a real stream does.
  """
  streamed = iter(turns)

  async def mock_generate_content_stream(
      self: AsyncModels, **kwargs: object
  ) -> AsyncGenerator[GenerateContentResponse, None]:
    del self, kwargs

    async def chunks() -> AsyncGenerator[GenerateContentResponse, None]:
      turn = next(streamed)
      for index, (part, usage) in enumerate(turn):
        yield GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[copy.deepcopy(part)]),
                    finish_reason=(
                        FinishReason.STOP
                        if index == len(turn) - 1
                        else FinishReason.FINISH_REASON_UNSPECIFIED
                    ),
                )
            ],
            usage_metadata=usage,
        )

    return chunks()

  monkeypatch.setattr(
      AsyncModels, "generate_content_stream", mock_generate_content_stream
  )
  monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key-for-tests")

  return Gemini(model=MODEL_NAME)


@contextmanager
def otel_instrumentor(
    monkeypatch: pytest.MonkeyPatch, providers: TelemetryProviders
) -> Iterator[None]:
  """Runs opentelemetry-instrumentation-google-genai over the SDK, for a while.

  Whatever it is to wrap has to be in place before this: it patches
  ``google.genai`` on the way in and restores what it found on the way out.
  """
  # PRIVATE: the instrumentation libraries resolve OTEL_SEMCONV_STABILITY_OPT_IN
  # once per process and cache it here. Reset that cache so the instrumentor
  # reads THIS case's env vars rather than whichever case ran first. See
  # ``test_semconv_stability_cache_can_be_reset``.
  monkeypatch.setattr(
      _OpenTelemetrySemanticConventionStability, "_initialized", False
  )
  monkeypatch.setattr(
      _OpenTelemetrySemanticConventionStability,
      "_OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING",
      {},
  )

  instrumentor = GoogleGenAiSdkInstrumentor()
  instrumentor.instrument(
      tracer_provider=providers.tracer_provider,
      logger_provider=providers.logger_provider,
      meter_provider=providers.meter_provider,
  )
  try:
    yield
  finally:
    instrumentor.uninstrument()


@contextmanager
def inference_under_test(
    instrumentation: InferenceInstrumentation,
    monkeypatch: pytest.MonkeyPatch,
    providers: TelemetryProviders,
    *,
    turns: tuple[Turn, ...] = TOOL_CALLING_TURNS,
    streamed_turns: tuple[StreamedTurn, ...] | None = None,
    model_exception: Exception | None = None,
) -> Iterator[BaseLlm]:
  """Yields the model to run a scenario with, its instrumentation active.

  Both come from here, so a scenario cannot end up running one
  instrumentation's model under the other's instrumentation.

  ``native`` yields a ``MockModel`` that never touches ``google.genai``, and
  ADK instruments it.

  ``otel`` yields a ``Gemini`` over the mocked-out SDK, with the real
  instrumentor wrapping it -- mocked FIRST so that what the instrumentor
  wraps is the mock. ADK sees the wrapped SDK and stands down for a Gemini
  agent, so the inference telemetry recorded is entirely OTel's.

  ``streamed_turns`` replaces ``turns`` with a conversation delivered in
  chunks. Both sides then run the same mocked-out SDK, so that what the
  recordings differ over is still only the instrumentation.
  """
  assert (
      streamed_turns is None or model_exception is None
  ), "No streamed failure case yet; the streaming model would swallow it."
  if instrumentation == "native":
    yield (
        streaming_gemini_test_model(monkeypatch, streamed_turns)
        if streamed_turns is not None
        else mock_test_model(turns=turns, model_exception=model_exception)
    )
  elif instrumentation == "otel":
    model = (
        streaming_gemini_test_model(monkeypatch, streamed_turns)
        if streamed_turns is not None
        else gemini_test_model(
            monkeypatch, turns=turns, model_exception=model_exception
        )
    )
    with otel_instrumentor(monkeypatch, providers):
      yield model
  else:
    assert_never(instrumentation)
