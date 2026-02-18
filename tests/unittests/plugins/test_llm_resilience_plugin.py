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

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from typing import Optional
from unittest import IsolatedAsyncioTestCase

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.plugins.llm_resilience_plugin import LlmResiliencePlugin
from google.genai import types

from ..testing_utils import create_invocation_context


class AlwaysFailModel(BaseLlm):
  model: str = "failing-model"

  @classmethod
  def supported_models(cls) -> list[str]:
    return ["failing-model"]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    # Always raise a timeout error to simulate transient failures
    raise asyncio.TimeoutError("Simulated timeout in AlwaysFailModel")


class SimpleSuccessModel(BaseLlm):
  model: str = "mock"

  @classmethod
  def supported_models(cls) -> list[str]:
    return ["mock"]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    # Return a single final response regardless of stream flag
    yield LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="final response from mock")],
        ),
        partial=False,
    )


class TestLlmResiliencePlugin(IsolatedAsyncioTestCase):

  @classmethod
  def setUpClass(cls):
    # Register test models in the registry once
    LLMRegistry.register(AlwaysFailModel)
    LLMRegistry.register(SimpleSuccessModel)

  async def test_retry_success_on_same_model(self):
    # Agent uses SimpleSuccessModel directly
    agent = LlmAgent(name="agent", model=SimpleSuccessModel())
    invocation_context = await create_invocation_context(agent)
    plugin = LlmResiliencePlugin(max_retries=2)

    # Build a minimal request
    llm_request = LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text="hi")])
        ]
    )

    # Simulate an initial transient error (e.g., 429/timeout)
    result = await plugin.on_model_error_callback(
        callback_context=invocation_context,
        llm_request=llm_request,
        error=asyncio.TimeoutError(),
    )

    self.assertIsNotNone(result)
    self.assertIsInstance(result, LlmResponse)
    self.assertFalse(result.partial)
    self.assertIsNotNone(result.content)
    self.assertEqual(
        result.content.parts[0].text.strip(), "final response from mock"
    )

  async def test_fallback_model_used_after_retries(self):
    # Agent starts with a failing string model; plugin will fallback to "mock"
    agent = LlmAgent(name="agent", model="failing-model")
    invocation_context = await create_invocation_context(agent)
    plugin = LlmResiliencePlugin(max_retries=1, fallback_models=["mock"])

    llm_request = LlmRequest(
        contents=[
            types.Content(
                role="user", parts=[types.Part.from_text(text="hello")]
            )
        ]
    )

    # Trigger resilience with a transient error
    result = await plugin.on_model_error_callback(
        callback_context=invocation_context,
        llm_request=llm_request,
        error=asyncio.TimeoutError(),
    )

    self.assertIsNotNone(result)
    self.assertIsInstance(result, LlmResponse)
    self.assertFalse(result.partial)
    self.assertEqual(
        result.content.parts[0].text.strip(), "final response from mock"
    )

  async def test_non_transient_error_bubbles(self):
    # Agent with success model, but error is non-transient → plugin should ignore
    agent = LlmAgent(name="agent", model=SimpleSuccessModel())
    invocation_context = await create_invocation_context(agent)
    plugin = LlmResiliencePlugin(max_retries=2)

    llm_request = LlmRequest(
        contents=[
            types.Content(
                role="user", parts=[types.Part.from_text(text="hello")]
            )
        ]
    )

    class NonTransientError(RuntimeError):
      pass

    # Non-transient error: status code not transient and not Timeout
    # The plugin should return None so that the original error propagates
    result = await plugin.on_model_error_callback(
        callback_context=invocation_context,
        llm_request=llm_request,
        error=NonTransientError("boom"),
    )
    self.assertIsNone(result)
