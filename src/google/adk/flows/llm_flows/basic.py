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

"""Handles basic information to build the LLM request."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import Generator

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ._base_llm_processor import BaseLlmRequestProcessor


class _BasicLlmRequestProcessor(BaseLlmRequestProcessor):

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    llm_request.model = (
        agent.canonical_model
        if isinstance(agent.canonical_model, str)
        else agent.canonical_model.model
    )
    llm_request.config = (
        agent.generate_content_config.model_copy(deep=True)
        if agent.generate_content_config
        else types.GenerateContentConfig()
    )
    if agent.output_schema:
      llm_request.set_output_schema(agent.output_schema)

    llm_request.live_connect_config.response_modalities = (
        invocation_context.run_config.response_modalities
    )
    llm_request.live_connect_config.speech_config = (
        invocation_context.run_config.speech_config
    )
    llm_request.live_connect_config.output_audio_transcription = (
        invocation_context.run_config.output_audio_transcription
    )
    llm_request.live_connect_config.input_audio_transcription = (
        invocation_context.run_config.input_audio_transcription
    )
    llm_request.live_connect_config.realtime_input_config = (
        invocation_context.run_config.realtime_input_config
    )
    llm_request.live_connect_config.enable_affective_dialog = (
        invocation_context.run_config.enable_affective_dialog
    )
    llm_request.live_connect_config.proactivity = (
        invocation_context.run_config.proactivity
    )

    # TODO: handle tool append here, instead of in BaseTool.process_llm_request.

    return
    yield  # Generator requires yield statement in function body.


request_processor = _BasicLlmRequestProcessor()
