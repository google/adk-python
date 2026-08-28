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

"""Handles basic information to build the LLM request."""

from __future__ import annotations

from typing import AsyncGenerator

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...utils import model_name_utils
from ._base_llm_processor import BaseLlmRequestProcessor
from ._invocation_utils import as_llm_agent
from ._invocation_utils import copy_http_options as _copy_http_options
from ._invocation_utils import copy_or_none as _copy_or_none
from ._invocation_utils import require_run_config


def _merge_run_config_http_options(
    config: types.GenerateContentConfig,
    run_config_http_options: types.HttpOptions,
) -> None:
  """Merges RunConfig http_options into the request config, RunConfig wins.

  The RunConfig's options are copied in rather than aliased, so request
  assembly cannot write back into the RunConfig.

  base_url and api_version are configuration-time settings, not request-time,
  so they are intentionally not merged into an existing config.http_options.
  """
  if config.http_options is None:
    config.http_options = _copy_http_options(run_config_http_options)
    return

  if run_config_http_options.headers:
    if config.http_options.headers is None:
      config.http_options.headers = {}
    config.http_options.headers.update(run_config_http_options.headers)

  for field in ('timeout', 'retry_options', 'extra_body'):
    value = getattr(run_config_http_options, field, None)
    if value is not None:
      setattr(config.http_options, field, value)


def _copy_request_scoped_fields(
    config: types.GenerateContentConfig,
) -> types.GenerateContentConfig:
  """Copies the agent config fields that request assembly goes on to mutate.

  ``model_copy`` is shallow, so every container the agent configured would still
  be the agent's own object, and a write during assembly would outlive the
  invocation and be seen by every later run of that agent.

  Every list and dict is copied, not just the fields assembly happens to touch
  today: a before-model callback receives the request config and can append to
  any of them. The elements themselves are shared, because assembly replaces
  entries rather than mutating them.

  ``http_options`` needs its own copy because it is a model rather than a
  container. It can hold a live httpx or aiohttp client and an SSL context,
  none of which survive a deep copy, so only its ``headers`` dict is copied.
  """
  updates: dict[str, object] = {}
  for name, value in config:
    if isinstance(value, list):
      updates[name] = list(value)
    elif isinstance(value, dict):
      updates[name] = dict(value)
  if config.http_options is not None:
    updates['http_options'] = _copy_http_options(config.http_options)
  return config.model_copy(update=updates)


def _build_basic_request(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
  """Populate basic LlmRequest fields from agent configuration.

  Sets up model, config, output_schema, and live connect configuration
  based on the agent and run configuration.

  Args:
    invocation_context: The invocation context containing agent and run config.
    llm_request: The LlmRequest to populate.
  """
  agent = as_llm_agent(invocation_context)
  run_config = require_run_config(invocation_context)
  model = agent.canonical_model
  llm_request.model = model.model

  # Preserved across the agent-config overwrite below, then merged back.
  run_config_http_options = llm_request.config.http_options

  generate_content_config = agent.generate_content_config
  llm_request.config = (
      _copy_request_scoped_fields(generate_content_config)
      if generate_content_config
      else types.GenerateContentConfig()
  )

  if run_config_http_options:
    _merge_run_config_http_options(llm_request.config, run_config_http_options)

  # Merge per-invocation user labels from RunConfig into the request config
  # (e.g. for billing, telemetry, and revenue attribution across services).
  if invocation_context.run_config and invocation_context.run_config.labels:
    if llm_request.config.labels is None:
      llm_request.config.labels = {}
    llm_request.config.labels.update(invocation_context.run_config.labels)
  # Only set output_schema if no tools are specified. as of now, model don't
  # support output_schema and tools together. we have a workaround to support
  # both output_schema and tools at the same time. see
  # _output_schema_processor.py for details
  #
  # task-mode agents skip output_schema configuration in
  # the basic flow. Structured output for tasks is collected via the
  # finish_task tool schema instead.
  if getattr(agent, 'mode', None) != 'task' and agent.output_schema:
    if not agent.tools or model.capabilities.output_schema_and_tools:
      llm_request.set_output_schema(agent.output_schema)

  llm_request.live_connect_config.response_modalities = (
      [types.Modality(m) for m in run_config.response_modalities]
      if run_config.response_modalities is not None
      else None
  )
  llm_request.live_connect_config.speech_config = run_config.speech_config
  llm_request.live_connect_config.output_audio_transcription = (
      run_config.output_audio_transcription
  )
  llm_request.live_connect_config.input_audio_transcription = (
      run_config.input_audio_transcription
  )
  llm_request.live_connect_config.realtime_input_config = (
      run_config.realtime_input_config
  )
  llm_request.live_connect_config.explicit_vad_signal = (
      run_config.explicit_vad_signal
  )
  llm_request.live_connect_config.translation_config = (
      run_config.translation_config
  )
  active_model_name = (
      getattr(getattr(agent, 'canonical_live_model', None), 'model', None)
      or llm_request.model
  )
  is_gemini_3_x = model_name_utils._is_gemini_3_x_live(active_model_name)
  llm_request.live_connect_config.enable_affective_dialog = (
      None if is_gemini_3_x else run_config.enable_affective_dialog
  )
  llm_request.live_connect_config.proactivity = (
      None if is_gemini_3_x else run_config.proactivity
  )
  # Copied rather than aliased: live request assembly writes into both of these
  # while the session runs. `BaseLlmFlow.run_live` stamps each server-issued
  # resumption handle onto `session_resumption`, and sets
  # `initial_history_in_client_content` on `history_config` when it seeds a
  # fresh connection with history. Aliasing the RunConfig's own objects makes
  # those writes outlive the invocation, so a RunConfig reused for a later run
  # would carry a stale handle into it. This mirrors what
  # `_copy_request_scoped_fields` already does for `llm_request.config`.
  llm_request.live_connect_config.session_resumption = _copy_or_none(
      run_config.session_resumption
  )
  llm_request.live_connect_config.history_config = _copy_or_none(
      run_config.history_config
  )
  llm_request.live_connect_config.context_window_compression = (
      run_config.context_window_compression
  )
  llm_request.live_connect_config.avatar_config = run_config.avatar_config


class _BasicLlmRequestProcessor(BaseLlmRequestProcessor):

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    _build_basic_request(invocation_context, llm_request)

    # TODO: handle tool append here, instead of in BaseTool.process_llm_request.

    return
    yield  # Generator requires yield statement in function body.


request_processor = _BasicLlmRequestProcessor()
