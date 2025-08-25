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

from __future__ import annotations

import contextlib
import logging
import os
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override
import websockets

from .base_llm import BaseLlm
from .base_llm_connection import BaseLlmConnection
from .llm_response import LlmResponse
from .openai_llm_connection import OpenAILlmConnection
from .openai_tool_schema import function_tools_to_openai_session_tools

logger = logging.getLogger('google_adk.' + __name__)

if TYPE_CHECKING:
  from .llm_request import LlmRequest


class OpenAIRealtime(BaseLlm):
  """Integration for OpenAI Realtime models (WebSocket).

  Supported models include gpt-4o-realtime-*, etc.
  """

  model: str = 'gpt-4o-realtime-preview'

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    # Match common realtime model names. Users can specify exact names.
    return [
        r'gpt-4o-realtime-.*',
        r'gpt-4o-realtime-preview',
    ]

  async def generate_content_async(
      self, llm_request: 'LlmRequest', stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Non-live text generation via realtime is not implemented.

    The ADK uses live flows for realtime. Fallback: raise NotImplementedError.
    """
    raise NotImplementedError(
        f'Async generation is not supported for {self.model}. Use live mode.'
    )
    yield  # satisfy generator type checker

  @contextlib.asynccontextmanager
  async def connect(
      self, llm_request: 'LlmRequest'
  ) -> AsyncGenerator[BaseLlmConnection, None]:
    """Open a WebSocket to OpenAI Realtime and configure session settings.

    Environment:
      - OPENAI_API_KEY must be set unless llm_request.live_connect_config sets headers.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key and (
        not llm_request.live_connect_config
        or not llm_request.live_connect_config.http_options
        or not llm_request.live_connect_config.http_options.headers
        or 'Authorization'
        not in llm_request.live_connect_config.http_options.headers
    ):
      raise ValueError('OPENAI_API_KEY is required for OpenAI Realtime.')

    model_name = llm_request.model or self.model
    url = f'wss://api.openai.com/v1/realtime?model={model_name}'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'OpenAI-Beta': 'realtime=v1',
    }
    # Allow user to add/override headers
    if (
        llm_request.live_connect_config
        and llm_request.live_connect_config.http_options
        and llm_request.live_connect_config.http_options.headers
    ):
      headers.update(llm_request.live_connect_config.http_options.headers)

    ws = await websockets.connect(
        url, additional_headers=headers, max_size=None
    )

    # Initial session.update; values come from RunConfig (openai_realtime_session) if present
    session_update = {
        'type': 'session.update',
        'session': {},
    }
    # Merge vendor-specific session overrides passed via RunConfig.basic processor
    try:
      labels = getattr(llm_request.config, 'labels', {}) or {}
      vendor_raw = labels.get('adk_openai_session_json')
      overrides = None
      if vendor_raw:
        if isinstance(vendor_raw, str):
          import json as _json

          try:
            overrides = _json.loads(vendor_raw)
          except Exception:
            # Accept single-quoted or Python-literal-like strings
            try:
              import ast as _ast

              overrides = _ast.literal_eval(vendor_raw)
            except Exception:
              overrides = None
        elif isinstance(vendor_raw, dict):
          overrides = vendor_raw
      if isinstance(overrides, dict):
        session_update['session'].update(overrides)
    except Exception:
      pass
    # Encourage function calling by default if not specified
    if 'tool_choice' not in session_update['session']:
      session_update['session']['tool_choice'] = 'auto'
    # Map key settings if present
    tool_param_names: dict[str, set[str]] = {}
    if llm_request.config:
      # Voice, modalities etc. are provided via overrides; we only pass system instruction/tools here
      if llm_request.config.system_instruction:
        session_update['session'][
            'instructions'
        ] = llm_request.config.system_instruction
      if llm_request.config.tools:
        tools = function_tools_to_openai_session_tools(llm_request.config.tools)
        # Track expected arg names for each tool for potential validation (not mapping)
        for tool in llm_request.config.tools:
          if (
              isinstance(tool, (types.Tool, types.ToolDict))
              and tool.function_declarations
          ):
            for decl in tool.function_declarations:
              if decl.parameters and getattr(
                  decl.parameters, 'properties', None
              ):
                tool_param_names[decl.name] = set(
                    decl.parameters.properties.keys()
                )
        if tools:
          session_update['session']['tools'] = tools

    # Send session update after connect (debug: inspect payload before sending)
    try:
      logger.debug(
          'OpenAI Realtime session.update payload: %s',
          json_dumps(session_update),
      )
    except Exception:
      pass
    # import pdb  # noqa: E402
    # pdb.set_trace()
    await ws.send(json_dumps(session_update))

    try:
      yield OpenAILlmConnection(
          websocket=ws,
          model_name=model_name,
          tool_param_names=tool_param_names,
      )
    finally:
      try:
        await ws.close()
      except Exception:
        pass


def json_dumps(obj) -> str:
  import json

  return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)
