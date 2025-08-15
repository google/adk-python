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
from functools import cached_property
import logging
import os
import sys
from typing import AsyncGenerator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from google.genai import Client
from google.genai import types
from google.genai.types import FinishReason
from typing_extensions import override

from .. import version
from ..utils.context_utils import Aclosing
from ..utils.variant_utils import GoogleLLMVariant
from .base_llm import BaseLlm
from .base_llm_connection import BaseLlmConnection
from .gemini_llm_connection import GeminiLlmConnection
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from .llm_request import LlmRequest

logger = logging.getLogger('google_adk.' + __name__)

_NEW_LINE = '\n'
_EXCLUDED_PART_FIELD = {'inline_data': {'data'}}
_AGENT_ENGINE_TELEMETRY_TAG = 'remote_reasoning_engine'
_AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME = 'GOOGLE_CLOUD_AGENT_ENGINE_ID'


class Gemini(BaseLlm):
  """Integration for Gemini models.

  Attributes:
    model: The name of the Gemini model.
  """

  model: str = 'gemini-1.5-flash'

  retry_options: Optional[types.HttpRetryOptions] = None
  """Allow Gemini to retry failed responses.

  Sample:
  ```python
  from google.genai import types

  # ...

  agent = Agent(
    model=Gemini(
      retry_options=types.HttpRetryOptions(initial_delay=1, attempts=2),
    )
  )
  ```
  """

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Provides the list of supported models.

    Returns:
      A list of supported models.
    """

    return [
        r'gemini-.*',
        # model optimizer pattern
        r'model-optimizer-.*',
        # fine-tuned vertex endpoint pattern
        r'projects\/.+\/locations\/.+\/endpoints\/.+',
        # vertex gemini long name
        r'projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+',
    ]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Sends a request to the Gemini model.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """
    await self._preprocess_request(llm_request)
    self._maybe_append_user_content(llm_request)
    logger.info(
        'Sending out request, model: %s, backend: %s, stream: %s',
        llm_request.model,
        self._api_backend,
        stream,
    )
    logger.debug(_build_request_log(llm_request))

    # Always add tracking headers to custom headers given it will override
    # the headers set in the api client constructor to avoid tracking headers
    # being dropped if user provides custom headers or overrides the api client.
    if llm_request.config:
      if not llm_request.config.http_options:
        llm_request.config.http_options = types.HttpOptions()
      llm_request.config.http_options.headers = self._merge_tracking_headers(
          llm_request.config.http_options.headers
      )

    if stream:
      responses = await self.api_client.aio.models.generate_content_stream(
          model=llm_request.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )
      response = None
      thought_text = ''
      text = ''
      usage_metadata = None
      # for sse, similar as bidi (see receive method in gemini_llm_connecton.py),
      # we need to mark those text content as partial and after all partial
      # contents are sent, we send an accumulated event which contains all the
      # previous partial content. The only difference is bidi rely on
      # complete_turn flag to detect end while sse depends on finish_reason.
      async with Aclosing(responses) as agen:
        async for response in agen:
          logger.debug(_build_response_log(response))
          llm_response = LlmResponse.create(response)
          usage_metadata = llm_response.usage_metadata
          if (
              llm_response.content
              and llm_response.content.parts
              and llm_response.content.parts[0].text
          ):
            part0 = llm_response.content.parts[0]
            if part0.thought:
              thought_text += part0.text
            else:
              text += part0.text
            llm_response.partial = True
          elif (thought_text or text) and (
              not llm_response.content
              or not llm_response.content.parts
              # don't yield the merged text event when receiving audio data
              or not llm_response.content.parts[0].inline_data
          ):
            parts = []
            if thought_text:
              parts.append(types.Part(text=thought_text, thought=True))
            if text:
              parts.append(types.Part.from_text(text=text))
            yield LlmResponse(
                content=types.ModelContent(parts=parts),
                usage_metadata=llm_response.usage_metadata,
            )
            thought_text = ''
            text = ''
          yield llm_response

      # generate an aggregated content at the end regardless the
      # response.candidates[0].finish_reason
      if (text or thought_text) and response and response.candidates:
        parts = []
        if thought_text:
          parts.append(types.Part(text=thought_text, thought=True))
        if text:
          parts.append(types.Part.from_text(text=text))
        yield LlmResponse(
            content=types.ModelContent(parts=parts),
            error_code=None
            if response.candidates[0].finish_reason == FinishReason.STOP
            else response.candidates[0].finish_reason,
            error_message=None
            if response.candidates[0].finish_reason == FinishReason.STOP
            else response.candidates[0].finish_message,
            usage_metadata=usage_metadata,
        )

    else:
      response = await self.api_client.aio.models.generate_content(
          model=llm_request.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )
      logger.info('Response received from the model.')
      logger.debug(_build_response_log(response))
      yield LlmResponse.create(response)

  @cached_property
  def api_client(self) -> Client:
    """Provides the api client.

    Returns:
      The api client.
    """
    return Client(
        http_options=types.HttpOptions(
            headers=self._tracking_headers,
            retry_options=self.retry_options,
        )
    )

  @cached_property
  def _api_backend(self) -> GoogleLLMVariant:
    return (
        GoogleLLMVariant.VERTEX_AI
        if self.api_client.vertexai
        else GoogleLLMVariant.GEMINI_API
    )

  @cached_property
  def _tracking_headers(self) -> dict[str, str]:
    framework_label = f'google-adk/{version.__version__}'
    if os.environ.get(_AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME):
      framework_label = f'{framework_label}+{_AGENT_ENGINE_TELEMETRY_TAG}'
    language_label = 'gl-python/' + sys.version.split()[0]
    version_header_value = f'{framework_label} {language_label}'
    tracking_headers = {
        'x-goog-api-client': version_header_value,
        'user-agent': version_header_value,
    }
    return tracking_headers

  @cached_property
  def _live_api_version(self) -> str:
    if self._api_backend == GoogleLLMVariant.VERTEX_AI:
      # use beta version for vertex api
      return 'v1beta1'
    else:
      # use v1alpha for using API KEY from Google AI Studio
      return 'v1alpha'

  @cached_property
  def _live_api_client(self) -> Client:
    return Client(
        http_options=types.HttpOptions(
            headers=self._tracking_headers, api_version=self._live_api_version
        )
    )

  @contextlib.asynccontextmanager
  async def connect(self, llm_request: LlmRequest) -> BaseLlmConnection:
    """Connects to the Gemini model and returns an llm connection.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.

    Yields:
      BaseLlmConnection, the connection to the Gemini model.
    """
    # add tracking headers to custom headers and set api_version given
    # the customized http options will override the one set in the api client
    # constructor
    if (
        llm_request.live_connect_config
        and llm_request.live_connect_config.http_options
    ):
      if not llm_request.live_connect_config.http_options.headers:
        llm_request.live_connect_config.http_options.headers = {}
      llm_request.live_connect_config.http_options.headers.update(
          self._tracking_headers
      )
      llm_request.live_connect_config.http_options.api_version = (
          self._live_api_version
      )

    llm_request.live_connect_config.system_instruction = types.Content(
        role='system',
        parts=[
            types.Part.from_text(text=llm_request.config.system_instruction)
        ],
    )
    llm_request.live_connect_config.tools = llm_request.config.tools
    logger.info('Connecting to live with llm_request:%s', llm_request)
    async with self._live_api_client.aio.live.connect(
        model=llm_request.model, config=llm_request.live_connect_config
    ) as live_session:
      yield GeminiLlmConnection(live_session)

  async def _adapt_computer_use_tool(self, llm_request: LlmRequest) -> None:
    """Adapt the google computer use predefined functions to the adk computer use toolset."""

    from ..tools.computer_use.computer_use_toolset import ComputerUseToolset

    async def convert_wait_to_wait_5_seconds(wait_func):
      async def wait_5_seconds():
        return await wait_func(5)

      return wait_5_seconds

    await ComputerUseToolset.adapt_computer_use_tool(
        'wait', convert_wait_to_wait_5_seconds, llm_request
    )

  async def _preprocess_request(self, llm_request: LlmRequest) -> None:

    if self._api_backend == GoogleLLMVariant.GEMINI_API:
      # Using API key from Google AI Studio to call model doesn't support labels.
      if llm_request.config:
        llm_request.config.labels = None

      if llm_request.contents:
        for content in llm_request.contents:
          if not content.parts:
            continue
          for part in content.parts:
            _remove_display_name_if_present(part.inline_data)
            _remove_display_name_if_present(part.file_data)

    # Initialize config if needed
    if llm_request.config and llm_request.config.tools:
      # Check if computer use is configured
      for tool in llm_request.config.tools:
        if (
            isinstance(tool, (types.Tool, types.ToolDict))
            and hasattr(tool, 'computer_use')
            and tool.computer_use
        ):
          llm_request.config.system_instruction = None
          await self._adapt_computer_use_tool(llm_request)

  def _merge_tracking_headers(self, headers: dict[str, str]) -> dict[str, str]:
    """Merge tracking headers to the given headers."""
    headers = headers or {}
    for key, tracking_header_value in self._tracking_headers.items():
      custom_value = headers.get(key, None)
      if not custom_value:
        headers[key] = tracking_header_value
        continue

      # Merge tracking headers with existing headers and avoid duplicates.
      value_parts = tracking_header_value.split(' ')
      for custom_value_part in custom_value.split(' '):
        if custom_value_part not in value_parts:
          value_parts.append(custom_value_part)
      headers[key] = ' '.join(value_parts)
    return headers


def _build_function_declaration_log(
    func_decl: types.FunctionDeclaration,
) -> str:
  param_str = '{}'
  if func_decl.parameters and func_decl.parameters.properties:
    param_str = str({
        k: v.model_dump(exclude_none=True)
        for k, v in func_decl.parameters.properties.items()
    })
  return_str = ''
  if func_decl.response:
    return_str = '-> ' + str(func_decl.response.model_dump(exclude_none=True))
  return f'{func_decl.name}: {param_str} {return_str}'


def _build_request_log(req: LlmRequest) -> str:
  function_decls: list[types.FunctionDeclaration] = cast(
      list[types.FunctionDeclaration],
      req.config.tools[0].function_declarations if req.config.tools else [],
  )
  function_logs = (
      [
          _build_function_declaration_log(func_decl)
          for func_decl in function_decls
      ]
      if function_decls
      else []
  )
  contents_logs = [
      content.model_dump_json(
          exclude_none=True,
          exclude={
              'parts': {
                  i: _EXCLUDED_PART_FIELD for i in range(len(content.parts))
              }
          },
      )
      for content in req.contents
  ]

  return f"""
LLM Request:
-----------------------------------------------------------
System Instruction:
{req.config.system_instruction}
-----------------------------------------------------------
Contents:
{_NEW_LINE.join(contents_logs)}
-----------------------------------------------------------
Functions:
{_NEW_LINE.join(function_logs)}
-----------------------------------------------------------
"""


def _build_response_log(resp: types.GenerateContentResponse) -> str:
  function_calls_text = []
  if function_calls := resp.function_calls:
    for func_call in function_calls:
      function_calls_text.append(
          f'name: {func_call.name}, args: {func_call.args}'
      )
  return f"""
LLM Response:
-----------------------------------------------------------
Text:
{resp.text}
-----------------------------------------------------------
Function calls:
{_NEW_LINE.join(function_calls_text)}
-----------------------------------------------------------
Raw response:
{resp.model_dump_json(exclude_none=True)}
-----------------------------------------------------------
"""


def _remove_display_name_if_present(
    data_obj: Union[types.Blob, types.FileData, None],
):
  """Sets display_name to None for the Gemini API (non-Vertex) backend.

  This backend does not support the display_name parameter for file uploads,
  so it must be removed to prevent request failures.
  """
  if data_obj and data_obj.display_name:
    data_obj.display_name = None
