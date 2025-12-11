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
import copy
from functools import cached_property
import logging
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from google.genai import types
from google.genai.errors import ClientError
from typing_extensions import override

from ..utils._client_labels_utils import get_client_labels
from ..utils.context_utils import Aclosing
from ..utils.streaming_utils import StreamingResponseAggregator
from ..utils.variant_utils import GoogleLLMVariant
from .base_llm import BaseLlm
from .base_llm_connection import BaseLlmConnection
from .gemini_llm_connection import GeminiLlmConnection
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from google.genai import Client

  from .llm_request import LlmRequest

logger = logging.getLogger('google_adk.' + __name__)

_NEW_LINE = '\n'
_EXCLUDED_PART_FIELD = {'inline_data': {'data'}}


_RESOURCE_EXHAUSTED_POSSIBLE_FIX_MESSAGE = """
On how to mitigate this issue, please refer to:

https://google.github.io/adk-docs/agents/models/#error-code-429-resource_exhausted
"""


class _ResourceExhaustedError(ClientError):
  """Represents an resources exhausted error received from the Model."""

  def __init__(
      self,
      client_error: ClientError,
  ):
    super().__init__(
        code=client_error.code,
        response_json=client_error.details,
        response=client_error.response,
    )

  def __str__(self):
    # We don't get override the actual message on ClientError, so we override
    # this method instead. This will ensure that when the exception is
    # stringified (for either publishing the exception on console or to logs)
    # we put in the required details for the developer.
    base_message = super().__str__()
    return f'{_RESOURCE_EXHAUSTED_POSSIBLE_FIX_MESSAGE}\n\n{base_message}'


class Gemini(BaseLlm):
  """Integration for Gemini models.

  Attributes:
    model: The name of the Gemini model.
    use_interactions_api: Whether to use the interactions API for model
      invocation.
  """

  model: str = 'gemini-2.5-flash'

  speech_config: Optional[types.SpeechConfig] = None

  use_interactions_api: bool = False
  """Whether to use the interactions API for model invocation.

  When enabled, uses the interactions API (client.aio.interactions.create())
  instead of the traditional generate_content API. The interactions API
  provides stateful conversation capabilities, allowing you to chain
  interactions using previous_interaction_id instead of sending full history.
  The response format will be converted to match the existing LlmResponse
  structure for compatibility.

  Sample:
  ```python
  agent = Agent(
    model=Gemini(use_interactions_api=True)
  )
  ```
  """

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

    # Handle context caching if configured
    cache_metadata = None
    cache_manager = None
    if llm_request.cache_config:
      from ..telemetry.tracing import tracer
      from .gemini_context_cache_manager import GeminiContextCacheManager

      with tracer.start_as_current_span('handle_context_caching') as span:
        cache_manager = GeminiContextCacheManager(self.api_client)
        cache_metadata = await cache_manager.handle_context_caching(llm_request)
        if cache_metadata:
          if cache_metadata.cache_name:
            span.set_attribute('cache_action', 'active_cache')
            span.set_attribute('cache_name', cache_metadata.cache_name)
          else:
            span.set_attribute('cache_action', 'fingerprint_only')

    logger.info(
        'Sending out request, model: %s, backend: %s, stream: %s',
        llm_request.model,
        self._api_backend,
        stream,
    )

    # Always add tracking headers to custom headers given it will override
    # the headers set in the api client constructor to avoid tracking headers
    # being dropped if user provides custom headers or overrides the api client.
    if llm_request.config:
      if not llm_request.config.http_options:
        llm_request.config.http_options = types.HttpOptions()
      llm_request.config.http_options.headers = self._merge_tracking_headers(
          llm_request.config.http_options.headers
      )

    try:
      # Use interactions API if enabled
      if self.use_interactions_api:
        async for llm_response in self._generate_content_via_interactions(
            llm_request, stream
        ):
          yield llm_response
        return

      logger.debug(_build_request_log(llm_request))

      if stream:
        responses = await self.api_client.aio.models.generate_content_stream(
            model=llm_request.model,
            contents=llm_request.contents,
            config=llm_request.config,
        )

        # for sse, similar as bidi (see receive method in
        # gemini_llm_connection.py), we need to mark those text content as
        # partial and after all partial contents are sent, we send an
        # accumulated event which contains all the previous partial content. The
        # only difference is bidi rely on complete_turn flag to detect end while
        # sse depends on finish_reason.
        aggregator = StreamingResponseAggregator()
        async with Aclosing(responses) as agen:
          async for response in agen:
            logger.debug(_build_response_log(response))
            async with Aclosing(
                aggregator.process_response(response)
            ) as aggregator_gen:
              async for llm_response in aggregator_gen:
                yield llm_response
        if (close_result := aggregator.close()) is not None:
          # Populate cache metadata in the final aggregated response for
          # streaming
          if cache_metadata:
            cache_manager.populate_cache_metadata_in_response(
                close_result, cache_metadata
            )
          yield close_result

      else:
        response = await self.api_client.aio.models.generate_content(
            model=llm_request.model,
            contents=llm_request.contents,
            config=llm_request.config,
        )
        logger.info('Response received from the model.')
        logger.debug(_build_response_log(response))

        llm_response = LlmResponse.create(response)
        if cache_metadata:
          cache_manager.populate_cache_metadata_in_response(
              llm_response, cache_metadata
          )
        yield llm_response
    except ClientError as ce:
      if ce.code == 429:
        # We expect running into a Resource Exhausted error to be a common
        # client error that developers would run into. We enhance the messaging
        # with possible fixes to this issue.
        raise _ResourceExhaustedError(ce) from ce

      raise ce

  async def _generate_content_via_interactions(
      self,
      llm_request: LlmRequest,
      stream: bool,
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generate content using the interactions API.

    The interactions API provides stateful conversation capabilities. When
    previous_interaction_id is set in the request, the API chains interactions
    instead of requiring full conversation history.

    Note: Context caching is not used with the Interactions API since it
    maintains conversation state via previous_interaction_id.

    Args:
      llm_request: The LLM request to send.
      stream: Whether to stream the response.

    Yields:
      LlmResponse objects converted from interaction responses.
    """
    from .interactions_utils import generate_content_via_interactions

    async for llm_response in generate_content_via_interactions(
        api_client=self.api_client,
        llm_request=llm_request,
        stream=stream,
    ):
      yield llm_response

  @cached_property
  def api_client(self) -> Client:
    """Provides the api client.

    Returns:
      The api client.
    """
    from google.genai import Client

    return Client(
        http_options=types.HttpOptions(
            headers=self._tracking_headers(),
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

  def _tracking_headers(self) -> dict[str, str]:
    labels = get_client_labels()
    header_value = ' '.join(labels)
    tracking_headers = {
        'x-goog-api-client': header_value,
        'user-agent': header_value,
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
    from google.genai import Client

    return Client(
        http_options=types.HttpOptions(
            headers=self._tracking_headers(), api_version=self._live_api_version
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
          self._tracking_headers()
      )
      llm_request.live_connect_config.http_options.api_version = (
          self._live_api_version
      )

    if self.speech_config is not None:
      llm_request.live_connect_config.speech_config = self.speech_config

    llm_request.live_connect_config.system_instruction = types.Content(
        role='system',
        parts=[
            types.Part.from_text(text=llm_request.config.system_instruction)
        ],
    )
    if (
        llm_request.live_connect_config.session_resumption
        and llm_request.live_connect_config.session_resumption.transparent
    ):
      logger.debug(
          'session resumption config: %s',
          llm_request.live_connect_config.session_resumption,
      )
      logger.debug(
          'self._api_backend: %s',
          self._api_backend,
      )
      if self._api_backend == GoogleLLMVariant.GEMINI_API:
        raise ValueError(
            'Transparent session resumption is only supported for Vertex AI'
            ' backend. Please use Vertex AI backend.'
        )
    llm_request.live_connect_config.tools = llm_request.config.tools
    logger.info('Connecting to live for model: %s', llm_request.model)
    logger.debug('Connecting to live with llm_request:%s', llm_request)
    logger.debug('Live connect config: %s', llm_request.live_connect_config)
    async with self._live_api_client.aio.live.connect(
        model=llm_request.model, config=llm_request.live_connect_config
    ) as live_session:
      yield GeminiLlmConnection(live_session, api_backend=self._api_backend)

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
            # Create copies to avoid mutating the original objects
            if part.inline_data:
              part.inline_data = copy.copy(part.inline_data)
              _remove_display_name_if_present(part.inline_data)
            if part.file_data:
              part.file_data = copy.copy(part.file_data)
              _remove_display_name_if_present(part.file_data)

    # Initialize config if needed
    if llm_request.config and llm_request.config.tools:
      # Check if computer use is configured
      for tool in llm_request.config.tools:
        if isinstance(tool, types.Tool) and tool.computer_use:
          llm_request.config.system_instruction = None
          await self._adapt_computer_use_tool(llm_request)

  def _merge_tracking_headers(self, headers: dict[str, str]) -> dict[str, str]:
    """Merge tracking headers to the given headers."""
    headers = headers or {}
    for key, tracking_header_value in self._tracking_headers().items():
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
  elif func_decl.parameters_json_schema:
    param_str = str(func_decl.parameters_json_schema)

  return_str = ''
  if func_decl.response:
    return_str = '-> ' + str(func_decl.response.model_dump(exclude_none=True))
  elif func_decl.response_json_schema:
    return_str = '-> ' + str(func_decl.response_json_schema)

  return f'{func_decl.name}: {param_str} {return_str}'


def _build_request_log(req: LlmRequest) -> str:
  # Find which tool contains function_declarations
  function_decls: list[types.FunctionDeclaration] = []
  function_decl_tool_index: Optional[int] = None

  if req.config.tools:
    for idx, tool in enumerate(req.config.tools):
      if tool.function_declarations:
        function_decls = cast(
            list[types.FunctionDeclaration], tool.function_declarations
        )
        function_decl_tool_index = idx
        break

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

  # Build exclusion dict for config logging
  tools_exclusion = (
      {function_decl_tool_index: {'function_declarations'}}
      if function_decl_tool_index is not None
      else True
  )

  try:
    config_log = str(
        req.config.model_dump(
            exclude_none=True,
            exclude={
                'system_instruction': True,
                'tools': tools_exclusion if req.config.tools else True,
            },
        )
    )
  except Exception:
    config_log = repr(req.config)

  return f"""
LLM Request:
-----------------------------------------------------------
System Instruction:
{req.config.system_instruction}
-----------------------------------------------------------
Config:
{config_log}
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
