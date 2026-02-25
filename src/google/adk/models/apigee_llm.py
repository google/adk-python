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
import atexit
import base64
import collections.abc
import enum
from functools import cached_property
import json
import logging
import os
from typing import Any
from typing import AsyncGenerator
from typing import Generator
from typing import Optional
from typing import TYPE_CHECKING

from google.adk import version as adk_version
from google.genai import types
import httpx
import tenacity
from typing_extensions import override

from ..utils.env_utils import is_env_enabled
from .google_llm import Gemini
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from google.genai import Client

  from .llm_request import LlmRequest


logger = logging.getLogger('google_adk.' + __name__)

_APIGEE_PROXY_URL_ENV_VARIABLE_NAME = 'APIGEE_PROXY_URL'
_GOOGLE_GENAI_USE_VERTEXAI_ENV_VARIABLE_NAME = 'GOOGLE_GENAI_USE_VERTEXAI'
_PROJECT_ENV_VARIABLE_NAME = 'GOOGLE_CLOUD_PROJECT'
_LOCATION_ENV_VARIABLE_NAME = 'GOOGLE_CLOUD_LOCATION'

_CUSTOM_METADATA_FIELDS = (
    'id',
    'created',
    'model',
    'service_tier',
    'object',
)


class ApigeeLlm(Gemini):
  """A BaseLlm implementation for calling Apigee proxy.

  Attributes:
    model: The name of the Gemini model.
  """

  class ApiType(str, enum.Enum):
    """The supported API types for Apigee LLM."""

    UNKNOWN = 'unknown'
    CHAT_COMPLETIONS = 'chat_completions'
    GENAI = 'genai'

    @classmethod
    def _missing_(cls, value):
      # Empty string or None should return UNKNOWN.
      if not value:
        return cls.UNKNOWN
      return super()._missing_(value)

  def __init__(
      self,
      *,
      model: str,
      proxy_url: str | None = None,
      custom_headers: dict[str, str] | None = None,
      retry_options: Optional[types.HttpRetryOptions] = None,
      api_type: ApiType | str = ApiType.UNKNOWN,
  ):
    """Initializes the Apigee LLM backend.

    Args:
      model: The model string specifies the LLM provider (e.g., Vertex AI,
        Gemini), API version, and the model ID. Supported format:
        `apigee/[<provider>/][<version>/]<model_id>`

        Components
          `provider` (optional): `vertex_ai` or `gemini`. If omitted, behavior
            depends on the `GOOGLE_GENAI_USE_VERTEXAI` environment variable. If
            that is not set to TRUE or 1, it defaults to `gemini`. `provider`
            takes precedence over `GOOGLE_GENAI_USE_VERTEXAI`.
          `version` (optional): The API version (e.g., `v1`, `v1beta`). If
            omitted, the default version for the provider is used.
          `model_id` (required): The model identifier (e.g.,
            `gemini-2.5-flash`).

        Examples
          - `apigee/gemini-2.5-flash`
          - `apigee/v1/gemini-2.5-flash`
          - `apigee/vertex_ai/gemini-2.5-flash`
          - `apigee/gemini/v1/gemini-2.5-flash`
          - `apigee/vertex_ai/v1beta/gemini-2.5-flash`
      proxy_url: The URL of the Apigee proxy.
      custom_headers: A dictionary of headers to be sent with the request.
        If needed, you can add authorization headers here, for example:
        {'Authorization': f'Bearer {API_KEY}'}. ApigeeLlm already handles
        authorization headers in Vertex AI and Gemini API calls.
      retry_options: Allow google-genai to retry failed responses.
      api_type: The type of API to use. One of `ApiType` or string.
    """  # fmt: skip

    super().__init__(model=model, retry_options=retry_options)
    # Validate the model string. Create a helper method to validate the model
    # string.
    if not _validate_model_string(model):
      raise ValueError(f'Invalid model string: {model}')
    if isinstance(api_type, str):
      api_type = ApigeeLlm.ApiType(api_type)
    if api_type and api_type != ApigeeLlm.ApiType.UNKNOWN:
      self._api_type = api_type
    elif model.startswith(('apigee/gemini/', 'apigee/vertex_ai/')):
      self._api_type = ApigeeLlm.ApiType.GENAI
    elif model.startswith('apigee/openai/'):
      self._api_type = ApigeeLlm.ApiType.CHAT_COMPLETIONS
    else:
      self._api_type = ApigeeLlm.ApiType.GENAI
    self._isvertexai = _identify_vertexai(model, self._api_type)

    # Set the project and location for Vertex AI.
    if self._isvertexai:
      self._project = os.environ.get(_PROJECT_ENV_VARIABLE_NAME)
      self._location = os.environ.get(_LOCATION_ENV_VARIABLE_NAME)

      if not self._project:
        raise ValueError(
            f'The {_PROJECT_ENV_VARIABLE_NAME} environment variable must be'
            ' set.'
        )

      if not self._location:
        raise ValueError(
            f'The {_LOCATION_ENV_VARIABLE_NAME} environment variable must be'
            ' set.'
        )

    self._api_version = _identify_api_version(model)
    self._proxy_url = proxy_url or os.environ.get(
        _APIGEE_PROXY_URL_ENV_VARIABLE_NAME
    )
    self._custom_headers = custom_headers or {}
    self._user_agent = f'google-adk/{adk_version.__version__}'

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Provides the list of supported models.

    Returns:
      A list of supported models.
    """

    return [
        r'apigee\/.*',
    ]

  @cached_property
  def _completions_http_client(self) -> CompletionsHTTPClient:
    """Provides the completions HTTP client."""
    return CompletionsHTTPClient(
        base_url=self._proxy_url,
        headers=self._merge_tracking_headers(self._custom_headers),
        retry_options=self.retry_options,
    )

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    if self._api_type == ApigeeLlm.ApiType.CHAT_COMPLETIONS:
      await self._preprocess_other_requests(llm_request)
      async for (
          response
      ) in self._completions_http_client.generate_content_async(
          llm_request, stream
      ):
        yield response
    else:
      async for response in super().generate_content_async(llm_request, stream):
        yield response

  async def _preprocess_other_requests(self, llm_request: LlmRequest) -> None:
    """Preprocesses the request for non-Gemini/Vertex AI models."""
    llm_request.model = _get_model_id(llm_request.model)
    if llm_request.config and llm_request.config.tools:
      # Check if computer use is configured
      for tool in llm_request.config.tools:
        if isinstance(tool, types.Tool) and tool.computer_use:
          llm_request.config.system_instruction = None
          await self._adapt_computer_use_tool(llm_request)
    self._maybe_append_user_content(llm_request)

  @cached_property
  def api_client(self) -> Client:
    """Provides the api client.

    Returns:
      The api client.
    """
    from google.genai import Client

    kwargs_for_http_options = {}
    if self._api_version:
      kwargs_for_http_options['api_version'] = self._api_version
    http_options = types.HttpOptions(
        base_url=self._proxy_url,
        headers=self._merge_tracking_headers(self._custom_headers),
        retry_options=self.retry_options,
        **kwargs_for_http_options,
    )

    kwargs_for_client = {}
    kwargs_for_client['vertexai'] = self._isvertexai
    if self._isvertexai:
      kwargs_for_client['project'] = self._project
      kwargs_for_client['location'] = self._location

    return Client(
        http_options=http_options,
        **kwargs_for_client,
    )

  @override
  async def _preprocess_request(self, llm_request: LlmRequest) -> None:
    llm_request.model = _get_model_id(llm_request.model)
    await super()._preprocess_request(llm_request)


def _identify_vertexai(model: str, api_type: ApigeeLlm.ApiType) -> bool:
  """Returns if a model is Vertex AI.

  1. The api_type is GENAI or UNKNOWN.
  2. The model is provider is Vertex AI model or the
    GOOGLE_GENAI_USE_VERTEXAI environment variable is set to TRUE or 1.

  Args:
    model: The model string.
    api_type: The type of API to use.
  """
  if api_type not in (ApigeeLlm.ApiType.GENAI, ApigeeLlm.ApiType.UNKNOWN):
    return False
  if model.startswith('apigee/gemini/'):
    return False
  if model.startswith('apigee/openai/'):
    return False
  return model.startswith('apigee/vertex_ai/') or is_env_enabled(
      _GOOGLE_GENAI_USE_VERTEXAI_ENV_VARIABLE_NAME
  )


def _identify_api_version(model: str) -> str:
  """Returns the api version for the model spec."""
  model = model.removeprefix('apigee/')
  components = model.split('/')

  if len(components) == 3:
    # Format: <provider>/<version>/<model_id>
    return components[1]
  if len(components) == 2:
    # Format: <version>/<model_id> or <provider>/<model_id>
    # _validate_model_string ensures that if the first component is not a
    # provider, it can be a version.
    if components[0] not in ('vertex_ai', 'gemini') and components[
        0
    ].startswith('v'):
      return components[0]
  return ''


def _get_model_id(model: str) -> str:
  """Returns the model ID for the model spec."""
  model = model.removeprefix('apigee/')
  components = model.split('/')

  # Model_id is the last component in the model string.
  return components[-1]


def _parse_logprobs(
    logprobs_data: dict[str, Any] | None,
) -> types.LogprobsResult | None:
  """Parses OpenAI logprobs data into LogprobsResult."""
  if not logprobs_data or 'content' not in logprobs_data:
    return None

  chosen_candidates = []
  top_candidates = []

  for item in logprobs_data['content']:
    chosen_candidates.append(
        types.LogprobsResultCandidate(
            token=item.get('token'),
            log_probability=item.get('logprob'),
            # OpenAI text format usually doesn't expose ID easily here
            token_id=None,
        )
    )

    if 'top_logprobs' in item:
      current_top_candidates = []
      for top_item in item['top_logprobs']:
        current_top_candidates.append(
            types.LogprobsResultCandidate(
                token=top_item.get('token'),
                log_probability=top_item.get('logprob'),
                token_id=None,
            )
        )
      top_candidates.append(
          types.LogprobsResultTopCandidates(candidates=current_top_candidates)
      )

  return types.LogprobsResult(
      chosen_candidates=chosen_candidates, top_candidates=top_candidates
  )


def _validate_model_string(model: str) -> bool:
  """Validates the model string for Apigee LLM.

  The model string specifies the LLM provider (e.g., Vertex AI, Gemini), API
  version, and the model ID.

  Args:
    model: The model string. Supported format:
      `apigee/[<provider>/][<version>/]<model_id>`

  Returns:
    True if the model string is valid, False otherwise.
  """
  if not model.startswith('apigee/'):
    return False

  # Remove leading "apigee/" from the model string.
  model = model.removeprefix('apigee/')

  # The string has to be non-empty. i.e. the model_id cannot be empty.
  if not model:
    return False

  components = model.split('/')
  # If the model string has exactly 1 component, it means only the model_id is
  # present. This is a valid format.
  if len(components) == 1:
    return True

  # If the model string has more than 3 components, it is invalid.
  if len(components) > 3:
    return False

  # If the model string has 3 components, it means only the provider, version,
  # and model_id are present. This is a valid format.
  if len(components) == 3:
    # Format: <provider>/<version>/<model_id>
    if components[0] not in ('vertex_ai', 'gemini', 'openai'):
      return False
    if not components[1].startswith('v'):
      return False
    return True

  # If the model string has 2 components, it means either the provider or the
  # version (but not both), and model_id are present.
  if len(components) == 2:
    if components[0] in ['vertex_ai', 'gemini', 'openai']:
      return True
    if components[0].startswith('v'):
      return True
    return False

  return False


class CompletionsHTTPClient:
  """A generic HTTP client for completions, compatible with OpenAI API."""

  def __init__(
      self,
      base_url: str,
      headers: dict[str, str] | None = None,
      retry_options: Optional[types.HttpRetryOptions] = None,
  ):
    self._base_url = base_url
    self._headers = headers or {}
    self.retry_options = retry_options

  def __del__(self) -> None:
    self.close()

  @cached_property
  def _client(self) -> httpx.AsyncClient:
    """Provides the httpx client."""
    client = httpx.AsyncClient(
        base_url=self._base_url,
        headers=self._headers,
        timeout=None,
        follow_redirects=True,
    )
    atexit.register(self._cleanup_client, client)
    return client

  @staticmethod
  def _cleanup_client(client: httpx.AsyncClient) -> None:
    """Cleans up the httpx client."""
    if client.is_closed:
      return
    try:
      loop = asyncio.get_running_loop()
      loop.create_task(client.aclose())
    except RuntimeError:
      try:
        # This fails if asyncio.run is already called in main and is closing.
        asyncio.run(client.aclose())
      except RuntimeError:
        pass

  def close(self) -> None:
    if '_client' not in self.__dict__:
      return
    self._cleanup_client(self._client)

  async def aclose(self) -> None:
    if '_client' not in self.__dict__:
      return
    if self._client.is_closed:
      return
    await self._client.aclose()

  def _get_retry_kwargs(self) -> dict[str, Any]:
    """Returns the retry kwargs for tenacity."""
    if not self.retry_options:
      return {'stop': tenacity.stop_after_attempt(1), 'reraise': True}

    default_attempts = 5
    default_initial_delay = 1.0
    default_max_delay = 60.0
    default_exp_base = 2
    default_jitter = 1
    default_status_codes = (408, 429, 500, 502, 503, 504)

    opts = self.retry_options
    stop = tenacity.stop_after_attempt(
        opts.attempts if opts.attempts is not None else default_attempts
    )

    retriable_codes = (
        opts.http_status_codes
        if opts.http_status_codes is not None
        else default_status_codes
    )

    retry_network = tenacity.retry_if_exception_type(httpx.NetworkError)

    def is_retriable(e: Exception) -> bool:
      if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in retriable_codes
      return False

    retry_status = tenacity.retry_if_exception(is_retriable)

    wait = tenacity.wait_exponential_jitter(
        initial=(
            opts.initial_delay
            if opts.initial_delay is not None
            else default_initial_delay
        ),
        max=(
            opts.max_delay if opts.max_delay is not None else default_max_delay
        ),
        exp_base=(
            opts.exp_base if opts.exp_base is not None else default_exp_base
        ),
        jitter=opts.jitter if opts.jitter is not None else default_jitter,
    )

    return {
        'stop': stop,
        'retry': tenacity.retry_any(retry_network, retry_status),
        'reraise': True,
        'wait': wait,
    }

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates content using the OpenAI-compatible HTTP API."""
    payload = self._construct_payload(llm_request, stream)
    headers = self._headers.copy()
    headers['Content-Type'] = 'application/json'

    url = self._base_url
    if not url:
      raise ValueError('Base URL is not set.')

    if not url.endswith('/chat/completions'):
      url = f"{url.rstrip('/')}/chat/completions"

    if stream:
      async for stream_res in self._handle_streaming(url, payload, headers):
        yield stream_res
    else:
      response = await self._httpx_post_with_retry(url, payload, headers)
      data = response.json()
      yield self._parse_response(data)

  async def _httpx_post_with_retry(
      self, url: str, payload: dict[str, Any], headers: dict[str, str]
  ) -> httpx.Response:
    """Sends a POST request and handles retries."""
    retry_kwargs = self._get_retry_kwargs()
    async for attempt in tenacity.AsyncRetrying(**retry_kwargs):
      with attempt:
        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response

  async def _handle_streaming(
      self, url: str, payload: dict[str, Any], headers: dict[str, str]
  ) -> AsyncGenerator[LlmResponse, None]:
    """Handles streaming response from OpenAI-compatible API."""
    accumulator = ChatCompletionsResponseHandler()
    async with self._client.stream(
        'POST',
        url,
        json=payload,
        headers=headers,
    ) as resp:
      resp.raise_for_status()
      async for line in resp.aiter_lines():
        if not line:
          continue
        line = line.strip()
        if line.startswith('data:'):
          line = line.removeprefix('data:')
        line = line.lstrip()
        if line == '[DONE]':
          break
        try:
          for res in self._parse_streaming_line(line, accumulator):
            yield res
        except json.JSONDecodeError:
          logger.warning('Failed to parse JSON chunk: %s', line)
          continue

  def _construct_payload(
      self, llm_request: LlmRequest, stream: bool
  ) -> dict[str, Any]:
    """Constructs the payload from the LlmRequest."""
    messages = []
    if llm_request.config and llm_request.config.system_instruction:
      content = self._serialize_system_instruction(
          llm_request.config.system_instruction
      )
      if content:
        messages.append({
            'role': 'system',
            'content': content,
        })

    for content in llm_request.contents:
      messages += self._content_to_messages(content)

    payload = {
        'model': _get_model_id(llm_request.model),
        'messages': messages,
        'stream': stream,
    }

    if llm_request.config:
      self._map_config_parameters(llm_request.config, payload)
      self._map_tools(llm_request.config, payload)

    return payload

  def _map_config_parameters(
      self, config: types.GenerateContentConfig, payload: dict[str, Any]
  ) -> None:
    """Maps configuration parameters to the payload."""
    if config.temperature is not None:
      payload['temperature'] = config.temperature
    if config.top_p is not None:
      payload['top_p'] = config.top_p
    if config.max_output_tokens is not None:
      payload['max_tokens'] = config.max_output_tokens
    if config.stop_sequences:
      payload['stop'] = config.stop_sequences
    if config.frequency_penalty is not None:
      payload['frequency_penalty'] = config.frequency_penalty
    if config.presence_penalty is not None:
      payload['presence_penalty'] = config.presence_penalty
    if config.seed is not None:
      payload['seed'] = config.seed
    if config.candidate_count is not None:
      payload['n'] = config.candidate_count
    if config.response_logprobs:
      payload['logprobs'] = True
      if config.logprobs is not None:
        payload['top_logprobs'] = config.logprobs

    if config.response_json_schema:
      payload['response_format'] = {
          'type': 'json_schema',
          'json_schema': config.response_json_schema,
      }
    elif config.response_mime_type == 'application/json':
      payload['response_format'] = {'type': 'json_object'}

  def _map_tools(
      self, config: types.GenerateContentConfig, payload: dict[str, Any]
  ) -> None:
    """Maps tools and tool configuration to the payload."""
    if config.tools:
      tools = []
      for tool in config.tools:
        if tool.function_declarations:
          for func in tool.function_declarations:
            tools.append(self._function_declaration_to_tool(func))
      if tools:
        payload['tools'] = tools
        if config.tool_config and config.tool_config.function_calling_config:
          mode = config.tool_config.function_calling_config.mode
          if mode == types.FunctionCallingConfigMode.ANY:
            payload['tool_choice'] = 'required'
          elif mode == types.FunctionCallingConfigMode.NONE:
            payload['tool_choice'] = 'none'
          elif mode == types.FunctionCallingConfigMode.AUTO:
            payload['tool_choice'] = 'auto'

  def _content_to_messages(
      self, content: types.Content
  ) -> list[dict[str, Any]]:
    """Converts a Content object to /chat/completions messages."""
    role = content.role
    if role == 'model':
      role = 'assistant'

    tool_calls = []
    content_parts = []

    function_responses = []

    for part in content.parts or []:
      self._process_content_part(content, part, tool_calls, content_parts)
      if part.function_response:
        function_responses.append({
            'role': 'tool',
            'tool_call_id': part.function_response.id,
            'content': json.dumps(part.function_response.response),
        })
    if function_responses:
      return function_responses

    message = {'role': role}
    if tool_calls:
      message['tool_calls'] = tool_calls
      if not content_parts:
        message['content'] = None

    if content_parts:
      if len(content_parts) == 1 and content_parts[0]['type'] == 'text':
        message['content'] = content_parts[0]['text']
      else:
        message['content'] = content_parts
    return [message]

  def _process_content_part(
      self,
      content: types.Content,
      part: types.Part,
      tool_calls: list[dict[str, Any]],
      content_parts: list[dict[str, Any]],
  ) -> None:
    """Processes a single Part and updates tool_calls or content_parts."""
    if content.role != 'user' and (
        part.inline_data
        or (
            part.file_data
            and part.file_data.mime_type
            and part.file_data.mime_type.startswith('image')
        )
    ):
      logger.warning('Image data is not supported for assistant turns.')
      return

    if part.function_call:
      tool_call = {
          'id': part.function_call.id or 'call_' + part.function_call.name,
          'type': 'function',
          'function': {
              'name': part.function_call.name,
              'arguments': (
                  json.dumps(part.function_call.args)
                  if part.function_call.args
                  else '{}'
              ),
          },
      }
      if part.thought_signature:
        sig = part.thought_signature
        if isinstance(sig, bytes):
          sig = base64.b64encode(sig).decode('utf-8')
        tool_call['extra_content'] = {
            'google': {
                'thought_signature': sig,
            },
        }
      tool_calls.append(tool_call)
    elif part.function_response:
      # Handled in the loop to return immediately
      pass
    elif part.text:
      content_parts.append({'type': 'text', 'text': part.text})
    elif part.inline_data:
      mime_type = part.inline_data.mime_type
      data = base64.b64encode(part.inline_data.data).decode('utf-8')
      url = f'data:{mime_type};base64,{data}'
      content_parts.append({'type': 'image_url', 'image_url': {'url': url}})
    elif part.file_data:
      if part.file_data.file_uri:
        content_parts.append({
            'type': 'image_url',
            'image_url': {'url': part.file_data.file_uri},
        })
    elif part.executable_code:
      logger.warning(
          'Executable code is not supported in the standard Chat Completions'
          ' API.'
      )
    elif part.code_execution_result:
      logger.warning(
          'Code execution result is not supported in the standard Chat'
          ' Completions API.'
      )

  def _function_declaration_to_tool(
      self, func: types.FunctionDeclaration
  ) -> dict[str, Any]:
    """Converts a FunctionDeclaration to an OpenAI tool dictionary."""
    parameters = {}
    if func.parameters_json_schema:
      parameters = func.parameters_json_schema
    elif func.parameters:
      parameters = func.parameters.model_dump(exclude_none=True)

    return {
        'type': 'function',
        'function': {
            'name': func.name,
            'description': func.description,
            'parameters': parameters,
        },
    }

  def _serialize_system_instruction(
      self, system_instruction: Optional[types.ContentUnion]
  ) -> str | None:
    """Serializes system instruction to a string from ContentUnion type."""
    if not system_instruction:
      return None
    if isinstance(system_instruction, str):
      return system_instruction
    if isinstance(system_instruction, types.Part):
      return system_instruction.text
    if isinstance(system_instruction, types.Content):
      return ''.join(
          part.text for part in system_instruction.parts if part.text
      )
    if isinstance(system_instruction, dict):
      part = types.Part(**system_instruction)
      return part.text
    if isinstance(system_instruction, collections.abc.Iterable):
      parts = []
      for item in system_instruction:
        if isinstance(item, str):
          parts.append(types.Part(text=item))
        elif isinstance(item, types.Part):
          parts.append(item)
        elif isinstance(item, dict):
          parts.append(types.Part(**item))
      return ''.join(part.text for part in parts if part.text)
    return None

  def _parse_response(self, response: dict[str, Any]) -> LlmResponse:
    """Parses an OpenAI response dictionary into an LlmResponse."""
    handler = ChatCompletionsResponseHandler()
    return handler.process_response(response)

  def _parse_streaming_line(
      self,
      line: str,
      accumulator: ChatCompletionsResponseHandler,
  ) -> Generator[LlmResponse]:
    """Parses a single line from the streaming response.

    Args:
      line: A single line from the streaming response, expected to be a JSON
        string.
      accumulator: An accumulator to manage partial chat completion choices
        across multiple chunks.

    Yields:
      An LlmResponse object parsed from the streaming line.
    """
    chunk = json.loads(line)
    for response in accumulator.process_chunk(chunk):
      yield response


class ChatCompletionsResponseHandler:
  """Accumulates responses from the /chat/completions endpoint.

  Useful for both streaming and non-streaming responses.
  """

  def __init__(self):
    self.content_parts = ''
    self.tool_call_parts = {}
    self.role = ''
    self.streaming_complete = False
    self.model = ''
    self.usage = {}
    self.logprobs = {}
    self.custom_metadata = {}

  def process_response(self, response: dict[str, Any]) -> LlmResponse:
    """Processes a complete non-streaming response."""
    choices = response.get('choices', [])
    if not choices:
      raise ValueError('No choices found in response.')
    if len(choices) > 1:
      logging.error(
          'Multiple choices found in response but only the first one will be'
          ' used.'
      )
    choice = choices[0]
    message = choice.get('message', {})
    _, role = self._add_chat_completion_message(message)
    parts = self._get_content_parts()

    usage = response.get('usage', {})
    usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage.get('prompt_tokens', 0),
        candidates_token_count=usage.get('completion_tokens', 0),
        total_token_count=usage.get('total_tokens', 0),
    )
    logprobs_result = _parse_logprobs(choice.get('logprobs'))

    custom_metadata = {}
    for k in _CUSTOM_METADATA_FIELDS:
      v = response.get(k)
      if v is not None:
        custom_metadata[k] = v

    return LlmResponse(
        content=types.Content(role=role, parts=parts),
        usage_metadata=usage_metadata,
        finish_reason=self._map_finish_reason(choice.get('finish_reason')),
        logprobs_result=logprobs_result,
        model_version=response.get('model'),
        custom_metadata=custom_metadata,
    )

  def process_chunk(
      self, chunk: dict[str, Any]
  ) -> Generator[LlmResponse, None, None]:
    """Processes a chunk and yields responses."""
    if 'model' in chunk:
      self.model = chunk['model']
    if 'usage' in chunk and chunk['usage']:
      self.usage.update(chunk['usage'])

    for k in _CUSTOM_METADATA_FIELDS:
      v = chunk.get(k)
      if v is not None:
        self.custom_metadata[k] = v

    usage_metadata = None
    if self.usage:
      usage_metadata = types.GenerateContentResponseUsageMetadata(
          prompt_token_count=self.usage.get('prompt_tokens', 0),
          candidates_token_count=self.usage.get('completion_tokens', 0),
          total_token_count=self.usage.get('total_tokens', 0),
      )

    choices = chunk.get('choices')
    if not choices:
      # If no choices, but we have usage or other metadata updates, yield them.
      if usage_metadata or self.custom_metadata:
        yield LlmResponse(
            partial=True,
            model_version=self.model,
            usage_metadata=usage_metadata,
            custom_metadata=self.custom_metadata,
        )
      return

    if len(choices) > 1:
      logging.error(
          'Multiple choices found in streaming response but only the first one'
          ' will be used.'
      )
    choice = choices[0]

    # Accumulate logprobs if present
    if 'logprobs' in choice and choice['logprobs']:
      self._accumulate_logprobs(choice['logprobs'])

    logprobs_result = None
    if self.logprobs:
      logprobs_result = _parse_logprobs(self.logprobs)

    delta = choice.get('delta', {})
    partial_parts, role = self._add_chat_completion_chunk_delta(delta)

    yield LlmResponse(
        partial=True,
        content=types.Content(role=role, parts=partial_parts),
        model_version=self.model,
        usage_metadata=usage_metadata,
        custom_metadata=self.custom_metadata,
        logprobs_result=logprobs_result,
    )

    finish_reason = choice.get('finish_reason')
    if finish_reason:
      yield LlmResponse(
          content=types.Content(
              role=role,
              parts=self._get_content_parts(),
          ),
          finish_reason=self._map_finish_reason(finish_reason),
          custom_metadata=self.custom_metadata,
          model_version=self.model,
          usage_metadata=usage_metadata,
          logprobs_result=logprobs_result,
      )
      # Exit because the 'finish_reason' chunk is the final chunk.
      return

  def _map_finish_reason(self, reason: str | None) -> types.FinishReason:
    if reason == 'stop':
      return types.FinishReason.STOP
    if reason == 'length':
      return types.FinishReason.MAX_TOKENS
    if reason == 'tool_calls':
      return types.FinishReason.STOP
    if reason == 'content_filter':
      return types.FinishReason.SAFETY
    return types.FinishReason.FINISH_REASON_UNSPECIFIED

  def _accumulate_logprobs(self, logprobs_chunk: dict[str, Any]) -> None:
    """Accumulates logprobs from a chunk."""
    if not self.logprobs:
      self.logprobs = {'content': [], 'refusal': []}

    if 'content' in logprobs_chunk and logprobs_chunk['content']:
      if 'content' not in self.logprobs:
        self.logprobs['content'] = []
      self.logprobs['content'].extend(logprobs_chunk['content'])

    if 'refusal' in logprobs_chunk and logprobs_chunk['refusal']:
      if 'refusal' not in self.logprobs:
        self.logprobs['refusal'] = []
      self.logprobs['refusal'].extend(logprobs_chunk['refusal'])

  def _append_content(self, content: str, refusal: str) -> str:
    if content and refusal:
      content += '\n'
      content += refusal
    elif refusal:
      content = refusal
    if content:
      self.content_parts += content
    return content

  def _add_chat_completion_chunk_delta(
      self, delta: dict[str, Any]
  ) -> (list[types.Part], str):
    """Adds a chunk delta from a streaming chat completions response.

    This method processes a single delta chunk from a streaming chat completions
    response, accumulating partial content and tool calls.

    Args:
      delta: A dictionary representing a single delta from the streaming chat
        completions API.

    Returns:
      A tuple containing:
        - A list of `types.Part` objects representing the content and tool calls
          in this chunk.
        - The role associated with the message.
    """
    parts = []
    for tool_call in delta.get('tool_calls', []):
      chunk_part = self._upsert_tool_call(tool_call)
      parts.append(chunk_part)
    content = delta.get('content')
    refusal = delta.get('refusal')
    merged_content = self._append_content(content, refusal)
    if merged_content:
      parts.append(types.Part.from_text(text=merged_content))

    self._get_or_create_role(delta.get('role', 'model'))
    return parts, self.role

  def _add_chat_completion_message(
      self, message: dict[str, Any]
  ) -> (list[types.Part], str):
    """Adds a complete chat completion message to the accumulator.

    This method processes a single message from a non-streaming chat completions
    response, extracting and accumulating content and tool calls.

    Args:
      message: A dictionary representing a single message from the chat
        completions API.

    Returns:
      A tuple containing:
        - A list of `types.Part` objects representing the content and tool calls
          in this message.
        - The role associated with the message.
    """
    for tool_call in message.get('tool_calls', []):
      self._upsert_tool_call(tool_call)
    function_call = message.get('function_call')
    if function_call:
      # function_call is a single tool call and does not have an id.
      self._upsert_tool_call({
          'type': 'function',
          'function': function_call,
      })
    content = message.get('content')
    refusal = message.get('refusal')
    self._append_content(content, refusal)

    self._get_or_create_role(message.get('role', 'model'))
    return self._get_content_parts(), self.role

  def _get_content_parts(self) -> list[types.Part]:
    """Returns the content parts from the accumulated response."""
    parts = []
    if self.content_parts:
      parts.append(types.Part.from_text(text=self.content_parts))
    sorted_indices = sorted(self.tool_call_parts.keys())
    for index in sorted_indices:
      parts.append(self.tool_call_parts[index])
    return parts

  def _upsert_tool_call(self, tool_call: dict[str, Any]) -> types.Part:
    """Upserts a tool call into the accumulated tool call parts.

    This method handles partial tool call chunks in streaming responses by
    updating existing tool call parts or creating new ones.

    Args:
      tool_call: A dictionary representing a tool call or a delta of a tool call
        from the chat completions API.

    Returns:
      A `types.Part` object representing the updated or newly created tool call.
    """
    index = tool_call.get('index')
    if index is None:
      # If index is not provided, we might be in a non-streaming response.
      # We just append it as a new tool call.
      index = len(self.tool_call_parts)

    if index not in self.tool_call_parts:
      self.tool_call_parts[index] = types.Part(
          function_call=types.FunctionCall()
      )
    part = self.tool_call_parts[index]
    chunk_part = types.Part(function_call=types.FunctionCall())
    call_type = tool_call.get('type')
    # TODO: Add support for 'custom' type.
    if call_type is not None and call_type != 'function':
      raise ValueError(
          f'Unsupported tool_call type: {call_type} in call {tool_call}'
      )
    func = tool_call.get('function', {})
    args_delta = func.get('arguments', '')
    if args_delta:
      try:
        args = json.loads(args_delta)
        chunk_part.function_call.args = args
        if not part.function_call.args:
          part.function_call.args = dict(args)
        else:
          part.function_call.args.update(args)
      except json.JSONDecodeError as e:
        raise ValueError(f'Failed to parse arguments: {args_delta}') from e

    func_name = func.get('name')
    if func_name:
      part.function_call.name = func_name
      chunk_part.function_call.name = func_name
    tool_call_id = tool_call.get('id')
    if tool_call_id:
      part.function_call.id = tool_call_id
      chunk_part.function_call.id = tool_call_id

    # Add support for gemini's thought_signature.
    thought_signature = (
        tool_call.get('extra_content', {})
        .get('google', {})
        .get('thought_signature', '')
    )
    if thought_signature:
      if isinstance(thought_signature, str):
        thought_signature = base64.b64decode(thought_signature)
      part.thought_signature = thought_signature
      chunk_part.thought_signature = thought_signature
    return chunk_part

  def _get_or_create_role(self, role: str = '') -> str:
    if self.role:
      return self.role
    if role == 'assistant':
      role = 'model'
    self.role = role
    return self.role
