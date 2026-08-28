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

"""Model Armor guardrail plugin.

Screens user input and model output with Google Cloud Model Armor in both
unary (``run_async``) and live (``run_live``) modes, through the ordinary
``before_model_callback`` and ``after_model_callback`` seams.

- input reaches ``before_model_callback`` as request content.
- output reaches ``after_model_callback`` as content parts in unary mode, and
  as an output transcription in live mode.
"""

from __future__ import annotations

import logging
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.api_core.gapic_v1.client_info import ClientInfo
from google.auth.credentials import Credentials
from google.genai import types

from ... import version
from ...agents.callback_context import CallbackContext
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...plugins.base_plugin import BasePlugin
from ._config import ModelArmorConfig

try:
  from google.cloud import modelarmor_v1 as modelarmor_v1
  from google.cloud.modelarmor_v1 import SanitizationResult
  from google.cloud.modelarmor_v1 import SanitizeModelResponseRequest
  from google.cloud.modelarmor_v1 import SanitizeModelResponseResponse
  from google.cloud.modelarmor_v1 import SanitizeUserPromptRequest
  from google.cloud.modelarmor_v1 import SanitizeUserPromptResponse
except ImportError as e:
  raise ImportError(
      'Model Armor support requires google-cloud-modelarmor. '
      "Install it with: pip install 'google-adk[gcp]'."
  ) from e

logger = logging.getLogger('google_adk.' + __name__)

USER_AGENT = f'adk-model-armor-plugin google-adk/{version.__version__}'


class ModelArmorPlugin(BasePlugin):
  """A plugin that screens input and output with Google Cloud Model Armor."""

  def __init__(
      self,
      *,
      config: ModelArmorConfig,
      name: str = 'model_armor_plugin',
      client: Optional[modelarmor_v1.ModelArmorAsyncClient] = None,
      credentials: Optional[Credentials] = None,
  ):
    """Initializes the Model Armor plugin.

    Args:
      config: The Model Armor configuration.
      name: A unique identifier for this plugin instance.
      client: An optional pre-constructed async client (e.g. for testing). If
        ``None``, one is lazily constructed on first use.
      credentials: Optional credentials used when constructing the client. If
        ``None``, Application Default Credentials are used.
    """
    super().__init__(name)
    self._config = config

    self._supplied_client = client
    self._client: Optional[modelarmor_v1.ModelArmorAsyncClient] = None
    self._credentials = credentials

    self._location = _shared_template_location(
        config.prompt_template_name, config.response_template_name
    )

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Screens user input text against the configured prompt template."""
    if not self._config.prompt_template_name:
      return None

    text = _extract_request_text(llm_request)
    if not text:
      return None

    try:
      result = await self._sanitize_user_prompt(
          text, self._config.prompt_template_name
      )
    except Exception:  # pylint: disable=broad-except
      logger.exception('Model Armor input screening call failed.')
      return self._handle_screening_failure(self._config.input_blocked_message)

    return self._handle_sanitization_result(
        result,
        direction='input',
        blocked_message=self._config.input_blocked_message,
    )

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    """Screens model output text against the configured response template."""
    if not self._config.response_template_name:
      return None

    text = _extract_response_text(llm_response)
    if not text:
      return None

    try:
      result = await self._sanitize_model_response(
          text, self._config.response_template_name
      )
    except Exception:  # pylint: disable=broad-except
      logger.exception('Model Armor output screening call failed.')
      return self._handle_screening_failure(self._config.output_blocked_message)

    return self._handle_sanitization_result(
        result,
        direction='output',
        blocked_message=self._config.output_blocked_message,
    )

  @property
  def client(self) -> modelarmor_v1.ModelArmorAsyncClient:
    """Returns the underlying async client, constructing it lazily.

    A ``grpc.aio`` channel binds to the event loop running when it is created.
    Plugins are usually built at import time with no loop, so the channel has
    to be created on first use to land on the loop that serves requests.
    """
    if self._supplied_client:
      return self._supplied_client
    if self._client is None:
      self._client = modelarmor_v1.ModelArmorAsyncClient(
          credentials=self._credentials,
          client_info=ClientInfo(user_agent=USER_AGENT),
          client_options=ClientOptions(
              api_endpoint=_regional_endpoint(self._location)
          ),
      )
    return self._client

  async def _sanitize_user_prompt(
      self, text: str, template_name: str
  ) -> SanitizationResult:
    """Screens user input text using the synchronous API."""
    request = SanitizeUserPromptRequest(
        name=template_name,
        user_prompt_data=modelarmor_v1.DataItem(text=text),
    )
    response: SanitizeUserPromptResponse = (
        await self.client.sanitize_user_prompt(request=request)
    )
    return response.sanitization_result

  async def _sanitize_model_response(
      self, text: str, template_name: str
  ) -> SanitizationResult:
    """Screens model output text using the synchronous API."""
    request = SanitizeModelResponseRequest(
        name=template_name,
        model_response_data=modelarmor_v1.DataItem(text=text),
    )
    response: SanitizeModelResponseResponse = (
        await self.client.sanitize_model_response(request=request)
    )
    return response.sanitization_result

  async def close(self) -> None:
    """Closes the underlying client."""
    if self._client:
      await self._client.transport.close()

  def _handle_sanitization_result(
      self,
      result: modelarmor_v1.SanitizationResult,
      *,
      direction: str,
      blocked_message: str,
  ) -> Optional[LlmResponse]:
    """Handles a Model Armor sanitization result."""
    if result.invocation_result != modelarmor_v1.InvocationResult.SUCCESS:
      logger.error(
          'Model Armor %s sanitization did not succeed: invocation_result=%r',
          direction,
          result.invocation_result,
      )
      return self._handle_screening_failure(blocked_message)

    if result.filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
      logger.warning('Model Armor %s sanitization match found.', direction)
      return self._blocked_response(blocked_message)

    return None

  def _handle_screening_failure(
      self, blocked_message: str
  ) -> Optional[LlmResponse]:
    """Handles a Model Armor screening failure or exception."""
    if self._config.block_on_screening_failure:
      return self._blocked_response(blocked_message)
    return None

  def _blocked_response(self, blocked_message: str) -> LlmResponse:
    """Builds the safe replacement response used when blocking."""
    return LlmResponse(
        content=types.Content(
            role='model',
            parts=[types.Part(text=blocked_message)],
        ),
        custom_metadata={'model_armor_blocked': True},
    )


def _extract_request_text(llm_request: LlmRequest) -> Optional[str]:
  """Extracts the latest user text from an LlmRequest."""
  if not llm_request.contents:
    return None
  # Screen the most recent user turn.
  for content in reversed(llm_request.contents):
    if content.role != 'user':
      continue
    text = _content_text(content)
    if text:
      return text
  return None


def _extract_response_text(llm_response: LlmResponse) -> Optional[str]:
  """Extracts screenable model output text from an LlmResponse.

  Prefers the output transcription for live bidi streaming, falls back to
  content parts for unary responses.
  """
  transcription = llm_response.output_transcription
  if transcription and transcription.text:
    return transcription.text
  if not llm_response.content:
    return None
  return _content_text(llm_response.content)


def _content_text(content: Optional[types.Content]) -> Optional[str]:
  """Joins all visible text parts of a Content into a single string."""
  if not content or not content.parts:
    return None
  texts = [
      part.text for part in content.parts if part.text and not part.thought
  ]
  if not texts:
    return None
  return '\n'.join(texts)


def _regional_endpoint(location: str) -> str:
  """Builds the Model Armor regional endpoint for a location."""
  return f'modelarmor.{location}.rep.googleapis.com'


def _shared_template_location(*template_names: Optional[str]) -> str:
  """Returns the location shared by the given templates.

  One client targets one regional endpoint, so every template it screens
  against has to live in the same location.

  Args:
    *template_names: Full template resource names. ``None`` entries are
      skipped, so a screening direction that is not configured can be passed
      through directly.

  Returns:
    The location the given templates share.

  Raises:
    ValueError: If a template name is not a full resource name, or the
      templates are in different locations.
  """
  locations: set[str] = set()
  for template in template_names:
    if not template:
      continue
    parsed = modelarmor_v1.ModelArmorAsyncClient.parse_template_path(template)
    if not parsed:
      raise ValueError(
          'Template names must be full resource names, '
          'projects/{project}/locations/{location}/templates/{template}; '
          f'got {template!r}.'
      )
    locations.add(str(parsed['location']))
  if len(locations) > 1:
    raise ValueError(
        f'Templates must be in the same location; got {sorted(locations)}.'
    )
  return locations.pop()
