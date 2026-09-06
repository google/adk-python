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

"""Companion plugin that screens tool output with Model Armor."""

from __future__ import annotations

import json
import logging
from typing import Any
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.api_core.gapic_v1.client_info import ClientInfo
from google.auth.credentials import Credentials

from google.adk import version
from google.adk.integrations.model_armor import ModelArmorConfig
from google.adk.integrations.model_armor._plugin import _regional_endpoint
from google.adk.integrations.model_armor._plugin import _shared_template_location
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

try:
  from google.cloud import modelarmor_v1 as modelarmor_v1
  from google.cloud.modelarmor_v1 import SanitizationResult
  from google.cloud.modelarmor_v1 import SanitizeUserPromptRequest
  from google.cloud.modelarmor_v1 import SanitizeUserPromptResponse
except ImportError as e:
  raise ImportError(
      'Model Armor support requires google-cloud-modelarmor. '
      "Install it with: pip install 'google-adk[gcp]'."
  ) from e

logger = logging.getLogger('google_adk.' + __name__)

USER_AGENT = (
    f'adk-model-armor-tool-output-sample google-adk/{version.__version__}'
)


def stringify_tool_result(result: dict[str, Any]) -> Optional[str]:
  """Serializes a tool result dict into text for Model Armor screening."""
  if not result:
    return None
  if len(result) == 1 and 'text' in result and result['text'] is not None:
    return str(result['text'])
  return json.dumps(result, default=str, sort_keys=True)


class ToolOutputModelArmorPlugin(BasePlugin):
  """Screens tool output via ``after_tool_callback`` using SanitizeUserPrompt."""

  def __init__(
      self,
      *,
      config: ModelArmorConfig,
      name: str = 'tool_output_model_armor_plugin',
      client: Optional[modelarmor_v1.ModelArmorAsyncClient] = None,
      credentials: Optional[Credentials] = None,
      tool_output_blocked_message: Optional[str] = None,
  ):
    super().__init__(name)
    if not config.prompt_template_name:
      raise ValueError(
          'prompt_template_name must be set to screen tool output with'
          ' SanitizeUserPrompt.'
      )
    self._config = config
    self._blocked_message = (
        tool_output_blocked_message or config.input_blocked_message
    )
    self._supplied_client = client
    self._client: Optional[modelarmor_v1.ModelArmorAsyncClient] = None
    self._credentials = credentials
    self._location = _shared_template_location(config.prompt_template_name)

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> Optional[dict[str, Any]]:
    """Screens tool output before it is returned to the agent."""
    del tool, tool_args, tool_context  # Unused; kept for callback signature parity.
    text = stringify_tool_result(result)
    if not text:
      return None

    try:
      sanitization_result = await self._sanitize_user_prompt(
          text, self._config.prompt_template_name
      )
    except Exception:  # pylint: disable=broad-except
      logger.exception('Model Armor tool-output screening call failed.')
      if self._config.block_on_screening_failure:
        return {'error': self._blocked_message}
      return None

    return self._handle_sanitization_result(sanitization_result)

  @property
  def client(self) -> modelarmor_v1.ModelArmorAsyncClient:
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
    request = SanitizeUserPromptRequest(
        name=template_name,
        user_prompt_data=modelarmor_v1.DataItem(text=text),
    )
    response: SanitizeUserPromptResponse = await self.client.sanitize_user_prompt(
        request=request
    )
    return response.sanitization_result

  def _handle_sanitization_result(
      self, result: SanitizationResult
  ) -> Optional[dict[str, Any]]:
    if result.invocation_result != modelarmor_v1.InvocationResult.SUCCESS:
      logger.error(
          'Model Armor tool-output sanitization did not succeed:'
          ' invocation_result=%r',
          result.invocation_result,
      )
      if self._config.block_on_screening_failure:
        return {'error': self._blocked_message}
      return None

    if result.filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
      logger.warning('Model Armor tool-output sanitization match found.')
      return {'error': self._blocked_message}

    return None

  async def close(self) -> None:
    if self._client:
      await self._client.transport.close()
