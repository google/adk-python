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

import logging
from typing import Optional
import uuid

from typing_extensions import override

from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..plugins.base_plugin import BasePlugin

logger = logging.getLogger("google_adk." + __name__)

_LLM_REQUEST_ID_KEY = "__llm_request_key__"


class _RequestIntercepterPlugin(BasePlugin):
  """A plugin that intercepts requests that are made to the model and couples them with the model response.

  NOTE: This implementation is intended for eval systems internal usage. Do not
  take direct dependency on it.

  Context behind the creation of this intercepter:
  Some of the newer AutoRater backed metrics need access the pieces of
  information that were presented to the model like instructions and the list
  of available tools.

  We intercept the llm_request using this intercepter and make it available to
  eval system.

  How is it done?
  The class maintains a cache of llm_requests that pass through it. Each request
  is given a unique id. The id is put in custom_metadata field of the response.
  Eval systems have access to the response and can use the request id to
  get the llm_request.
  """

  def __init__(self, name: str):
    super().__init__(name=name)
    self._llm_requests_cache: dict[str, LlmRequest] = {}

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    # We add the llm_request to the call back context so that we can fetch
    # it later.
    request_id = str(uuid.uuid4())
    self._llm_requests_cache[request_id] = llm_request
    callback_context.state[_LLM_REQUEST_ID_KEY] = request_id

  @override
  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    # Fetch the request_id from the callback_context
    if callback_context and _LLM_REQUEST_ID_KEY in callback_context.state:
      if llm_response.custom_metadata is None:
        llm_response.custom_metadata = {}

      llm_response.custom_metadata[_LLM_REQUEST_ID_KEY] = (
          callback_context.state[_LLM_REQUEST_ID_KEY]
      )

  def get_model_request(
      self, llm_response: LlmResponse
  ) -> Optional[LlmRequest]:
    """Fetches the request object, if found."""
    if (
        llm_response.custom_metadata
        and _LLM_REQUEST_ID_KEY in llm_response.custom_metadata
    ):
      request_id = llm_response.custom_metadata[_LLM_REQUEST_ID_KEY]

      if request_id in self._llm_requests_cache:
        return self._llm_requests_cache[request_id]
      else:
        logger.warning("`%s` not found in llm_request_cache.", request_id)
