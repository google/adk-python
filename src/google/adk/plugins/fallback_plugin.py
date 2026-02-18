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

import logging
from typing import Optional

from opentelemetry import trace

from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..version import __version__
from .base_plugin import BasePlugin

logger: logging.Logger = logging.getLogger("google_adk." + __name__)
tracer = trace.get_tracer("google.adk.plugins.fallback_plugin", __version__)

_FALLBACK_ATTEMPTS_MAX_SIZE = 100
_FALLBACK_ATTEMPTS_PRUNE_COUNT = 50


class FallbackPlugin(BasePlugin):
  """Plugin that implements transparent model fallback on specific HTTP errors.

  This plugin intercepts LLM requests and responses to provide automatic model
  fallback when the primary model returns a configured error code (e.g., rate
  limit or timeout). Fallback is **non-persistent**: every new request always
  starts with the ``root_model``; only that particular request may be retried
  with the ``fallback_model``.

  The plugin itself does not re-issue the request. The actual retry must be
  handled by the underlying model implementation (e.g. LiteLlm's ``fallbacks``
  parameter). This plugin is responsible for:

  - Resetting the model to ``root_model`` at the start of each request so that
    fallback state does not leak across turns.
  - Detecting error responses whose ``error_code`` is in ``error_status`` and
    annotating the ``LlmResponse`` with structured fallback metadata so that
    the caller or the model layer can take remedial action.
  - Tracking the number of fallback attempts per request context and
    pruning the tracking dictionary to avoid unbounded memory growth.

  Example:
      >>> from google.adk.plugins.fallback_plugin import FallbackPlugin
      >>> fallback_plugin = FallbackPlugin(
      ...     root_model="gemini-2.0-flash",
      ...     fallback_model="gemini-1.5-pro",
      ...     error_status=[429, 504],
      ... )
      >>> runner = Runner(
      ...     agents=[my_agent],
      ...     plugins=[fallback_plugin],
      ... )
  """

  def __init__(
      self,
      name: str = "fallback_plugin",
      root_model: Optional[str] = None,
      fallback_model: Optional[str] = None,
      error_status: Optional[list[int]] = None,  # noqa: B006
  ) -> None:
    """Initializes the FallbackPlugin.

    Args:
      name: The name of the plugin. Defaults to ``"fallback_plugin"``.
      root_model: The primary model identifier that every request should start
        with. When ``None`` the plugin does not override the model set on the
        request.
      fallback_model: The backup model identifier to record in the response
        metadata when an error matching ``error_status`` is detected. When
        ``None`` the plugin logs a warning but does not write any metadata.
      error_status: A list of HTTP-style numeric status codes that should be
        treated as retriable failures and trigger fallback tracking. Defaults
        to ``[429, 504]``.
    """
    super().__init__(name=name)
    self.root_model = root_model
    self.fallback_model = fallback_model
    self.error_status = error_status if error_status is not None else [429, 504]
    self._error_status_set = {str(s) for s in self.error_status}

    # Maps id(callback_context) -> number of fallback attempts for that context.
    self._fallback_attempts: dict[int, int] = {}

  async def before_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
  ) -> Optional[LlmResponse]:
    """Resets the request model to ``root_model`` before each LLM call.

    This callback is invoked before every LLM request. It ensures non-persistent
    fallback behaviour by unconditionally resetting the model to ``root_model``
    whenever no fallback attempt is currently in progress for this context,
    so that a fallback from a previous turn cannot bleed into a new one.

    Args:
      callback_context: The context for the current agent call. Used as the key
        for tracking per-request fallback state.
      llm_request: The prepared request object about to be sent to the model.
        Its ``model`` field may be mutated to enforce the ``root_model``.

    Returns:
      ``None`` always, so that normal LLM processing continues.
    """
    context_id = id(callback_context)

    # Initialise the attempt counter for this context on first contact.
    if context_id not in self._fallback_attempts:
      self._fallback_attempts[context_id] = 0

    # Only reset to root_model when we are NOT mid-fallback.
    if self.root_model and self._fallback_attempts.get(context_id, 0) == 0:
      if hasattr(llm_request, "model") and llm_request.model != self.root_model:
        logger.info(
            "Resetting model from %s to root model: %s",
            llm_request.model,
            self.root_model,
        )
        llm_request.model = self.root_model

    return await super().before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )

  async def after_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_response: LlmResponse,
  ) -> Optional[LlmResponse]:
    """Detects retriable errors and annotates the response with fallback metadata.

    This callback is invoked after every LLM response. When the response
    carries an ``error_code`` that matches one of the configured ``error_status``
    codes **and** a ``fallback_model`` is configured, the plugin writes the
    following keys into ``llm_response.custom_metadata``:

    - ``fallback_triggered`` (``bool``): Always ``True``.
    - ``original_model`` (``str``): The value of ``root_model``.
    - ``fallback_model`` (``str``): The value of ``fallback_model``.
    - ``fallback_attempt`` (``int``): The cumulative attempt count for this
      context.
    - ``error_code`` (``str``): The string representation of the error code.

    The tracking dictionary is pruned to at most 50 entries whenever its size
    exceeds 100, to prevent unbounded memory growth in long-running processes.

    Args:
      callback_context: The context for the current agent call. Used as the key
        for tracking per-request fallback state.
      llm_response: The response received from the model. Its
        ``custom_metadata`` field may be populated with fallback tracking data.

    Returns:
      ``None`` always, so that normal post-model processing continues.
    """
    context_id = id(callback_context)

    if llm_response.error_code and str(llm_response.error_code) in self._error_status_set:
      logger.warning(
          "Model call failed with error code %s. Error message: %s",
          llm_response.error_code,
          llm_response.error_message,
      )

      self._fallback_attempts[context_id] = (
          self._fallback_attempts.get(context_id, 0) + 1
      )

      if self.fallback_model:
        logger.info(
            "Fallback triggered: %s -> %s (attempt %d)",
            self.root_model,
            self.fallback_model,
            self._fallback_attempts[context_id],
        )
        if not llm_response.custom_metadata:
          llm_response.custom_metadata = {}
        llm_response.custom_metadata["fallback_triggered"] = True
        llm_response.custom_metadata["original_model"] = self.root_model
        llm_response.custom_metadata["fallback_model"] = self.fallback_model
        llm_response.custom_metadata["fallback_attempt"] = (
            self._fallback_attempts[context_id]
        )
        llm_response.custom_metadata["error_code"] = str(llm_response.error_code)
      else:
        logger.warning("No fallback model configured, cannot retry.")

    # Prune the tracking dict to avoid unbounded memory growth.
    if len(self._fallback_attempts) > _FALLBACK_ATTEMPTS_MAX_SIZE:
      oldest_keys = list(self._fallback_attempts.keys())[:_FALLBACK_ATTEMPTS_PRUNE_COUNT]
      for key in oldest_keys:
        del self._fallback_attempts[key]

    return await super().after_model_callback(
        callback_context=callback_context, llm_response=llm_response
    )
