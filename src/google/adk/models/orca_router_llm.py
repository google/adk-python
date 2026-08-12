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

"""OrcaRouter integration for Anthropic-compatible models.

OrcaRouter (https://www.orcarouter.ai) exposes an Anthropic-compatible
``/v1/messages`` endpoint at ``https://api.orcarouter.ai``. Models are
addressed with the ``anthropic/`` namespace, e.g.
``anthropic/claude-haiku-4.5``. This class mirrors the ``Claude`` subclass of
``AnthropicLlm``: it only overrides client construction so requests go to
OrcaRouter with an ``ORCAROUTER_API_KEY`` instead of the Anthropic API.
"""

from __future__ import annotations

from functools import cached_property
import os
from typing import Optional

from anthropic import AsyncAnthropic
from typing_extensions import override

from ..utils._google_client_headers import get_tracking_headers
from .anthropic_llm import AnthropicLlm

__all__ = ["OrcaRouterLlm"]

_ORCAROUTER_BASE_URL = "https://api.orcarouter.ai"
_ORCAROUTER_DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


class OrcaRouterLlm(AnthropicLlm):
  """Integration with Anthropic-compatible models served from OrcaRouter.

  OrcaRouter (https://www.orcarouter.ai) is an Anthropic-compatible API
  provider. Requests are sent to ``https://api.orcarouter.ai`` and model names
  must carry the ``anthropic/`` namespace (e.g.
  ``anthropic/claude-haiku-4.5``), which is the default model.

  Credentials are read from the ``ORCAROUTER_API_KEY`` environment variable.
  Keys are issued by OrcaRouter and start with the ``sk-orca-`` prefix.

  Attributes:
    model: The name of the OrcaRouter-hosted model.
    max_tokens: The maximum number of tokens to generate.
  """

  model: str = _ORCAROUTER_DEFAULT_MODEL

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return [r"orcarouter/.*"]

  def _resolve_model_name(self, model: Optional[str]) -> str:
    if not model:
      return self.model
    if model.startswith("orcarouter/"):
      return model[len("orcarouter/") :]
    return model

  @cached_property
  @override
  def _anthropic_client(self) -> AsyncAnthropic:
    api_key = os.environ.get("ORCAROUTER_API_KEY")
    if not api_key:
      raise ValueError(
          "No OrcaRouter credential was found for calling Anthropic-compatible"
          " models through OrcaRouter. Set ORCAROUTER_API_KEY to a key from"
          " OrcaRouter, e.g. `export ORCAROUTER_API_KEY=<your-key>`."
      )
    return AsyncAnthropic(
        api_key=api_key,
        base_url=_ORCAROUTER_BASE_URL,
        default_headers=get_tracking_headers(),
    )
