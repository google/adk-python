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

"""`CascadeLive`: a live-composing model built from a text LLM, STT and TTS."""

from __future__ import annotations

import contextlib
from typing import Any
from typing import AsyncGenerator
from typing import Optional
from typing import Union

from pydantic import PrivateAttr
from typing_extensions import override

from ..features import experimental
from ..features import FeatureName
from ..models._capabilities import LlmCapabilities
from ..models.base_llm import BaseLlm
from ..models.base_llm_connection import BaseLlmConnection
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ._cascade_live_connection import CascadeLiveConnection
from ._transforms import LiveEgress
from ._transforms import LiveIngress


@experimental(FeatureName.CASCADE_LIVE)
class CascadeLive(BaseLlm):
  """A text-reasoning model wrapped in speech-to-text and text-to-speech.

  Live composition is expressed through the agent's ``model`` attribute::

      LlmAgent(
          model=CascadeLive(
              model='gemini-3.5-flash',
              stt=ElevenLabsSTT(),
              tts=ElevenLabsTTS(),
          ),
          instruction='...',
          tools=[...],
      )

  Tools, instructions, and agent workflows apply unchanged because the
  underlying model is a standard text LLM. In non-live requests,
  `CascadeLive` delegates directly to the underlying model.
  """

  model: str
  """The model name of the underlying text LLM."""

  stt: LiveIngress
  """Audio to text transform (STT)."""

  tts: LiveEgress
  """Text to audio transform (TTS)."""

  # Holds the LLM once it exists: the instance passed to `__init__`, or
  # the one `_resolve_llm` builds from a name.
  _llm: Optional[BaseLlm] = PrivateAttr(default=None)

  def __init__(self, model: Union[str, BaseLlm], **data: Any) -> None:
    """Initializes CascadeLive with a text model name or instance, STT, and TTS."""
    if not model:
      raise ValueError('CascadeLive requires an LLM `model`.')
    if isinstance(model, BaseLlm):
      super().__init__(model=model.model, **data)
      self._llm = model
    else:
      super().__init__(model=model, **data)

  @property
  @override
  def capabilities(self) -> LlmCapabilities:
    """Returns the capabilities of the underlying LLM."""
    return self._resolve_llm().capabilities

  def _resolve_llm(self) -> BaseLlm:
    """Resolves and caches the inner text model."""
    if self._llm is not None:
      # An instance was passed to `__init__`, or a name was resolved earlier.
      return self._llm

    self._llm = LLMRegistry.new_llm(self.model)
    return self._llm

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Delegates to the underlying text model for non-live requests."""
    async for response in self._resolve_llm().generate_content_async(
        llm_request, stream=stream
    ):
      yield response

  @contextlib.asynccontextmanager
  async def connect(
      self, llm_request: LlmRequest
  ) -> AsyncGenerator[BaseLlmConnection, None]:
    """Opens a cascaded live connection using the configured STT and TTS transforms."""
    connection = CascadeLiveConnection(
        llm_request,
        self._resolve_llm(),
        self.stt,
        self.tts,
    )
    try:
      yield connection
    finally:
      await connection.close()
