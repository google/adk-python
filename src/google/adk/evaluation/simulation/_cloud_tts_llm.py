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

"""BaseLlm adapter for Google Cloud Text-to-Speech.

Wraps the Cloud TTS API behind the standard ``BaseLlm`` interface.
Voice selection is read from ``LlmRequest.config.speech_config``; audio
is returned as ``inline_data`` in the ``LlmResponse``.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from typing import AsyncGenerator
from typing import Optional

import google.api_core.exceptions
from google.genai import types as genai_types
from pydantic import Field

from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse

logger = logging.getLogger("google_adk." + __name__)

# Mapping from Cloud TTS AudioEncoding names to MIME types.
_TTS_ENCODING_TO_MIME_TYPE = {
    "LINEAR16": "audio/l16",
    "MP3": "audio/mpeg",
    "OGG_OPUS": "audio/ogg",
    "MULAW": "audio/basic",
    "ALAW": "audio/alaw",
}


class _CloudTTSLlm(BaseLlm):
  """A BaseLlm that delegates to Google Cloud Text-to-Speech.

  Voice selection is read from GenerateContentConfig.speech_config.
  Extract voice_name and language_code from LlmRequest.config.speech_config.

    - language_code: (Default: "en-US") is a BCP-47 language tag
    (e.g. "en-US", "en-GB", "fr-FR", "de-DE", "ja-JP")
    - voice_name: (Default: "en-US-Studio-O") must be a Cloud TTS voice
    that matches the language code.
    Examples include:

    - ``"en-US-Studio-O"`` — US English, Studio quality
    - ``"en-US-Neural2-A"`` — US English, Neural2
    - ``"en-GB-Neural2-A"`` — British English, Neural2
    - ``"fr-FR-Neural2-A"`` — French, Neural2

    See https://cloud.google.com/text-to-speech/docs/voices for the full
    list of supported voices.

  TTS-specific parameters (encoding, speaking speed, pitch) are exposed
  as optional model-level fields.  The GCP project is read from the
  GOOGLE_CLOUD_PROJECT environment variable.
  """

  # -- TTS-specific fields (not available in GenerateContentConfig) ----------

  audio_encoding: str = Field(
      default="LINEAR16",
      description=(
          "Audio encoding format for TTS output. Supported values:"
          " LINEAR16, MP3, OGG_OPUS, MULAW, ALAW."
      ),
  )

  speaking_speed: Optional[float] = Field(
      default=1.0,
      description="Speaking speed multiplier (0.25–4.0). 1.0 is normal speed.",
  )

  pitch: Optional[float] = Field(
      default=0.0,
      description="Pitch adjustment in semitones (-20.0 to 20.0).",
  )

  # -- Internal state --------------------------------------------------------

  _tts_client: Any = None

  @classmethod
  def supported_models(cls) -> list[str]:  # pragma: no cover
    return [r"cloud_tts"]

  # -- Helpers ---------------------------------------------------------------

  @staticmethod
  def _extract_text(llm_request: LlmRequest) -> str:
    """Extract plain text from the request contents."""
    texts = []
    for content in llm_request.contents:
      if content.parts:
        for part in content.parts:
          if part.text:
            texts.append(part.text)
    if not texts:
      raise ValueError("_CloudTTSLlm requires text in LlmRequest.contents")
    return " ".join(texts)

  @staticmethod
  def _extract_voice_config(
      llm_request: LlmRequest,
  ) -> tuple[str, str]:
    """Extract voice config from LlmRequest.config.speech_config."""
    voice_name = "en-US-Studio-O"
    language_code = "en-US"

    config = llm_request.config
    if config and isinstance(config.speech_config, genai_types.SpeechConfig):
      sc = config.speech_config
      if sc.language_code:
        language_code = sc.language_code
      if (
          sc.voice_config
          and sc.voice_config.prebuilt_voice_config
          and sc.voice_config.prebuilt_voice_config.voice_name
      ):
        voice_name = sc.voice_config.prebuilt_voice_config.voice_name

    return voice_name, language_code

  # -- BaseLlm interface -----------------------------------------------------

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Synthesize speech from text via the Cloud TTS API.

    Args:
      llm_request: Request containing text contents and optional speech_config
        for voice selection.
      stream: Ignored — TTS always returns a single response.

    Yields:
      A single ``LlmResponse`` with audio data in ``inline_data``.
    """
    # Lazy imports to avoid mandatory dependency when TTS is not used.
    from google.cloud.texttospeech_v1 import TextToSpeechAsyncClient
    from google.cloud.texttospeech_v1.types import cloud_tts

    # Initialise client lazily.
    if self._tts_client is None:
      project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", None)
      if project_id:
        from google.api_core import client_options as client_options_lib

        opts = client_options_lib.ClientOptions(
            quota_project_id=project_id,
        )
        self._tts_client = TextToSpeechAsyncClient(client_options=opts)
      else:
        self._tts_client = TextToSpeechAsyncClient()

    # Extract text and voice config from the request.
    text = self._extract_text(llm_request)
    voice_name, language_code = self._extract_voice_config(llm_request)

    # Map encoding string to the Cloud TTS enum.
    try:
      audio_encoding_enum = cloud_tts.AudioEncoding[self.audio_encoding]
    except KeyError as exc:
      raise ValueError(
          f"Unsupported audio_encoding: '{self.audio_encoding}'."
          f" Supported: {[e.name for e in cloud_tts.AudioEncoding]}"
      ) from exc

    # Build voice selection params.
    voice_params = cloud_tts.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
    )

    # Build audio config with optional rate/pitch overrides.
    audio_config_kwargs: dict[str, Any] = {
        "audio_encoding": audio_encoding_enum,
    }
    if self.speaking_speed is not None:
      audio_config_kwargs["speaking_rate"] = self.speaking_speed
    if self.pitch is not None:
      audio_config_kwargs["pitch"] = self.pitch
    audio_config = cloud_tts.AudioConfig(**audio_config_kwargs)

    # Call the Cloud TTS API.
    try:
      tts_response = await self._tts_client.synthesize_speech(
          input=cloud_tts.SynthesisInput(text=text),
          voice=voice_params,
          audio_config=audio_config,
      )
    except google.api_core.exceptions.GoogleAPICallError as e:
      logger.error("Cloud TTS synthesis failed: %s", e)
      yield LlmResponse(
          error_code="TTS_SYNTHESIS_FAILED",
          error_message=str(e),
      )
      return

    mime_type = _TTS_ENCODING_TO_MIME_TYPE.get(self.audio_encoding, "audio/l16")
    logger.info(
        "Cloud TTS synthesis completed: %d bytes of %s audio",
        len(tts_response.audio_content),
        self.audio_encoding,
    )

    yield LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    inline_data=genai_types.Blob(
                        mime_type=mime_type,
                        data=tts_response.audio_content,
                    )
                )
            ],
        ),
    )
