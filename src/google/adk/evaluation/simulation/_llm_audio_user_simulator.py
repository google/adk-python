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

"""LLM-backed audio user simulator.

This module provides ``_LlmAudioUserSimulator``, a ``UserSimulator`` that
generates audio user messages.

Text generation is delegated to an internal ``LlmBackedUserSimulator``
(used as a black-box via composition).  The text output is then fed to a
second ``BaseLlm`` (resolved from ``audio_model``) to produce audio bytes.

The simulator is **agnostic to the audio provider**.  Both built-in
providers and arbitrary model names are supported; voice configuration
uses the standard ``GenerateContentConfig.speech_config``.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from google.genai import types as genai_types
from pydantic import Field
from pydantic import field_validator
from typing_extensions import Literal
from typing_extensions import override

from ...events.event import Event
from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ...models.registry import LLMRegistry
from ...utils.context_utils import Aclosing
from ...utils.feature_decorator import experimental
from .._retry_options_utils import add_default_retry_options_if_not_present
from ..evaluator import Evaluator
from .llm_backed_user_simulator_prompts import is_valid_user_simulator_template
from .user_simulator import BaseUserSimulatorConfig
from .user_simulator import NextUserMessage
from .user_simulator import Status
from .user_simulator import UserSimulator

logger = logging.getLogger("google_adk." + __name__)

_AUTHOR_USER = "user"
_DEFAULT_MODEL = "cloud_tts"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class LlmAudioUserSimulatorConfig(BaseUserSimulatorConfig):
  """Configuration for _LlmAudioUserSimulator."""

  type: Literal["llm_audio"] = Field(
      default="llm_audio",
      description=(
          "Discriminator tag for this config subclass. See"
          " `BaseUserSimulatorConfig.type` for the rationale."
      ),
  )

  # --- Text-generation LLM fields ---

  model: str = Field(
      default="gemini-2.5-flash",
      description="The model to use for user simulation text generation.",
  )

  model_configuration: genai_types.GenerateContentConfig = Field(
      default_factory=lambda: genai_types.GenerateContentConfig(
          thinking_config=genai_types.ThinkingConfig(
              include_thoughts=True,
              thinking_budget=10240,
          )
      ),
      description="The configuration for the text-generation model.",
  )

  max_allowed_invocations: int = Field(
      default=20,
      description="""Maximum number of invocations allowed by the simulated
interaction. This property allows us to stop a run-off conversation, where the
agent and the user simulator get into a never ending loop. The initial fixed
prompt is also counted as an invocation.

(Not recommended) If you don't want a limit, you can set the value to -1.""",
  )

  custom_instructions: str | None = Field(
      default=None,
      description="""Custom instructions for the user simulator. The
instructions must contain the following formatting placeholders following Jinja syntax:
* {{ stop_signal }} : text to be generated when the user simulator decides that the
  conversation is over.
* {{ conversation_plan }} : the overall plan for the conversation that the user
  simulator must follow.
* {{ conversation_history }} : the conversation between the user and the agent so
  far.
* {{ persona }} : Only needed if specifying user_persona in the conversation scenario.
""",
  )

  include_function_calls: bool = Field(
      default=False,
      description="""Whether to include function calls and responses in the
conversation history prompt provided to the user simulator.""",
  )

  @field_validator("custom_instructions")
  @classmethod
  def validate_custom_instructions(cls, value: str | None) -> str | None:
    if value is None:
      return value
    if not is_valid_user_simulator_template(
        value,
        required_params=[
            "stop_signal",
            "conversation_plan",
            "conversation_history",
        ],
    ):
      raise ValueError(
          "custom_instructions must contain each of the following formatting"
          " placeholders using Jinja syntax: {{ stop_signal }}, {{"
          " conversation_plan }}, {{ conversation_history }}"
      )
    return value

  # --- Audio-generation fields ---

  audio_model: str = Field(
      default=_DEFAULT_MODEL,
      description=(
          "Model name for audio generation.  Use 'cloud_tts' for Google"
          " Cloud Text-to-Speech (default), or a model name string"
          " (e.g. 'gemini-2.5-flash-preview-tts') for Gemini TTS model."
      ),
  )

  audio_model_configuration: genai_types.GenerateContentConfig = Field(
      default_factory=lambda: genai_types.GenerateContentConfig(
          speech_config=genai_types.SpeechConfig(
              voice_config=genai_types.VoiceConfig(
                  prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                      voice_name="en-US-Studio-O",
                  )
              ),
              language_code="en-US",
          ),
      ),
      description=(
          "Configuration for the audio model.  Voice selection uses"
          " speech_config.  For native model audio, additionally set"
          " response_modalities=['AUDIO']."
      ),
  )

  include_text_with_audio: bool = Field(
      default=True,
      description=(
          "Whether to include the text part alongside the audio part in"
          " the generated Content.  If True, the Content will have both a"
          " text Part and an inline_data audio Part.  If False, only the"
          " audio Part is included."
      ),
  )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


@experimental
class _LlmAudioUserSimulator(UserSimulator):
  """A UserSimulator that generates *audio* user messages.

  Acts as a **decorator** over a text-producing ``UserSimulator``
  (the ``text_simulator``): it consumes that simulator's
  ``get_next_user_message`` output (text) and feeds the text to a second
  ``BaseLlm`` (resolved from ``audio_model``) to produce audio bytes. The
  wrapped simulator is treated as a black-box — only its public
  ``get_next_user_message`` output is consumed — so it may be any
  ``UserSimulator``, e.g. an ``LlmBackedUserSimulator`` (scenario-driven) or
  a ``StaticUserSimulator`` (pre-authored turns).

  Pipeline:

  1. Delegate to the wrapped ``text_simulator`` to generate the next
     conversational turn as **text**.
  2. Feed the text to the audio ``BaseLlm`` to produce audio bytes.
  3. Return a ``Content`` with text+audio or audio-only parts.

  Non-SUCCESS results (e.g. ``STOP_SIGNAL_DETECTED``, ``TURN_LIMIT_REACHED``)
  and SUCCESS results carrying no text are passed through unchanged.

  The simulator is **provider-agnostic** — switching audio backends is a
  config-only change.
  """

  config_type: ClassVar[type[LlmAudioUserSimulatorConfig]] = (
      LlmAudioUserSimulatorConfig
  )

  # Narrow the inherited ``_config`` attribute (populated by the base
  # ``UserSimulator.__init__`` via ``config_type.model_validate``) to this
  # simulator's concrete config type so static type checkers can see the
  # audio-specific fields.
  _config: LlmAudioUserSimulatorConfig

  def __init__(
      self,
      *,
      config: BaseUserSimulatorConfig,
      text_simulator: UserSimulator,
  ):
    """Initializes the audio simulator as a decorator over a text simulator.

    Args:
      config: The audio simulator config (``LlmAudioUserSimulatorConfig``).
      text_simulator: The wrapped ``UserSimulator`` that produces the text
        turns to be converted to audio. May be any ``UserSimulator`` (e.g.
        ``LlmBackedUserSimulator`` for scenarios, or ``StaticUserSimulator``
        for pre-authored turns).
    """
    super().__init__(config, config_type=self.config_type)

    # The wrapped text-producing simulator. Only its public
    # `get_next_user_message` output is consumed.
    self._text_simulator = text_simulator

    # Audio-generation LLM.
    llm_registry = LLMRegistry()
    self._audio_llm = self._resolve_audio_llm(
        self._config.audio_model, llm_registry
    )

  @staticmethod
  def _resolve_audio_llm(
      audio_model: str, llm_registry: LLMRegistry
  ) -> BaseLlm:
    """Resolve the audio model to a ``BaseLlm`` instance."""
    audio_llm_class = llm_registry.resolve(audio_model)
    return audio_llm_class(model=audio_model)

  # -----------------------------------------------------------------
  # Main entry point (override)
  # -----------------------------------------------------------------

  @override
  async def get_next_user_message(
      self,
      events: list[Event],
  ) -> NextUserMessage:
    """Returns the next user message (with audio) to send to the agent.

    Delegates text generation to the wrapped ``text_simulator``, then
    converts the text to audio.

    Args:
      events: The unaltered conversation history between the user and the
        agent(s) under evaluation.

    Returns:
      A NextUserMessage containing text and/or audio, or a status
      indicating why no message was generated.

    Raises:
      RuntimeError: If audio generation fails.
    """
    # Delegate text generation to the wrapped text simulator.
    text_result = await self._text_simulator.get_next_user_message(events)

    # If the text simulator didn't produce a successful message, pass
    # through the result unchanged (e.g. TURN_LIMIT_REACHED, STOP_SIGNAL).
    if text_result.status != Status.SUCCESS:
      return text_result

    # Extract text from the result.
    text = ""
    if text_result.user_message and text_result.user_message.parts:
      for part in text_result.user_message.parts:
        if part.text:
          text += part.text

    if not text:
      return text_result

    return NextUserMessage(
        status=Status.SUCCESS,
        user_message=await self.to_audio_content(text),
    )

  async def to_audio_content(self, text: str) -> genai_types.Content:
    """Convert ``text`` into a user ``Content`` carrying audio.

    This is the single, reusable audio-generation entry point. Both this
    simulator's own ``get_next_user_message`` and ``StaticUserSimulator``
    (for pre-authored text turns) route through here, so audio generation
    lives in exactly one place.

    Args:
      text: The text to convert to audio.

    Returns:
      A ``Content`` with role ``user``. When ``include_text_with_audio`` is
      ``True`` the content has a text ``Part`` followed by an audio
      ``inline_data`` ``Part``; otherwise it has only the audio ``Part``.

    Raises:
      RuntimeError: If audio generation fails.
    """
    parts = []

    # Optionally include the raw text part.
    if self._config.include_text_with_audio:
      parts.append(genai_types.Part(text=text))

    # Generate audio via the audio LLM (provider-agnostic).
    audio_bytes, mime_type = await self._generate_audio(text)
    parts.append(
        genai_types.Part(
            inline_data=genai_types.Blob(
                mime_type=mime_type,
                data=audio_bytes,
            )
        )
    )

    return genai_types.Content(parts=parts, role=_AUTHOR_USER)

  # -----------------------------------------------------------------
  # Audio generation (provider-agnostic)
  # -----------------------------------------------------------------

  async def _generate_audio(self, text: str) -> tuple[bytes, str]:
    """Generate audio from text using the configured audio model.

    Args:
      text: The text to convert to audio.

    Returns:
      A tuple of (audio_bytes, mime_type).

    Raises:
      RuntimeError: If audio generation fails.
    """
    llm_request = LlmRequest(
        model=self._config.audio_model,
        config=self._config.audio_model_configuration,
        contents=[
            genai_types.Content(
                parts=[genai_types.Part(text=text)],
                role=_AUTHOR_USER,
            ),
        ],
    )
    add_default_retry_options_if_not_present(llm_request)

    audio_bytes = b""
    mime_type = "audio/pcm"
    async with Aclosing(
        self._audio_llm.generate_content_async(llm_request)
    ) as agen:
      async for llm_response in agen:
        if llm_response.error_code:
          raise RuntimeError(
              f"Audio generation failed: {llm_response.error_code}"
              f" — {llm_response.error_message}"
          )

        if (
            llm_response.content
            and hasattr(llm_response.content, "parts")
            and llm_response.content.parts
        ):
          for part in llm_response.content.parts:
            if part.inline_data and part.inline_data.data:
              audio_bytes += part.inline_data.data
              if part.inline_data.mime_type:
                mime_type = part.inline_data.mime_type

    if not audio_bytes:
      raise RuntimeError("Audio model returned no audio data")

    return audio_bytes, mime_type

  # -----------------------------------------------------------------
  # Evaluator hook
  # -----------------------------------------------------------------

  @override
  def get_simulation_evaluator(
      self,
  ) -> Evaluator | None:
    """Returns an Evaluator that evaluates if the simulation was successful or not."""
    raise NotImplementedError()
