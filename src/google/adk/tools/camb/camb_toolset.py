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

"""CAMB AI Toolset for Google ADK.

Provides audio and speech tools powered by camb.ai, including text-to-speech,
translation, transcription, translated TTS, voice cloning, voice listing,
text-to-sound generation, and audio separation.
"""

from __future__ import annotations

from typing import List
from typing import Optional
from typing import Union

from typing_extensions import override

from ..base_tool import BaseTool
from ..base_toolset import BaseToolset
from ..base_toolset import ToolPredicate
from ..function_tool import FunctionTool
from ._helpers import CambHelpers
from ._tools import make_audio_separation_func
from ._tools import make_clone_voice_func
from ._tools import make_list_voices_func
from ._tools import make_text_to_sound_func
from ._tools import make_transcribe_func
from ._tools import make_translate_func
from ._tools import make_translated_tts_func
from ._tools import make_tts_func

try:
  from ...agents.readonly_context import ReadonlyContext
except ImportError:  # pragma: no cover
  ReadonlyContext = None  # type: ignore[assignment,misc]


class CambAIToolset(BaseToolset):
  """Toolset that exposes camb.ai audio/speech services as ADK tools.

  Each enabled service is wrapped in a :class:`FunctionTool` so that ADK
  agents can call it.  The underlying ``camb`` SDK is imported lazily;
  installing the ``camb`` extra is only required when the toolset is used.

  Example::

      from google.adk.tools.camb import CambAIToolset

      toolset = CambAIToolset(api_key="your-key")
      agent = Agent(
          name="audio_agent",
          model="gemini-2.0-flash",
          tools=[toolset],
      )

  Args:
    api_key: camb.ai API key.  Falls back to ``CAMB_API_KEY`` env var.
    timeout: Request timeout in seconds.
    max_poll_attempts: Maximum polling iterations for async tasks.
    poll_interval: Seconds between polling attempts.
    tool_filter: Optional filter to select a subset of tools.
    include_tts: Include the text-to-speech tool.
    include_translation: Include the translation tool.
    include_transcription: Include the transcription tool.
    include_translated_tts: Include the translated TTS tool.
    include_voice_clone: Include the voice cloning tool.
    include_voice_list: Include the voice listing tool.
    include_text_to_sound: Include the text-to-sound tool.
    include_audio_separation: Include the audio separation tool.
  """

  def __init__(
      self,
      *,
      api_key: Optional[str] = None,
      timeout: float = 60.0,
      max_poll_attempts: int = 60,
      poll_interval: float = 2.0,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      include_tts: bool = True,
      include_translation: bool = True,
      include_transcription: bool = True,
      include_translated_tts: bool = True,
      include_voice_clone: bool = True,
      include_voice_list: bool = True,
      include_text_to_sound: bool = True,
      include_audio_separation: bool = True,
  ) -> None:
    super().__init__(tool_filter=tool_filter)
    self._helpers = CambHelpers(
        api_key=api_key,
        timeout=timeout,
        max_poll_attempts=max_poll_attempts,
        poll_interval=poll_interval,
    )
    self._include_tts = include_tts
    self._include_translation = include_translation
    self._include_transcription = include_transcription
    self._include_translated_tts = include_translated_tts
    self._include_voice_clone = include_voice_clone
    self._include_voice_list = include_voice_list
    self._include_text_to_sound = include_text_to_sound
    self._include_audio_separation = include_audio_separation

  @override
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> List[BaseTool]:
    """Return the enabled CAMB AI tools as :class:`FunctionTool` instances."""
    helpers = self._helpers
    all_tools: list[BaseTool] = []

    if self._include_tts:
      all_tools.append(FunctionTool(make_tts_func(helpers)))
    if self._include_translation:
      all_tools.append(FunctionTool(make_translate_func(helpers)))
    if self._include_transcription:
      all_tools.append(FunctionTool(make_transcribe_func(helpers)))
    if self._include_translated_tts:
      all_tools.append(FunctionTool(make_translated_tts_func(helpers)))
    if self._include_voice_clone:
      all_tools.append(FunctionTool(make_clone_voice_func(helpers)))
    if self._include_voice_list:
      all_tools.append(FunctionTool(make_list_voices_func(helpers)))
    if self._include_text_to_sound:
      all_tools.append(FunctionTool(make_text_to_sound_func(helpers)))
    if self._include_audio_separation:
      all_tools.append(FunctionTool(make_audio_separation_func(helpers)))

    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  @override
  async def close(self) -> None:
    """Release resources held by the toolset."""
    pass
