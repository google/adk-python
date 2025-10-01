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

from enum import Enum
import logging
import sys
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

logger = logging.getLogger('google_adk.' + __name__)


class StreamingMode(Enum):
  NONE = None
  SSE = 'sse'
  BIDI = 'bidi'


class RunConfig(BaseModel):
  """Configs for runtime behavior of agents."""

  model_config = ConfigDict(
      extra='forbid',
  )
  """The pydantic model config."""

  speech_config: Optional[types.SpeechConfig] = None
  """Speech configuration for the live agent."""

  response_modalities: Optional[list[str]] = None
  """The output modalities. If not set, it's default to AUDIO."""

  save_input_blobs_as_artifacts: bool = Field(
      default=False,
      deprecated=True,
      description=(
          'Whether or not to save the input blobs as artifacts. DEPRECATED: Use'
          ' SaveFilesAsArtifactsPlugin instead for better control and'
          ' flexibility. See google.adk.plugins.SaveFilesAsArtifactsPlugin.'
      ),
  )

  support_cfc: bool = False
  """
  Whether to support CFC (Compositional Function Calling). Only applicable for
  StreamingMode.SSE. If it's true. the LIVE API will be invoked. Since only LIVE
  API supports CFC

  .. warning::
      This feature is **experimental** and its API or behavior may change
      in future releases.
  """

  streaming_mode: StreamingMode = StreamingMode.NONE
  """Streaming mode, None or StreamingMode.SSE or StreamingMode.BIDI."""

  output_audio_transcription: Optional[types.AudioTranscriptionConfig] = Field(
      default_factory=types.AudioTranscriptionConfig
  )
  """Output transcription for live agents with audio response."""

  input_audio_transcription: Optional[types.AudioTranscriptionConfig] = Field(
      default_factory=types.AudioTranscriptionConfig
  )
  """Input transcription for live agents with audio input from user."""

  realtime_input_config: Optional[types.RealtimeInputConfig] = None
  """Realtime input config for live agents with audio input from user."""

  enable_affective_dialog: Optional[bool] = None
  """If enabled, the model will detect emotions and adapt its responses accordingly."""

  proactivity: Optional[types.ProactivityConfig] = None
  """Configures the proactivity of the model. This allows the model to respond proactively to the input and to ignore irrelevant input."""

  session_resumption: Optional[types.SessionResumptionConfig] = None
  """Configures session resumption mechanism. Only support transparent session resumption mode now."""

  save_live_audio: bool = False
  """Saves live video and audio data to session and artifact service.

  Right now, only audio is supported.
  """

  max_llm_calls: int = 500
  """
  A limit on the total number of llm calls for a given run.

  Valid Values:
    - More than 0 and less than sys.maxsize: The bound on the number of llm
      calls is enforced, if the value is set in this range.
    - Less than or equal to 0: This allows for unbounded number of llm calls.
  """

  @field_validator('max_llm_calls', mode='after')
  @classmethod
  def validate_max_llm_calls(cls, value: int) -> int:
    if value == sys.maxsize:
      raise ValueError(f'max_llm_calls should be less than {sys.maxsize}.')
    elif value <= 0:
      logger.warning(
          'max_llm_calls is less than or equal to 0. This will result in'
          ' no enforcement on total number of llm calls that will be made for a'
          ' run. This may not be ideal, as this could result in a never'
          ' ending communication between the model and the agent in certain'
          ' cases.',
      )

    return value
