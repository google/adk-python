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
import os
import sys
from typing import Any
from typing import Optional
import warnings

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from ..sessions.base_session_service import GetSessionConfig
from ..telemetry.context import TelemetryConfig
from ._streaming_mode import StreamingMode

logger = logging.getLogger('google_adk.' + __name__)

_DEFAULT_MAX_LLM_CALLS = 500


def _default_max_llm_calls() -> int:
  """Resolves the default max LLM calls limit from environment or fallback."""
  if env_val := os.getenv('ADK_MAX_LLM_CALLS'):
    try:
      return int(env_val)
    except ValueError:
      logger.warning(
          'Invalid value for ADK_MAX_LLM_CALLS env var: %s. Using default %s.',
          env_val,
          _DEFAULT_MAX_LLM_CALLS,
      )
  return _DEFAULT_MAX_LLM_CALLS


class ToolThreadPoolConfig(BaseModel):
  """Configuration for the tool thread pool executor.

  Attributes:
    max_workers: Maximum number of worker threads in the pool. Defaults to 4.
  """

  model_config = ConfigDict(
      extra='forbid',
  )

  max_workers: int = Field(
      default=4,
      description='Maximum number of worker threads in the pool.',
      ge=1,
  )


class RunConfig(BaseModel):
  """Configs for runtime behavior of agents.

  The configs here will be overridden by agent-specific configurations.
  """

  model_config = ConfigDict(
      extra='forbid',
  )
  """The pydantic model config."""

  speech_config: Optional[types.SpeechConfig] = None
  """Speech configuration for the live agent."""

  http_options: Optional[types.HttpOptions] = None
  """HTTP options for the agent execution (e.g. custom headers)."""

  labels: Optional[dict[str, str]] = None
  """User labels for the current invocation (e.g. for billing/attribution)."""

  response_modalities: Optional[list[types.Modality]] = None
  """The output modalities. If not set, it's default to AUDIO."""

  avatar_config: Optional[types.AvatarConfig] = None
  """Avatar configuration for the live agent."""

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

  explicit_vad_signal: Optional[bool] = None
  """Whether to enable explicit voice activity detection (VAD) signals from the model."""

  translation_config: Optional[types.TranslationConfig] = None
  """Configures real-time speech-to-speech translation.

  Only supported by translation models such as
  `gemini-3.5-live-translate-preview`.
  """

  enable_affective_dialog: Optional[bool] = None
  """If enabled, the model will detect emotions and adapt its responses accordingly."""

  proactivity: Optional[types.ProactivityConfig] = None
  """Configures the proactivity of the model. This allows the model to respond proactively to the input and to ignore irrelevant input."""

  session_resumption: Optional[types.SessionResumptionConfig] = None
  """Configures session resumption mechanism. Only support transparent session resumption mode now."""

  history_config: Optional[types.HistoryConfig] = None
  """Configures the exchange of history between the client and the server."""

  context_window_compression: Optional[types.ContextWindowCompressionConfig] = (
      None
  )
  """Configuration for context window compression. If set, this will enable context window compression for LLM input."""

  save_live_blob: bool = False
  """Saves live video and audio data to session and artifact service."""

  tool_thread_pool_config: Optional[ToolThreadPoolConfig] = None
  """Configuration for running tools in a thread pool for live mode.

  When set, tool executions will run in a separate thread pool executor
  instead of the main event loop. When None (default), tools run in the
  main event loop. One pool serves every invocation running on the same event
  loop and is shut down once that loop is gone, so its worker threads do not
  outlive it.

  This helps keep the event loop responsive for:
  - User interruptions to be processed immediately
  - Model responses to continue being received

  Both sync and async tools are supported. Async tools are run in a new event
  loop within the background thread, which helps catch blocking I/O mistakenly
  used inside async functions.

  IMPORTANT - GIL (Global Interpreter Lock) Considerations:

  Thread pool HELPS with (GIL is released):
  - Blocking I/O: time.sleep(), network calls, file I/O, database queries
  - C extensions: numpy, hashlib, image processing libraries
  - Async functions containing blocking I/O (common user mistake)

  Thread pool does NOT help with (GIL is held):
  - Pure Python CPU-bound code: loops, calculations, recursive algorithms
  - The GIL prevents true parallel execution for Python bytecode

  Cancelling an invocation drops a tool call that has not started yet, but
  Python cannot stop a thread that is already running, so a started call keeps
  its worker thread until it returns.

  For CPU-intensive Python code, consider alternatives:
  - Use C extensions that release the GIL
  - Break work into chunks with periodic `await asyncio.sleep(0)`
  - Use multiprocessing (ProcessPoolExecutor) for true parallelism

  Example:
    ```python
    from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig

    # Enable thread pool with default settings
    run_config = RunConfig(
        tool_thread_pool_config=ToolThreadPoolConfig(),
    )

    # Enable thread pool with custom max_workers
    run_config = RunConfig(
        tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8),
    )
    ```
  """

  save_live_audio: bool = Field(
      default=False,
      deprecated=True,
      description=(
          'DEPRECATED: Use save_live_blob instead. If set to True, it saves'
          ' live video and audio data to session and artifact service.'
      ),
  )

  max_llm_calls: int = Field(
      default_factory=_default_max_llm_calls,
      description=(
          'A limit on the total number of llm calls for a given run. Can be'
          ' overridden by ADK_MAX_LLM_CALLS environment variable.'
      ),
  )
  """
  A limit on the total number of llm calls for a given run.

  This limit can be overridden by setting the `ADK_MAX_LLM_CALLS` environment
  variable.

  Valid Values:
    - More than 0 and less than sys.maxsize: The bound on the number of llm
      calls is enforced, if the value is set in this range.
    - Less than or equal to 0: This allows for unbounded number of llm calls.
  """

  custom_metadata: Optional[dict[str, Any]] = None
  """Custom metadata for the current invocation."""

  telemetry: TelemetryConfig | None = None
  """Per-request OpenTelemetry configuration.

  Overrides the process-global telemetry env vars for the duration of this
  invocation. Each ``None`` field on the
  :class:`~google.adk.telemetry.TelemetryConfig` falls back to its
  corresponding env var. Lets multi-tenant hosts toggle telemetry knobs per
  request without leaking configuration across concurrent invocations.

  .. warning::
      Experimental; API may change.
  """

  get_session_config: Optional[GetSessionConfig] = None
  """Configuration for controlling which events are fetched when loading
  a session.

  When set, the Runner will pass this configuration to the session service's
  ``get_session`` method, allowing the caller to limit the events returned
  (e.g. via ``num_recent_events`` or ``after_timestamp``).  This is especially
  useful in combination with ``EventsCompactionConfig`` to avoid loading the
  full event history on every invocation.

  Example::

      from google.adk.agents.run_config import RunConfig
      from google.adk.sessions.base_session_service import GetSessionConfig

      run_config = RunConfig(
          get_session_config=GetSessionConfig(num_recent_events=50),
      )
  """

  model_input_context: list[types.Content] | None = None
  """Transient context to include in the model input for this invocation.

  The Runner does not persist these contents to the session. They are only
  added to the LLM request assembled for the current invocation, which lets
  callers provide per-turn context without changing the conversation history.
  """

  include_thoughts_from_other_agents: bool = False
  """Whether to include other agents' thought parts in LLM context.

  By default, thoughts from other agents are excluded when their messages are
  reformatted as user context for the current agent. Enable this only when
  agents are expected to share internal reasoning with one another.
  """

  @model_validator(mode='before')
  @classmethod
  def check_for_deprecated_save_live_audio(cls, data: Any) -> Any:
    """If save_live_audio is passed, use it to set save_live_blob."""
    if isinstance(data, dict) and 'save_live_audio' in data:
      warnings.warn(
          'The `save_live_audio` config is deprecated and will be removed in a'
          ' future release. Please use `save_live_blob` instead.',
          DeprecationWarning,
          stacklevel=2,
      )
      if data['save_live_audio']:
        data['save_live_blob'] = True
    return data

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
