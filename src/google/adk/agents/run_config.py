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

from enum import Enum
import logging
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

logger = logging.getLogger('google_adk.' + __name__)


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


class StreamingMode(Enum):
  """Streaming modes for agent execution.

  This enum defines different streaming behaviors for how the agent returns
  events as model response.
  """

  NONE = None
  """Non-streaming mode (default).

  In this mode:
  - The runner returns one single content in a turn (one user / model
    interaction).
  - No partial/intermediate events are produced
  - Suitable for: CLI tools, batch processing, synchronous workflows

  Example:
    ```python
    config = RunConfig(streaming_mode=StreamingMode.NONE)
    async for event in runner.run_async(..., run_config=config):
      # event.partial is always False
      # Only final responses are yielded
      if event.content:
        print(event.content.parts[0].text)
    ```
  """

  SSE = 'sse'
  """Server-Sent Events (SSE) streaming mode.

  In this mode:
  - The runner yields events progressively as the LLM generates responses
  - Both partial events (streaming chunks) and aggregated events are yielded
  - Suitable for: real-time display with typewriter effects in Web UIs, chat
    applications, interactive displays

  Event Types in SSE Mode:
  - **Partial text events** (event.partial=True, contains text):
    Streaming text chunks for typewriter effect. These should typically be
    displayed to users in real-time.

  - **Partial function call events** (event.partial=True, contains function_call):
    Internal streaming chunks used to progressively build function call
    arguments. These are typically NOT displayed to end users.

  - **Aggregated events** (event.partial=False):
    The complete, aggregated response after all streaming chunks. Contains
    the full text or complete function call with all arguments.

  Important Considerations:
  1. **Duplicate text issue**: With Progressive SSE Streaming enabled
     (default), you will receive both partial text chunks AND a final
     aggregated text event. To avoid displaying text twice:
     - Option A: Only display partial text events, skip final text events
     - Option B: Only display final events, skip all partial events
     - Option C: Track what's been displayed and skip duplicates

  2. **Event filtering**: Applications should filter events based on their
     needs. Common patterns:

     # Pattern 1: Display only partial text + final function calls
     async for event in runner.run_async(...):
       if event.partial and event.content and event.content.parts:
         # Check if it's text (not function call)
         if any(part.text for part in event.content.parts):
           if not any(part.function_call for part in event.content.parts):
             # Display partial text for typewriter effect
             text = ''.join(p.text or '' for p in event.content.parts)
             print(text, end='', flush=True)
       elif not event.partial and event.get_function_calls():
         # Display final function calls
         for fc in event.get_function_calls():
           print(f"Calling {fc.name}({fc.args})")

     # Pattern 2: Display only final events (no streaming effect)
     async for event in runner.run_async(...):
       if not event.partial:
         # Only process final responses
         if event.content:
           text = ''.join(p.text or '' for p in event.content.parts)
           print(text)

  3. **Progressive SSE Streaming feature**: Controlled by the
     ADK_ENABLE_PROGRESSIVE_SSE_STREAMING environment variable (default: ON).
     - When ON: Preserves original part ordering, supports function call
       argument streaming, produces partial events + final aggregated event
     - When OFF: Simple text accumulation, may lose some information

  Example:
    ```python
    config = RunConfig(streaming_mode=StreamingMode.SSE)
    displayed_text = ""

    async for event in runner.run_async(..., run_config=config):
      if event.partial:
        # Partial streaming event
        if event.content and event.content.parts:
          # Check if this is text (not a function call)
          has_text = any(part.text for part in event.content.parts)
          has_fc = any(part.function_call for part in event.content.parts)

          if has_text and not has_fc:
            # Display partial text chunks for typewriter effect
            text = ''.join(p.text or '' for p in event.content.parts)
            print(text, end='', flush=True)
            displayed_text += text
      else:
        # Final event - check if we already displayed this content
        if event.content:
          final_text = ''.join(p.text or '' for p in event.content.parts)
          if final_text != displayed_text:
            # New content not yet displayed
            print(final_text)
    ```

  See Also:
  - Event.is_final_response() for identifying final responses
  """

  BIDI = 'bidi'
  """Bidirectional streaming mode.

  So far this mode is not used in the standard execution path. The actual
  bidirectional streaming behavior via runner.run_live() uses a completely
  different code path that doesn't rely on streaming_mode.

  For bidirectional streaming, use runner.run_live() instead of run_async().
  """


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
  main event loop.

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

  max_llm_calls: int = 500
  """
  A limit on the total number of llm calls for a given run.

  Valid Values:
    - More than 0 and less than sys.maxsize: The bound on the number of llm
      calls is enforced, if the value is set in this range.
    - Less than or equal to 0: This allows for unbounded number of llm calls.
  """

  custom_metadata: Optional[dict[str, Any]] = None
  """Custom metadata for the current invocation."""

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
