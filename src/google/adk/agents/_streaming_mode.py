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
