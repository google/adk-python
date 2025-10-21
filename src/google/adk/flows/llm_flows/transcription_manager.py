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

import logging
import time
from typing import TYPE_CHECKING

from google.genai import types

from ...events.event import Event

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext

logger = logging.getLogger('google_adk.' + __name__)


class TranscriptionManager:
  """Manages transcription events for live streaming flows."""

  async def handle_input_transcription(
      self,
      invocation_context: InvocationContext,
      transcription: types.Transcription,
  ) -> None:
    """Handle user input transcription events.

    Args:
      invocation_context: The current invocation context.
      transcription: The transcription data from user input.
    """
    return await self._create_and_save_transcription_event(
        invocation_context=invocation_context,
        transcription=transcription,
        author='user',
        is_input=True,
    )

  async def handle_output_transcription(
      self,
      invocation_context: InvocationContext,
      transcription: types.Transcription,
  ) -> None:
    """Handle model output transcription events.

    Args:
      invocation_context: The current invocation context.
      transcription: The transcription data from model output.
    """
    return await self._create_and_save_transcription_event(
        invocation_context=invocation_context,
        transcription=transcription,
        author=invocation_context.agent.name,
        is_input=False,
    )

  async def _create_and_save_transcription_event(
      self,
      invocation_context: InvocationContext,
      transcription: types.Transcription,
      author: str,
      is_input: bool,
  ) -> None:
    """Create and save a transcription event to session service.

    Args:
      invocation_context: The current invocation context.
      transcription: The transcription data.
      author: The author of the transcription event.
      is_input: Whether this is an input (user) or output (model) transcription.
    """
    try:
      transcription_event = Event(
          id=Event.new_id(),
          invocation_id=invocation_context.invocation_id,
          author=author,
          input_transcription=transcription if is_input else None,
          output_transcription=transcription if not is_input else None,
          timestamp=time.time(),
      )

      # Save transcription event to session

      logger.debug(
          'Saved %s transcription event for %s: %s',
          'input' if is_input else 'output',
          author,
          transcription.text
          if hasattr(transcription, 'text')
          else 'audio transcription',
      )

      return transcription_event
    except Exception as e:
      logger.error(
          'Failed to save %s transcription event: %s',
          'input' if is_input else 'output',
          e,
      )
      raise

  def get_transcription_stats(
      self, invocation_context: InvocationContext
  ) -> dict[str, int]:
    """Get statistics about transcription events in the session.

    Args:
      invocation_context: The current invocation context.

    Returns:
      Dictionary containing transcription statistics.
    """
    input_count = 0
    output_count = 0

    for event in invocation_context.session.events:
      if hasattr(event, 'input_transcription') and event.input_transcription:
        input_count += 1
      if hasattr(event, 'output_transcription') and event.output_transcription:
        output_count += 1

    return {
        'input_transcriptions': input_count,
        'output_transcriptions': output_count,
        'total_transcriptions': input_count + output_count,
    }
