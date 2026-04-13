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
from typing import Any
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from ...models.google_llm import Gemini

if TYPE_CHECKING:
  from ...models.llm_request import LlmRequest
  from ...models.llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)


class ReplayVerificationError(Exception):
  """Exception raised when replay verification fails."""


class _ConformanceTestGemini(Gemini):
  """A mocked Gemini model for conformance test replay mode.

  This class is used to mock the Gemini model in conformance test replay mode.
  It is a subclass of Gemini and overrides the `generate_content_async`` method to
  return a mocked response from the provided recordingss.
  """

  def __init__(
      self,
      *,
      config: dict[str, Any],
      **kwargs: Any,
  ) -> None:
    super().__init__(**kwargs)
    recordings = config.get('_adk_replay_recordings')
    self._user_message_index = config.get('user_message_index')
    self._agent_name = config.get('agent_name')
    self._replay_index = config.get('current_replay_index')
    # Pre-filter LLM recordings for this agent and message index
    self._agent_llm_recordings = [
        recording.llm_recording
        for recording in recordings.recordings
        if recording.agent_name == self._agent_name
        and recording.user_message_index == self._user_message_index
        and recording.llm_recording
    ]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Replay LLM response from recordings instead of making real call."""
    logger.debug(
        'Replaying LLM response for agent %s (index %d)',
        self._agent_name,
        self._replay_index,
    )

    if self._replay_index >= len(self._agent_llm_recordings):
      raise ReplayVerificationError(
          'Runtime sent more LLM requests than expected for agent'
          f" '{self._agent_name}' at user_message_index"
          f' {self._user_message_index}. Expected'
          f' {len(self._agent_llm_recordings)}, but got request at index'
          f' {self._replay_index}'
      )

    recording = self._agent_llm_recordings[self._replay_index]

    # Verify request matches
    self._verify_llm_request_match(
        recording.llm_request, llm_request, self._replay_index
    )

    for response in recording.llm_responses:
      yield response

  def _verify_llm_request_match(
      self,
      recorded_request: LlmRequest,
      current_request: LlmRequest,
      replay_index: int,
  ) -> None:
    """Verify that the current LLM request exactly matches the recorded one."""
    # Comprehensive exclude dict for all fields that can differ between runs
    excluded_fields = {
        'live_connect_config': True,
        'config': {  # some config fields can vary per run
            'http_options': True,
            'labels': True,
        },
    }

    # Compare using model dumps with nested exclude dict
    recorded_dict = recorded_request.model_dump(
        exclude_none=True, exclude=excluded_fields, exclude_defaults=True
    )
    current_dict = current_request.model_dump(
        exclude_none=True, exclude=excluded_fields, exclude_defaults=True
    )

    if recorded_dict != current_dict:
      raise ReplayVerificationError(
          f"""LLM request mismatch in turn {self._user_message_index} for agent '{self._agent_name}' (index {replay_index}):
recorded: {recorded_dict}
current: {current_dict}"""
      )
