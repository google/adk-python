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

"""Tests for google.adk.live._flow_utils."""

from __future__ import annotations

from unittest import mock

from google.adk.events.event import Event
from google.adk.live._flow_utils import handle_control_event_flush
from google.adk.live._flow_utils import require_live_request_queue
from google.adk.live.live_request_queue import LiveRequestQueue
from google.adk.models.llm_response import LlmResponse
import pytest


def test_require_live_request_queue_returns_queue():
  """Returns the active LiveRequestQueue when present on the context."""
  queue = LiveRequestQueue()
  invocation_context = mock.Mock(live_request_queue=queue)

  result = require_live_request_queue(invocation_context)

  assert result is queue


def test_require_live_request_queue_raises_when_missing():
  """Raises ValueError when the context has no LiveRequestQueue."""
  invocation_context = mock.Mock(live_request_queue=None)

  with pytest.raises(ValueError, match='LiveRequestQueue'):
    require_live_request_queue(invocation_context)


async def test_handle_control_event_flush_flushes_model_audio_on_interrupt():
  """Flushes only model audio when the response indicates an interruption."""
  flushed_event = Event(author='model')
  audio_cache_manager = mock.Mock()
  audio_cache_manager.flush_caches = mock.AsyncMock(
      return_value=[flushed_event]
  )
  flow = mock.Mock(audio_cache_manager=audio_cache_manager)
  invocation_context = mock.Mock()
  llm_response = LlmResponse(interrupted=True)

  events = await handle_control_event_flush(
      flow, invocation_context, llm_response
  )

  assert events == [flushed_event]
  audio_cache_manager.flush_caches.assert_awaited_once_with(
      invocation_context,
      flush_user_audio=False,
      flush_model_audio=True,
  )


async def test_handle_control_event_flush_flushes_both_on_turn_complete():
  """Flushes both user and model audio when the turn completes."""
  flushed_event = Event(author='model')
  audio_cache_manager = mock.Mock()
  audio_cache_manager.flush_caches = mock.AsyncMock(
      return_value=[flushed_event]
  )
  flow = mock.Mock(audio_cache_manager=audio_cache_manager)
  invocation_context = mock.Mock()
  llm_response = LlmResponse(turn_complete=True)

  events = await handle_control_event_flush(
      flow, invocation_context, llm_response
  )

  assert events == [flushed_event]
  audio_cache_manager.flush_caches.assert_awaited_once_with(
      invocation_context,
      flush_user_audio=True,
      flush_model_audio=True,
  )
