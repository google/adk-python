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

import os
from unittest import mock

from google.adk.models.llm_request import LlmRequest
from google.adk.models.openai_llm import OpenAIRealtime
from google.genai import types
import pytest


@pytest.mark.asyncio
async def test_connect_builds_session_update_from_labels(monkeypatch):
  os.environ['OPENAI_API_KEY'] = 'test'
  llm = OpenAIRealtime(model='gpt-4o-realtime-preview')

  req = LlmRequest()
  req.model = 'gpt-4o-realtime-preview'
  req.config.system_instruction = 'You are a test.'
  # Simulate tools attached
  req.config.tools = [
      types.Tool(
          function_declarations=[
              types.FunctionDeclaration(
                  name='get_time',
                  parameters=types.Schema(type=types.Type.OBJECT),
              )
          ]
      )
  ]
  req.config.labels = {
      'adk_openai_session_json': {'modalities': ['text'], 'voice': 'ash'}
  }

  fake_ws = mock.AsyncMock()
  sent = []

  async def fake_connect(url, additional_headers=None, max_size=None):
    fake_ws.send = mock.AsyncMock(
        side_effect=lambda payload: sent.append(payload)
    )
    return fake_ws

  monkeypatch.setattr('websockets.connect', fake_connect)

  async with llm.connect(req) as conn:
    assert conn is not None

  # First send should be a session.update including our overrides and instructions/tools
  assert sent, 'No payloads sent to websocket.'
  payload = sent[0]
  assert 'session.update' in payload
  assert 'You are a test.' in payload
  assert 'modalities' in payload and 'voice' in payload


def test_supported_models_regex():
  patterns = OpenAIRealtime.supported_models()
  assert any('realtime' in p for p in patterns)
