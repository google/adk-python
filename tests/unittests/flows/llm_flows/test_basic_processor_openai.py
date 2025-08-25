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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.basic import request_processor
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest


@pytest.mark.asyncio
async def test_basic_processor_openai_session_labels_passthrough():
  agent = LlmAgent(name="a", model="gpt-4o-realtime-preview")
  ic_run_cfg = RunConfig(
      openai_realtime_session={
          "modalities": ["text"],
          "turn_detection": {"type": "none"},
      }
  )
  llm_request = LlmRequest()

  class Ctx:

    def __init__(self):
      self.agent = agent
      self.run_config = ic_run_cfg

  ctx = Ctx()

  # Run processor
  agen = request_processor.run_async(ctx, llm_request)
  async for _ in agen:
    pass

  # Labels should contain vendor session under key
  assert llm_request.config.labels["adk_openai_session_json"]["modalities"] == [
      "text"
  ]
