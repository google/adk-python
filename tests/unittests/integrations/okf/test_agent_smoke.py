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

"""End-to-end: an agent calls ``okf_read`` and answers from the concept body."""

from __future__ import annotations

from pathlib import Path

from google.adk.agents.llm_agent import Agent
from google.adk.integrations.okf import OkfToolset
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from tests.unittests.testing_utils import InMemoryRunner
from tests.unittests.testing_utils import MockModel

FIXTURES = Path(__file__).parent / "fixtures" / "bundles"


@pytest.mark.asyncio
async def test_agent_reads_a_concept_through_the_toolset():
  """The toolset resolves into a real agent turn: model → tool → bundle."""
  call = types.Part.from_function_call(
      name="okf_read", args={"path": "/metrics/revenue.md"}
  )
  mock_model = MockModel.create([
      LlmResponse(content=types.Content(role="model", parts=[call])),
      LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part.from_text(text="Revenue is ...")]
          )
      ),
  ])
  agent = Agent(
      name="root_agent",
      model=mock_model,
      instruction="Answer from the knowledge bundle.",
      tools=[
          OkfToolset(
              bundle=str(FIXTURES / "acme_retail"), tool_name_prefix="okf"
          )
      ],
  )
  runner = InMemoryRunner(agent)

  await runner.run_async("What is revenue?")

  assert len(mock_model.requests) == 2
  payloads = [
      part.function_response.response
      for content in mock_model.requests[1].contents
      for part in content.parts or []
      if part.function_response
  ]
  assert len(payloads) == 1

  payload = payloads[0]
  assert payload["rel_path"] == "metrics/revenue.md"
  assert payload["type"] == "Metric"
  assert payload["trust_tier"] == "human-reviewed"
  assert payload["status"] == "stable"
  assert payload["content_is_untrusted_data"] is True
  assert "# Definition" in payload["body"]
