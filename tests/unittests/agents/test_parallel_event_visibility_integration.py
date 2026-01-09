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

"""Integration tests for parallel agent event visibility (GitHub issue #3470)."""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
import pytest

from tests.unittests import testing_utils


@pytest.mark.asyncio
async def test_sequence_of_parallels():
  """Test: Sequential[Parallel1[A,B,C], Parallel2[D,E,F]].

  KEY test from GitHub issue #3470. D,E,F should see A,B,C outputs.
  """
  agent_a = LlmAgent(
      name="AgentA", model=testing_utils.MockModel.create(responses=["A"])
  )
  agent_d = LlmAgent(
      name="AgentD", model=testing_utils.MockModel.create(responses=["D"])
  )

  parallel1 = ParallelAgent(name="P1", sub_agents=[agent_a])
  parallel2 = ParallelAgent(name="P2", sub_agents=[agent_d])
  root = SequentialAgent(name="Root", sub_agents=[parallel1, parallel2])

  runner = InMemoryRunner(agent=root, app_name="test")
  session = await runner.session_service.create_session(
      app_name="test", user_id="user"
  )

  async for event in runner.run_async(
      user_id="user",
      session_id=session.id,
      new_message=types.Content(role="user", parts=[types.Part(text="go")]),
  ):
    pass

  final_session = await runner.session_service.get_session(
      app_name="test", user_id="user", session_id=session.id
  )

  # Debug: print all events and their branches
  print("\n=== All Events in Session ===")
  for event in final_session.events:
    branch_info = event.branch.active_forks if event.branch else {}
    print(f"{event.author:15} | branch={branch_info}")

  agent_a_branch = next(
      e.branch for e in final_session.events if e.author == "AgentA"
  )
  agent_d_branch = next(
      e.branch for e in final_session.events if e.author == "AgentD"
  )

  # KEY: D's branch should be able to see A's branch
  assert agent_d_branch.can_see(agent_a_branch), (
      f"AgentD should see AgentA. A={agent_a_branch},"
      f" D={agent_d_branch}"
  )
