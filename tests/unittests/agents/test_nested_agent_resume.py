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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from ..testing_utils import MockModel


async def audit_tool(tool_context: ToolContext) -> str:
  user_auth = None
  if (
      tool_context.tool_confirmation
      and tool_context.tool_confirmation.confirmed
  ):
    user_auth = tool_context.tool_confirmation.payload

  if not user_auth:
    tool_context.request_confirmation(hint="Please confirm")
    return "pending"
  return f"approved {user_auth}"


@pytest.mark.asyncio
async def test_nested_agent_resume_after_lro_pause() -> None:
  """Tests that a parent LlmAgent properly suspends when a nested sub-agent pauses.

  This addresses Issue #5349: Bug 2. When a nested sub-agent triggers a Long-Running
  Operation pause, the orchestrating LlmAgent must not incorrectly set end_of_agent=True.
  """
  # 1. Setup MockModels
  worker_tool_call = types.Part.from_function_call(name="audit_tool", args={})
  worker_tool_call.function_call.id = "audit_tool_1"

  worker_model = MockModel.create([[worker_tool_call], "worker done"])

  root_transfer_call = types.Part.from_function_call(
      name="transfer_to_agent", args={"agent_name": "QA_Loop"}
  )
  root_transfer_call.function_call.id = "transfer_1"

  root_model = MockModel.create([[root_transfer_call], "root done"])

  # 2. Setup Nested Agents Topology: Root -> Loop -> Seq -> Worker
  worker = LlmAgent(
      name="Worker", model=worker_model, tools=[FunctionTool(func=audit_tool)]
  )

  seq = SequentialAgent(name="Pipeline", sub_agents=[worker])

  loop = LoopAgent(name="QA_Loop", sub_agents=[seq], max_iterations=1)

  root = LlmAgent(name="Root", model=root_model, sub_agents=[loop])

  app = App(
      name="test_app",
      root_agent=root,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )

  # 3. First run: Should hit the tool and pause
  session_service = InMemorySessionService()
  runner = Runner(
      app=app,
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      memory_service=InMemoryMemoryService(),
  )
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  events_run_1 = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part.from_text(text="start")]
      ),
  ):
    events_run_1.append(event)

  # Check that it asked for confirmation (HITL pause)
  hitl_triggered = False
  for event in events_run_1:
    for fc in event.get_function_calls():
      if fc.name == "adk_request_confirmation":
        hitl_triggered = True
  assert hitl_triggered, "Invocation should be paused for confirmation"

  session_from_db = await session_service.get_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )
  assert session_from_db is not None

  # Crucial Bug Check: Root agent should NOT have emitted end_of_agent=True
  for event in session_from_db.events:
    if event.actions and event.actions.end_of_agent:
      assert (
          event.author != "Root"
      ), "Root agent should not have ended prematurely"

  # 4. Resume run: Provide the confirmation
  confirmation_part = types.Part(
      function_response=types.FunctionResponse(
          id="placeholder",
          name="adk_request_confirmation",
          response={"confirmed": True, "payload": "yes"},
      )
  )

  # Find the adk_request_confirmation ID to mock user response
  req_id = None
  inv_id = None
  for event in session_from_db.events:
    for fc in event.get_function_calls():
      if fc.name == "adk_request_confirmation":
        req_id = fc.id
        inv_id = event.invocation_id

  assert req_id is not None, "adk_request_confirmation function call not found"
  confirmation_part.function_response.id = req_id

  resume_msg = types.Content(role="user", parts=[confirmation_part])

  events_run_2 = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=resume_msg,
      invocation_id=inv_id,
  ):
    events_run_2.append(event)

  # Validate final closure: Root should finish cleanly now
  root_ended = False
  for event in events_run_2:
    if event.actions and event.actions.end_of_agent and event.author == "Root":
      root_ended = True

  assert (
      root_ended
  ), "Root agent should properly end after sub-agents are completely done"
