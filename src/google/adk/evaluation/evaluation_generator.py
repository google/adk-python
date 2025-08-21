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

import importlib
from typing import Any
from typing import Optional
import uuid

from pydantic import BaseModel

from ..agents.llm_agent import Agent
from ..artifacts.base_artifact_service import BaseArtifactService
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..memory.base_memory_service import BaseMemoryService
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from ..utils.context_utils import Aclosing
from .eval_case import EvalCase
from .eval_case import IntermediateData
from .eval_case import Invocation
from .eval_case import SessionInput
from .eval_set import EvalSet


class EvalCaseResponses(BaseModel):
  """Contains multiple responses associated with an EvalCase.

  Multiple responses are a result of repeated requests to genereate inferences.
  """

  eval_case: EvalCase
  responses: list[list[Invocation]]


class EvaluationGenerator:
  """Generates evaluation responses for agents."""

  @staticmethod
  async def generate_responses(
      eval_set: EvalSet,
      agent_module_path: str,
      repeat_num: int = 3,
      agent_name: str = None,
  ) -> list[EvalCaseResponses]:
    """Returns evaluation responses for the given dataset and agent.

    Args:
      eval_set: The eval set that needs to be scraped for responses.
      agent_module_path: Path to the module that contains the root agent.
      repeat_num: Number of time the eval dataset should be repeated. This is
        usually done to remove uncertainty that a single run may bring.
      agent_name: The name of the agent that should be evaluated. This is
        usually the sub-agent.
    """
    results = []

    for eval_case in eval_set.eval_cases:
      responses = []
      for _ in range(repeat_num):
        response_invocations = await EvaluationGenerator._process_query(
            eval_case.conversation,
            agent_module_path,
            agent_name,
            eval_case.session_input,
        )
        responses.append(response_invocations)

      results.append(
          EvalCaseResponses(eval_case=eval_case, responses=responses)
      )

    return results

  @staticmethod
  def generate_responses_from_session(session_path, eval_dataset):
    """Returns evaluation responses by combining session data with eval data.

    Args:
      session_path: Path to a json file that contains session data.
      eval_dataset: The eval data set that should be combined with the session
        data.
    """
    results = []

    with open(session_path, "r") as f:
      session_data = Session.model_validate_json(f.read())
      print("loaded session", session_path)

    for data in eval_dataset:
      # load session data from session_path
      results.append(
          EvaluationGenerator._process_query_with_session(
              session_data,
              data,
          )
      )

    return results

  @staticmethod
  async def _process_query(
      invocations: list[Invocation],
      module_name: str,
      agent_name: Optional[str] = None,
      initial_session: Optional[SessionInput] = None,
  ) -> list[Invocation]:
    """Process a query using the agent and evaluation dataset."""
    module_path = f"{module_name}"
    agent_module = importlib.import_module(module_path)
    root_agent = agent_module.agent.root_agent

    reset_func = getattr(agent_module.agent, "reset_data", None)

    agent_to_evaluate = root_agent
    if agent_name:
      agent_to_evaluate = root_agent.find_agent(agent_name)
      assert agent_to_evaluate, f"Sub-Agent `{agent_name}` not found."

    return await EvaluationGenerator._generate_inferences_from_root_agent(
        invocations, agent_to_evaluate, reset_func, initial_session
    )

  @staticmethod
  async def _generate_inferences_from_root_agent(
      invocations: list[Invocation],
      root_agent: Agent,
      reset_func: Optional[Any] = None,
      initial_session: Optional[SessionInput] = None,
      session_id: Optional[str] = None,
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      memory_service: Optional[BaseMemoryService] = None,
  ) -> list[Invocation]:
    """Scrapes the root agent given the list of Invocations."""
    if not session_service:
      session_service = InMemorySessionService()

    if not memory_service:
      memory_service = InMemoryMemoryService()

    app_name = (
        initial_session.app_name if initial_session else "EvaluationGenerator"
    )
    user_id = initial_session.user_id if initial_session else "test_user_id"
    session_id = session_id if session_id else str(uuid.uuid4())

    _ = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        state=initial_session.state if initial_session else {},
        session_id=session_id,
    )

    if not artifact_service:
      artifact_service = InMemoryArtifactService()

    runner = Runner(
        app_name=app_name,
        agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
        memory_service=memory_service,
    )

    # Reset agent state for each query
    if callable(reset_func):
      reset_func()

    response_invocations = []

    for invocation in invocations:
      final_response = None
      user_content = invocation.user_content
      tool_uses = []
      invocation_id = ""

      async with Aclosing(
          runner.run_async(
              user_id=user_id, session_id=session_id, new_message=user_content
          )
      ) as agen:
        async for event in agen:
          invocation_id = (
              event.invocation_id if not invocation_id else invocation_id
          )

          if (
              event.is_final_response()
              and event.content
              and event.content.parts
          ):
            final_response = event.content
          elif event.get_function_calls():
            for call in event.get_function_calls():
              tool_uses.append(call)

      response_invocations.append(
          Invocation(
              invocation_id=invocation_id,
              user_content=user_content,
              final_response=final_response,
              intermediate_data=IntermediateData(tool_uses=tool_uses),
          )
      )

    return response_invocations

  @staticmethod
  def _process_query_with_session(session_data, data):
    """Process the queries using the existing session data without invoking the runner."""
    responses = data.copy()

    # Iterate through the provided queries and align them with the session
    # events
    for index, eval_entry in enumerate(responses):
      query = eval_entry["query"]
      actual_tool_uses = []
      response = None

      # Search for the corresponding session events
      for event in session_data.events:
        # Match the query to a user event
        if (
            event.author == "user"
            and event.content
            and event.content.parts
            and event.content.parts[0].text == query
        ):
          # Look for subsequent tool usage or model responses
          for subsequent_event in session_data.events:
            if subsequent_event.invocation_id == event.invocation_id:
              # Extract tool usage
              if subsequent_event.content.parts[0].function_call:
                call = subsequent_event.content.parts[0].function_call
                actual_tool_uses.append(
                    {"tool_name": call.name, "tool_input": call.args}
                )
              # Extract final response
              elif subsequent_event.author != "user":
                response = subsequent_event.content.parts[0].text

      # Update the results for the current query
      responses[index]["actual_tool_use"] = actual_tool_uses
      responses[index]["response"] = response
    return responses
