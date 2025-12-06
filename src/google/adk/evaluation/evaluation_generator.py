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

import copy
import importlib
from typing import Any
from typing import AsyncGenerator
from typing import Optional
import uuid

from google.genai.types import Content
from pydantic import BaseModel

from ..agents.llm_agent import Agent
from ..artifacts.base_artifact_service import BaseArtifactService
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..events.event import Event
from ..memory.base_memory_service import BaseMemoryService
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from ..utils.context_utils import Aclosing
from ._retry_options_utils import EnsureRetryOptionsPlugin
from .app_details import AgentDetails
from .app_details import AppDetails
from .eval_case import EvalCase
from .eval_case import Invocation
from .eval_case import InvocationEvent
from .eval_case import InvocationEvents
from .eval_case import SessionInput
from .eval_set import EvalSet
from .request_intercepter_plugin import _RequestIntercepterPlugin
from .simulation.user_simulator import Status as UserSimulatorStatus
from .simulation.user_simulator import UserSimulator
from .simulation.user_simulator_provider import UserSimulatorProvider

_USER_AUTHOR = "user"
_DEFAULT_AUTHOR = "agent"


class EvalCaseResponses(BaseModel):
  """Contains multiple responses associated with an EvalCase.

  Multiple responses are a result of repeated requests to generate inferences.
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
      # assume only static conversations are needed
      user_simulator = UserSimulatorProvider().provide(eval_case)
      responses = []
      for _ in range(repeat_num):
        response_invocations = await EvaluationGenerator._process_query(
            agent_module_path,
            user_simulator,
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
      module_name: str,
      user_simulator: UserSimulator,
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
        agent_to_evaluate,
        user_simulator=user_simulator,
        reset_func=reset_func,
        initial_session=initial_session,
    )

  @staticmethod
  async def _generate_inferences_for_single_user_invocation(
      runner: Runner,
      user_id: str,
      session_id: str,
      user_content: Content,
  ) -> AsyncGenerator[Event, None]:
    invocation_id = None

    async with Aclosing(
        runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
        )
    ) as agen:

      async for event in agen:
        if not invocation_id:
          invocation_id = event.invocation_id
          yield Event(
              content=user_content,
              author=_USER_AUTHOR,
              invocation_id=invocation_id,
          )

        yield event

  @staticmethod
  async def _generate_inferences_from_root_agent(
      root_agent: Agent,
      user_simulator: UserSimulator,
      reset_func: Optional[Any] = None,
      initial_session: Optional[SessionInput] = None,
      session_id: Optional[str] = None,
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      memory_service: Optional[BaseMemoryService] = None,
  ) -> list[Invocation]:
    """Scrapes the root agent in coordination with the user simulator."""

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

    # Reset agent state for each query
    if callable(reset_func):
      reset_func()

    request_intercepter_plugin = _RequestIntercepterPlugin(
        name="request_intercepter_plugin"
    )
    # We ensure that there is some kind of retries on the llm_requests that are
    # generated from the Agent. This is done to make inferencing step of evals
    # more resilient to temporary model failures.
    ensure_retry_options_plugin = EnsureRetryOptionsPlugin(
        name="ensure_retry_options"
    )
    async with Runner(
        app_name=app_name,
        agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
        memory_service=memory_service,
        plugins=[request_intercepter_plugin, ensure_retry_options_plugin],
    ) as runner:
      events = []
      while True:
        next_user_message = await user_simulator.get_next_user_message(
            copy.deepcopy(events)
        )
        if next_user_message.status == UserSimulatorStatus.SUCCESS:
          async for (
              event
          ) in EvaluationGenerator._generate_inferences_for_single_user_invocation(
              runner, user_id, session_id, next_user_message.user_message
          ):
            events.append(event)
        else:  # no message generated
          break

      app_details_by_invocation_id = (
          EvaluationGenerator._get_app_details_by_invocation_id(
              events, request_intercepter_plugin
          )
      )
      return EvaluationGenerator.convert_events_to_eval_invocations(
          events, app_details_by_invocation_id
      )

  @staticmethod
  def convert_events_to_eval_invocations(
      events: list[Event],
      app_details_per_invocation: Optional[dict[str, AppDetails]] = None,
  ) -> list[Invocation]:
    """Converts a list of events to eval invocations."""
    events_by_invocation_id = (
        EvaluationGenerator._collect_events_by_invocation_id(events)
    )

    invocations = []
    for invocation_id, events in events_by_invocation_id.items():
      final_response = None
      user_content = ""
      invocation_timestamp = 0
      app_details = None
      if (
          app_details_per_invocation
          and invocation_id in app_details_per_invocation
      ):
        app_details = app_details_per_invocation[invocation_id]

      events_to_add = []

      for event in events:
        current_author = (event.author or _DEFAULT_AUTHOR).lower()

        if current_author == _USER_AUTHOR:
          # If the author is the user, then we just identify it and move on
          # to the next event.
          user_content = event.content
          invocation_timestamp = event.timestamp
          continue

        if event.content and event.content.parts:
          if event.is_final_response():
            final_response = event.content
          else:
            for p in event.content.parts:
              if p.function_call or p.function_response or p.text:
                events_to_add.append(event)
                break

      invocation_events = [
          InvocationEvent(author=e.author, content=e.content)
          for e in events_to_add
      ]
      invocations.append(
          Invocation(
              invocation_id=invocation_id,
              user_content=user_content,
              final_response=final_response,
              intermediate_data=InvocationEvents(
                  invocation_events=invocation_events
              ),
              creation_timestamp=invocation_timestamp,
              app_details=app_details,
          )
      )

    return invocations

  @staticmethod
  def _get_app_details_by_invocation_id(
      events: list[Event], request_intercepter: _RequestIntercepterPlugin
  ) -> dict[str, AppDetails]:
    """Creates an AppDetails object from the list of events."""
    events_by_invocation_id = (
        EvaluationGenerator._collect_events_by_invocation_id(events)
    )
    app_details_by_invocation_id = {}

    for invocation_id, events in events_by_invocation_id.items():
      app_details = AppDetails(agent_details={})
      app_details_by_invocation_id[invocation_id] = app_details

      for event in events:
        if event.author == _USER_AUTHOR:
          continue

        llm_request = request_intercepter.get_model_request(event)

        if not llm_request:
          continue

        if event.author not in app_details.agent_details:
          agent_name = event.author
          app_details.agent_details[agent_name] = AgentDetails(
              name=agent_name,
              instructions=llm_request.config.system_instruction,
              tool_declarations=llm_request.config.tools or [],
          )

    return app_details_by_invocation_id

  @staticmethod
  def _collect_events_by_invocation_id(events: list[Event]) -> dict[str, Event]:
    # Group Events by invocation id. Events that share the same invocation id
    # belong to the same invocation.
    events_by_invocation_id: dict[str, list[Event]] = {}

    for event in events:
      invocation_id = event.invocation_id

      if invocation_id not in events_by_invocation_id:
        events_by_invocation_id[invocation_id] = []

      events_by_invocation_id[invocation_id].append(event)

    return events_by_invocation_id

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
