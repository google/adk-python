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

import asyncio
import copy
import importlib
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Optional
import uuid

from google.genai import types
from google.genai.types import Content
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import ConnectionClosedOK

from ..agents.callback_context import CallbackContext
from ..agents.invocation_context import InvocationContext
from ..agents.live_request_queue import LiveRequestQueue
from ..agents.llm_agent import Agent
from ..agents.run_config import RunConfig
from ..agents.run_config import StreamingMode
from ..artifacts.base_artifact_service import BaseArtifactService
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..events.event import Event
from ..flows.llm_flows.functions import handle_function_calls_live
from ..memory.base_memory_service import BaseMemoryService
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..models.llm_request import LlmRequest
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from ..utils.context_utils import Aclosing
from ._retry_options_utils import EnsureRetryOptionsPlugin
from .app_details import AgentDetails
from .app_details import AppDetails
from .constants import DEFAULT_LIVE_TIMEOUT_SECONDS
from .eval_case import EvalCase
from .eval_case import Invocation
from .eval_case import InvocationEvent
from .eval_case import InvocationEvents
from .eval_case import SessionInput
from .eval_set import EvalSet
from .request_intercepter_plugin import _RequestIntercepterPlugin
from .simulation.user_simulator import BaseUserSimulatorConfig
from .simulation.user_simulator import Status as UserSimulatorStatus
from .simulation.user_simulator import UserSimulator
from .simulation.user_simulator_provider import UserSimulatorProvider

logger = logging.getLogger("google_adk." + __name__)

_USER_AUTHOR = "user"
_DEFAULT_AUTHOR = "agent"


class EvalCaseResponses(BaseModel):
  """Contains multiple responses associated with an EvalCase.

  Multiple responses are a result of repeated requests to generate inferences.
  """

  eval_case: EvalCase
  responses: list[list[Invocation]]


class _LiveSession:
  """Manages the background task and state for a live session."""

  def __init__(
      self,
      runner: Runner,
      session: Session,
      user_id: str,
      session_id: str,
  ):
    self.runner = runner
    self.session = session
    self.user_id = user_id
    self.session_id = session_id
    self.live_request_queue = LiveRequestQueue()
    self.event_queue = asyncio.Queue()
    self.turn_complete_event = asyncio.Event()
    self.live_finished = asyncio.Event()
    self.current_invocation_id = Event.new_id()
    self.consume_task = None

  async def __aenter__(self) -> _LiveSession:
    """Starts the background task."""
    self.consume_task = asyncio.create_task(self._consume_events())
    return self

  async def _consume_events(self) -> None:
    """Background task: consume events from run_live."""
    try:
      run_config = RunConfig(
          streaming_mode=StreamingMode.BIDI,
          response_modalities=["AUDIO"],
          output_audio_transcription=types.AudioTranscriptionConfig(),
          input_audio_transcription=types.AudioTranscriptionConfig(),
      )

      invocation_context = self.runner._new_invocation_context_for_live(
          self.session,
          live_request_queue=self.live_request_queue,
          run_config=run_config,
      )
      invocation_context.agent = self.runner._find_agent_to_run(
          self.session, self.runner.agent
      )

      callback_context = None
      llm_request = LlmRequest()

      async with Aclosing(
          invocation_context.agent._llm_flow._preprocess_async(
              invocation_context, llm_request
          )
      ) as agen:
        async for _ in agen:
          pass

      callback_context = CallbackContext(invocation_context)
      # By default, live API calls do not include before_model_callback and
      # after_model_callback. These callbacks are needed by the plugins to
      # include the agent instructions and tool declarations in the eval
      # invocations for autorater evaluation.
      await invocation_context.plugin_manager.run_before_model_callback(
          callback_context=callback_context,
          llm_request=llm_request,
      )

      in_function_call_loop = False
      async with Aclosing(
          invocation_context.agent.run_live(invocation_context)
      ) as agen:
        async for event in agen:
          assert event is not None
          event.invocation_id = self.current_invocation_id
          if callback_context:
            await invocation_context.plugin_manager.run_after_model_callback(
                callback_context=callback_context,
                llm_response=event,
            )
          await self.event_queue.put(event)
          if not event.partial:
            await self.runner.session_service.append_event(
                session=self.session, event=event
            )
          function_calls = event.get_function_calls()
          if function_calls:
            in_function_call_loop = True
            inv_context = InvocationContext(
                session_service=self.runner.session_service,
                invocation_id=event.invocation_id,
                agent=self.runner.agent,
                session=self.session,
                run_config=run_config,
            )

            if isinstance(self.runner.agent, Agent):
              resolved_tools = await self.runner.agent.canonical_tools(
                  inv_context
              )
              tools_dict = {t.name: t for t in resolved_tools}
            else:
              tools_dict = {}

            try:
              response_event = await handle_function_calls_live(
                  invocation_context=inv_context,
                  function_call_event=event,
                  tools_dict=tools_dict,
              )

              if (
                  response_event
                  and response_event.content
                  and response_event.content.parts
              ):
                for part in response_event.content.parts:
                  if part.function_response:
                    tool_content = types.Content(
                        role="tool",
                        parts=[part],
                    )
                    self.live_request_queue.send_content(tool_content)
            except (ValueError, RuntimeError, KeyError, TypeError) as e:
              logger.error(
                  "Failed to handle function calls: %s",
                  e,
                  exc_info=True,
              )
              for fc in function_calls:
                response_content = types.FunctionResponse(
                    name=fc.name,
                    id=fc.id,
                    response={"error": str(e)},
                )
                tool_content = types.Content(
                    role="tool",
                    parts=[types.Part(function_response=response_content)],
                )
                self.live_request_queue.send_content(tool_content)
          if event.turn_complete and event.author != _USER_AUTHOR:
            if not in_function_call_loop:
              self.turn_complete_event.set()
            else:
              in_function_call_loop = False
    finally:
      self.live_finished.set()
      self.turn_complete_event.set()  # Unblock any waiters

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Closes the queue and waits for the background task to finish."""
    from google.genai import errors

    self.live_request_queue.close()
    try:
      await asyncio.wait_for(self.consume_task, timeout=30)
    except asyncio.TimeoutError:
      logger.warning("Timed out waiting for run_live to finish.")
      assert self.consume_task is not None
      self.consume_task.cancel()
      try:
        await self.consume_task
      except asyncio.CancelledError:
        pass
    except (ConnectionClosed, errors.APIError) as e:
      # The Gemini Live API uses WebSockets. When the session ends normally, the
      # connection is closed with code 1000. Some client libraries may raise an
      # exception rather than handling it silently. We log this as INFO to
      # avoid false-positive error reports for expected behavior.
      is_normal_closure = isinstance(e, ConnectionClosedOK) or (
          isinstance(e, errors.APIError) and e.code == 1000
      )

      if is_normal_closure:
        logger.info("Ignored WebSocket normal closure exception: %s", e)
      else:
        raise


class EvaluationGenerator:
  """Generates evaluation responses for agents."""

  @staticmethod
  async def generate_responses(
      eval_set: EvalSet,
      agent_module_path: str,
      repeat_num: int = 3,
      agent_name: str = None,
      user_simulator_config: Optional[BaseUserSimulatorConfig] = None,
  ) -> list[EvalCaseResponses]:
    """Returns evaluation responses for the given dataset and agent.

    Args:
      eval_set: The eval set that needs to be scraped for responses.
      agent_module_path: Path to the module that contains the root agent.
      repeat_num: Number of time the eval dataset should be repeated. This is
        usually done to remove uncertainty that a single run may bring.
      agent_name: The name of the agent that should be evaluated. This is
        usually the sub-agent.
      user_simulator_config: Optional configuration for the user simulator.
        Only relevant for eval cases that use a `conversation_scenario` (which
        are driven by `LlmBackedUserSimulator`); ignored for static
        conversations. Pass an `LlmBackedUserSimulatorConfig` to override the
        user-simulation model, max invocations, or custom instructions.
    """
    results = []

    for eval_case in eval_set.eval_cases:
      responses = []
      for _ in range(repeat_num):
        user_simulator = UserSimulatorProvider(
            user_simulator_config=user_simulator_config
        ).provide(eval_case)
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
      logger.info("Loaded session %s", session_path)

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
  async def _generate_inferences_for_single_user_invocation_live(
      live_request_queue: LiveRequestQueue,
      event_queue: asyncio.Queue[Event],
      user_message: Content,
      current_invocation_id: str,
      turn_complete_event: asyncio.Event,
      live_timeout_seconds: int,
      agent_name: str = _DEFAULT_AUTHOR,
  ) -> AsyncGenerator[Event, None]:
    """Generates inferences for a single user invocation in live mode."""
    yield Event(
        content=user_message,
        author=_USER_AUTHOR,
        invocation_id=current_invocation_id,
    )

    live_request_queue.send_content(user_message)

    try:
      await asyncio.wait_for(
          turn_complete_event.wait(),
          timeout=live_timeout_seconds,
      )
    except asyncio.TimeoutError:
      logger.warning(
          "Timed out waiting for model turn completion in live mode."
      )
      raise

    while not event_queue.empty():
      event = await event_queue.get()
      if event.invocation_id == current_invocation_id:
        yield event
        # Emit a synthetic text event for each transcription, preserving
        # the order in which events are received.
        if (
            event.author != _USER_AUTHOR
            and event.output_transcription
            and event.output_transcription.text
            and event.partial
        ):
          yield Event(
              content=Content(
                  role="model",
                  parts=[types.Part(text=event.output_transcription.text)],
              ),
              author=agent_name,
              invocation_id=current_invocation_id,
          )

  @staticmethod
  async def _generate_inferences_from_root_agent_live(
      root_agent: Agent,
      user_simulator: UserSimulator,
      reset_func: Optional[Any] = None,
      initial_session: Optional[SessionInput] = None,
      session_id: Optional[str] = None,
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      memory_service: Optional[BaseMemoryService] = None,
      live_timeout_seconds: int = DEFAULT_LIVE_TIMEOUT_SECONDS,
  ) -> list[Invocation]:
    """Scrapes the root agent in coordination with the user simulator in live mode."""
    if not session_service:
      session_service = InMemorySessionService()

    if not memory_service:
      memory_service = InMemoryMemoryService()

    app_name = (
        initial_session.app_name if initial_session else "EvaluationGenerator"
    )
    user_id = initial_session.user_id if initial_session else "test_user_id"
    session_id = session_id if session_id else str(uuid.uuid4())

    session = await session_service.create_session(
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

    # We ensure that there is some kind of retries on the llm_requests that are
    # generated from the Agent. This is done to make inferencing step of evals
    # more resilient to temporary model failures.
    ensure_retry_options_plugin = EnsureRetryOptionsPlugin(
        name="ensure_retry_options"
    )
    request_intercepter_plugin = _RequestIntercepterPlugin(
        name="request_intercepter_plugin"
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

      # `_LiveSession` is a runtime connection manager wrapping the `Session`
      # data model (which stores conversation history/state). It manages the
      # active bidirectional WebSocket stream and background consumer tasks.
      live_session = _LiveSession(runner, session, user_id, session_id)
      await live_session.__aenter__()

      try:
        turn_idx = 0
        while True:
          turn_idx += 1
          next_user_message = await user_simulator.get_next_user_message(
              copy.deepcopy(events)
          )
          if next_user_message.status == UserSimulatorStatus.SUCCESS:
            live_session.current_invocation_id = Event.new_id()
            live_session.turn_complete_event.clear()

            logger.info("Waiting for model to complete turn %d...", turn_idx)

            async for (
                event
            ) in EvaluationGenerator._generate_inferences_for_single_user_invocation_live(
                live_request_queue=live_session.live_request_queue,
                event_queue=live_session.event_queue,
                user_message=next_user_message.user_message,
                current_invocation_id=live_session.current_invocation_id,
                turn_complete_event=live_session.turn_complete_event,
                live_timeout_seconds=live_timeout_seconds,
                agent_name=runner.agent.name,
            ):
              events.append(event)

            if live_session.live_finished.is_set():
              logger.info("Live session finished signal detected.")
              break
          else:  # no message generated
            break
      finally:
        await live_session.__aexit__(None, None, None)

      app_details_by_invocation_id = (
          EvaluationGenerator._get_app_details_by_invocation_id(
              events, request_intercepter_plugin
          )
      )
      return EvaluationGenerator.convert_events_to_eval_invocations(
          events, app_details_by_invocation_id
      )

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
      final_event: Optional[Event] = None
      user_content = Content(parts=[])
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
            final_event = event

          for p in event.content.parts:
            if (
                p.function_call
                or p.function_response
                or p.text
                or p.inline_data
            ):
              events_to_add.append(event)
              break

      invocation_events = [
          InvocationEvent(author=e.author, content=e.content)
          for e in events_to_add
          if final_event is None
          or e is not final_event
          or e.get_function_calls()
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
