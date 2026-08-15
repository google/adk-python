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
from typing import Callable
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
import uuid

from google.genai import types
from google.genai.types import Content
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import ConnectionClosedOK

from ..agents.base_agent import BaseAgent
from ..agents.callback_context import CallbackContext
from ..agents.invocation_context import InvocationContext
from ..agents.live_request_queue import LiveRequestQueue
from ..agents.llm_agent import Agent
from ..agents.readonly_context import ReadonlyContext
from ..agents.run_config import RunConfig
from ..agents.run_config import StreamingMode
from ..apps.app import App
from ..artifacts.base_artifact_service import BaseArtifactService
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..events.event import Event
from ..flows.llm_flows.functions import handle_function_calls_live
from ..memory.base_memory_service import BaseMemoryService
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..models.llm_request import LlmRequest
from ..plugins.base_plugin import BasePlugin
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from ..utils.context_utils import Aclosing
from ..workflow import BaseNode
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

if TYPE_CHECKING:
  from types import TracebackType

logger = logging.getLogger("google_adk." + __name__)

_USER_AUTHOR = "user"
_DEFAULT_AUTHOR = "agent"

# Function calls that end the agent and hand off instead of continuing the turn
# with a tool response, so their `turn_complete` is real. See
# `_consume_node_events`.
_TURN_ENDING_FUNCTION_CALLS = frozenset({
    "finish_task",
    "transfer_to_agent",
    "task_completed",
})

# Chunk size for streaming audio blobs to the Live API.
# See https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api#technical-specifications
_AUDIO_CHUNK_BYTES = 16000

# WebSocket close code for a normal (successful) closure.
_WEBSOCKET_NORMAL_CLOSURE_CODE = 1000

# Live run config shared by all live drivers. Server-side voice-activity
# detection is disabled so turn boundaries are controlled explicitly via
# activity markers around the sent audio.
_LIVE_RUN_CONFIG = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    response_modalities=["AUDIO"],
    output_audio_transcription=types.AudioTranscriptionConfig(),
    input_audio_transcription=types.AudioTranscriptionConfig(),
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            disabled=True
        )
    ),
)


def _send_audio_to_live(
    live_request_queue: LiveRequestQueue, content: Content
) -> None:
  """Streams a user turn's audio to the Live API as realtime input."""
  live_request_queue.send_activity_start()
  for part in content.parts or []:
    blob = part.inline_data
    if not (blob and blob.data):
      continue
    for start in range(0, len(blob.data), _AUDIO_CHUNK_BYTES):
      chunk = blob.data[start : start + _AUDIO_CHUNK_BYTES]
      live_request_queue.send_realtime(
          types.Blob(data=chunk, mime_type=blob.mime_type)
      )
  live_request_queue.send_activity_end()


async def _get_or_create_eval_session(
    session_service: BaseSessionService,
    initial_session: Optional[SessionInput],
    fallback_session_id: Optional[str],
) -> Session:
  """Returns the session an eval case runs in."""
  app_name = (
      initial_session.app_name if initial_session else "EvaluationGenerator"
  )
  user_id = initial_session.user_id if initial_session else "test_user_id"
  pinned_session_id = initial_session.session_id if initial_session else None

  if pinned_session_id:
    # A pinned id may name a session the caller prepared, so reuse it instead
    # of replacing it; `initial_session.state` then applies only on create.
    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=pinned_session_id
    )
    if session:
      return session

  return await session_service.create_session(
      app_name=app_name,
      user_id=user_id,
      state=initial_session.state if initial_session else {},
      session_id=pinned_session_id or fallback_session_id or str(uuid.uuid4()),
  )


# Keyword-argument names accepted by `Runner`, used when building the eval
# Runner kwargs so the strings are not duplicated at each call site.
_APP_NAME_KEY = "app_name"
_AGENT_KEY = "agent"
_PLUGINS_KEY = "plugins"
_APP_KEY = "app"


def _build_eval_runner_kwargs(
    root_agent: BaseAgent | BaseNode,
    app_name: str,
    app: Optional[App],
    internal_eval_plugins: list[BasePlugin],
) -> dict[str, Any]:
  """Returns the Runner kwargs used to evaluate `root_agent`.

  When `app` is provided, the Runner is built from a copy of the App with the
  internal eval plugins merged into `app.plugins`, so the App's
  `context_cache_config`, `resumability_config`, and any other
  application-wide configuration participate in the eval run. The copy leaves
  the caller's App instance untouched, and `root_agent` is overridden so the
  Runner targets the agent the caller asked to evaluate, which may be a
  sub-agent. When `app` is None, the Runner is built from the bare
  `root_agent` with only the internal eval plugins.
  """
  if app is None:
    return {
        _APP_NAME_KEY: app_name,
        _AGENT_KEY: root_agent,
        _PLUGINS_KEY: internal_eval_plugins,
    }

  runner_app = app.model_copy(
      update={
          "plugins": list(app.plugins) + internal_eval_plugins,
          "root_agent": root_agent,
      }
  )
  return {_APP_KEY: runner_app, _APP_NAME_KEY: app_name}


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
    self.event_queue: asyncio.Queue[Event] = asyncio.Queue()
    self.turn_complete_event = asyncio.Event()
    self.live_finished = asyncio.Event()
    self.current_invocation_id = Event.new_id()
    self.consume_task: Optional[asyncio.Task[None]] = None

  async def __aenter__(self) -> _LiveSession:
    """Starts the background task."""
    self.consume_task = asyncio.create_task(self._consume_events())
    return self

  async def _consume_events(self) -> None:
    """Background task: consume events from run_live."""
    try:
      # Workflows have no _llm_flow/run_live; drive through Runner.run_live
      if isinstance(self.runner.agent, BaseNode) and not isinstance(
          self.runner.agent, BaseAgent
      ):
        await self._consume_node_events()
        return

      run_config = _LIVE_RUN_CONFIG

      # Non-agent nodes are already routed to _consume_node_events above, so the
      # root here is a BaseAgent and the resolved agent is expected to be an
      # LlmAgent driven via _llm_flow.
      root_agent = self.runner.agent
      if not isinstance(root_agent, BaseAgent):
        raise TypeError("Live evaluation requires a root agent or workflow.")

      invocation_context = self.runner._new_invocation_context_for_live(
          self.session,
          live_request_queue=self.live_request_queue,
          run_config=run_config,
      )
      agent_to_run = self.runner._find_agent_to_run(self.session, root_agent)
      if not isinstance(agent_to_run, Agent):
        raise TypeError(
            f"Cannot drive {type(agent_to_run).__name__} via the LlmAgent live"
            " flow."
        )
      invocation_context.agent = agent_to_run

      callback_context = None
      llm_request = LlmRequest()

      async with Aclosing(
          agent_to_run._llm_flow._preprocess_async(
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
      async with Aclosing(agent_to_run.run_live(invocation_context)) as agen:
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
                agent=root_agent,
                session=self.session,
                run_config=run_config,
            )

            if isinstance(self.runner.agent, Agent):
              resolved_tools = await self.runner.agent.canonical_tools(
                  ReadonlyContext(inv_context)
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

  async def _consume_node_events(self) -> None:
    """Drives a non-Agent `BaseNode` (e.g. `Workflow`) via `Runner.run_live`."""
    from google.genai import errors

    # TODO: Remove once the live flow fires before/after_model_callback natively.
    callback_context_by_author = await self._record_node_app_details()

    # Track tool-call turns so the user simulator isn't prompted before the
    # agent has actually finished responding.
    in_function_call_loop = False

    try:
      async with Aclosing(
          self.runner.run_live(
              user_id=self.user_id,
              session_id=self.session_id,
              live_request_queue=self.live_request_queue,
              run_config=_LIVE_RUN_CONFIG,
          )
      ) as agen:
        async for event in agen:
          event.invocation_id = self.current_invocation_id
          callback_context = callback_context_by_author.get(event.author)
          if callback_context is not None:
            await self.runner.plugin_manager.run_after_model_callback(
                callback_context=callback_context,
                llm_response=event,
            )
          await self.event_queue.put(event)
          # Terminal/handoff calls end the agent; the next `turn_complete` is
          # real, so they must not arm the guard.
          if any(
              fc.name not in _TURN_ENDING_FUNCTION_CALLS
              for fc in event.get_function_calls()
          ):
            in_function_call_loop = True
          if event.turn_complete and event.author != _USER_AUTHOR:
            if not in_function_call_loop:
              self.turn_complete_event.set()
            else:
              in_function_call_loop = False
    except (ConnectionClosed, errors.APIError) as e:
      # A clean session close ends the stream; keep the transcript so far
      # instead of failing the eval case.
      if not self._is_normal_closure(e):
        raise
      logger.info("Ignored WebSocket normal closure exception: %s", e)

  @staticmethod
  def _is_normal_closure(exc: BaseException) -> bool:
    """Reports whether an exception is a normal Live WebSocket closure (1000)."""
    from google.genai import errors

    return isinstance(exc, ConnectionClosedOK) or (
        isinstance(exc, errors.APIError)
        and exc.code == _WEBSOCKET_NORMAL_CLOSURE_CODE
    )

  @staticmethod
  async def _record_app_details_for_agent(
      invocation_context: InvocationContext,
  ) -> CallbackContext:
    """Records the agent's live request so the autorater can score it.

    By default, live API calls do not fire before_model_callback, but the
    plugins rely on it to capture the agent instructions and tool declarations
    that the autorater needs. We run the callback manually here and return the
    callback context so callers can replay after_model_callback per event.

    TODO: Remove once the live flow fires before/after_model_callback natively.
    """
    agent = invocation_context.agent
    if not isinstance(agent, Agent):
      raise TypeError(
          f"Cannot record app details for {type(agent).__name__}; an LlmAgent"
          " is required."
      )
    llm_request = LlmRequest()
    async with Aclosing(
        agent._llm_flow._preprocess_async(invocation_context, llm_request)
    ) as agen:
      async for _ in agen:
        pass

    callback_context = CallbackContext(invocation_context)
    await invocation_context.plugin_manager.run_before_model_callback(
        callback_context=callback_context,
        llm_request=llm_request,
    )
    return callback_context

  async def _record_node_app_details(self) -> dict[str, CallbackContext]:
    """Records live requests for each agent in a node graph root.

    Multi-agent node roots serve several agents over one live stream, so we
    record each agent's request up front and return a {author: callback_context}
    map the caller uses to fire after_model_callback for each agent's events. A
    failure for one agent is logged and skipped so it never aborts the eval run.

    Scope: only top-level ``graph.nodes`` agents are recorded; agents nested in
    sub-workflows or wrapper nodes are covered once native Live callbacks land.

    TODO: Remove once the live flow fires before/after_model_callback natively.
    """
    callback_context_by_author: dict[str, CallbackContext] = {}

    graph = getattr(self.runner.agent, "graph", None)
    if graph is None:
      return callback_context_by_author

    base_invocation_context = self.runner._new_invocation_context_for_live(
        self.session,
        live_request_queue=self.live_request_queue,
        run_config=_LIVE_RUN_CONFIG,
    )

    for node in graph.nodes:
      if not isinstance(node, Agent):
        continue
      try:
        invocation_context = base_invocation_context.model_copy(
            update={"agent": node}
        )
        callback_context_by_author[node.name] = (
            await self._record_app_details_for_agent(invocation_context)
        )
      except Exception:  # pylint: disable=broad-except
        logger.warning(
            "Failed to record app details for agent %s.",
            node.name,
            exc_info=True,
        )

    return callback_context_by_author

  async def __aexit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: TracebackType | None,
  ) -> None:
    """Closes the queue and waits for the background task to finish."""
    from google.genai import errors

    self.live_request_queue.close()
    consume_task = self.consume_task
    if consume_task is None:
      raise RuntimeError("Live session was exited before it was started.")
    try:
      await asyncio.wait_for(consume_task, timeout=30)
    except asyncio.TimeoutError:
      logger.warning("Timed out waiting for run_live to finish.")
      consume_task.cancel()
      try:
        await consume_task
      except asyncio.CancelledError:
        pass
    except (ConnectionClosed, errors.APIError) as e:
      # The Gemini Live API uses WebSockets. When the session ends normally, the
      # connection is closed with code 1000. Some client libraries may raise an
      # exception rather than handling it silently. We log this as INFO to
      # avoid false-positive error reports for expected behavior.
      if self._is_normal_closure(e):
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
      agent_name: Optional[str] = None,
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
  def generate_responses_from_session(
      session_path: str,
      eval_dataset: list[list[dict[str, object]]],
  ) -> list[list[dict[str, object]]]:
    """Returns evaluation responses by combining session data with eval data.

    Args:
      session_path: Path to a json file that contains session data.
      eval_dataset: The eval data set that should be combined with the session
        data.
    """
    results = []

    with open(session_path, "r", encoding="utf-8") as f:
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
    agent_package = getattr(agent_module, "agent", None)
    # Prefer the wrapping `App` when the module exposes one, so that
    # `app.plugins`, context-cache, and resumability configs participate
    # in eval runs the same way they do for `adk web` / `adk run`.
    app_obj = getattr(agent_package, "app", None)
    if isinstance(app_obj, App):
      root_agent = app_obj.root_agent
    else:
      app_obj = None
      root_agent = getattr(agent_package, "root_agent", None)
    if root_agent is None:
      # Matches the original `agent_module.agent.root_agent` attribute access,
      # which raised when the module exposed no root. A BaseNode root is a
      # supported App.root_agent value, so it is not rejected here.
      raise TypeError(
          f"Module {module_name!r} does not expose agent.root_agent."
      )
    root_agent = cast(BaseAgent, root_agent)

    reset_candidate = getattr(agent_package, "reset_data", None)
    reset_func: Optional[Callable[[], object]] = None
    if reset_candidate is not None:
      if not callable(reset_candidate):
        raise TypeError("agent.reset_data must be callable when provided.")
      reset_func = cast(Callable[[], object], reset_candidate)

    agent_to_evaluate = root_agent
    if agent_name:
      selected_agent = root_agent.find_agent(agent_name)
      if selected_agent is None:
        raise ValueError(f"Sub-Agent {agent_name!r} not found.")
      agent_to_evaluate = selected_agent

    return await EvaluationGenerator._generate_inferences_from_root_agent(
        agent_to_evaluate,
        user_simulator=user_simulator,
        reset_func=reset_func,
        initial_session=initial_session,
        app=app_obj,
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
  ) -> AsyncGenerator[Event, None]:
    """Generates inferences for a single user invocation in live mode."""
    yield Event(
        content=user_message,
        author=_USER_AUTHOR,
        invocation_id=current_invocation_id,
    )

    # If the user message contains audio parts, send only the audio to the
    # agent so a native-audio Live model receives audio-only input. The full
    # Content (with text) is preserved in the Event above for trajectory
    # logging and autorater evaluation.
    has_audio = any(p.inline_data for p in user_message.parts or [])
    if has_audio:
      _send_audio_to_live(live_request_queue, user_message)
    else:
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

    # Yield raw events; transcription-bearing events are normalized later by
    # `_normalize_live_transcriptions` before they are consumed.
    while not event_queue.empty():
      event = await event_queue.get()
      if event.invocation_id == current_invocation_id:
        yield event

  @staticmethod
  async def _generate_inferences_from_root_agent_live(
      root_agent: BaseAgent | BaseNode,
      user_simulator: UserSimulator,
      reset_func: Optional[Callable[[], object]] = None,
      initial_session: Optional[SessionInput] = None,
      session_id: Optional[str] = None,
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      memory_service: Optional[BaseMemoryService] = None,
      live_timeout_seconds: int = DEFAULT_LIVE_TIMEOUT_SECONDS,
      app: Optional[App] = None,
  ) -> list[Invocation]:
    """Scrapes the root agent in coordination with the user simulator in live mode.

    Mirrors `_generate_inferences_from_root_agent`: when `app` is provided the
    Runner carries the App's plugins and configuration, otherwise the bare
    `root_agent` is used.
    """
    if not session_service:
      session_service = InMemorySessionService()

    if not memory_service:
      memory_service = InMemoryMemoryService()

    session = await _get_or_create_eval_session(
        session_service, initial_session, session_id
    )
    app_name = session.app_name
    user_id = session.user_id
    session_id = session.id

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
    runner_kwargs = _build_eval_runner_kwargs(
        root_agent=root_agent,
        app_name=app_name,
        app=app,
        internal_eval_plugins=[
            request_intercepter_plugin,
            ensure_retry_options_plugin,
        ],
    )

    async with Runner(
        **runner_kwargs,
        artifact_service=artifact_service,
        session_service=session_service,
        memory_service=memory_service,
    ) as runner:
      events: list[Event] = []

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
              EvaluationGenerator._normalize_live_transcriptions(
                  copy.deepcopy(events)
              )
          )
          if next_user_message.status == UserSimulatorStatus.SUCCESS:
            user_message = next_user_message.user_message
            if user_message is None:
              raise RuntimeError(
                  "A successful user-simulator result must include a message."
              )
            live_session.current_invocation_id = Event.new_id()
            live_session.turn_complete_event.clear()

            logger.info("Waiting for model to complete turn %d...", turn_idx)

            async for (
                event
            ) in EvaluationGenerator._generate_inferences_for_single_user_invocation_live(
                live_request_queue=live_session.live_request_queue,
                event_queue=live_session.event_queue,
                user_message=user_message,
                current_invocation_id=live_session.current_invocation_id,
                turn_complete_event=live_session.turn_complete_event,
                live_timeout_seconds=live_timeout_seconds,
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
          EvaluationGenerator._normalize_live_transcriptions(events),
          app_details_by_invocation_id,
      )

  @staticmethod
  async def _generate_inferences_from_root_agent(
      root_agent: BaseAgent | BaseNode,
      user_simulator: UserSimulator,
      reset_func: Optional[Callable[[], object]] = None,
      initial_session: Optional[SessionInput] = None,
      session_id: Optional[str] = None,
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      memory_service: Optional[BaseMemoryService] = None,
      app: Optional[App] = None,
  ) -> list[Invocation]:
    """Scrapes the root agent in coordination with the user simulator.

    If `app` is provided, the eval Runner is built from a copy of the App
    with internal eval plugins merged into `app.plugins`, preserving the
    App's `context_cache_config`, `resumability_config`, and any other
    application-wide configuration. Otherwise the Runner is built from
    the bare `root_agent` with only the internal eval plugins, matching
    the legacy behavior.
    """

    if not session_service:
      session_service = InMemorySessionService()

    if not memory_service:
      memory_service = InMemoryMemoryService()

    session = await _get_or_create_eval_session(
        session_service, initial_session, session_id
    )
    app_name = session.app_name
    user_id = session.user_id
    session_id = session.id

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
    runner_kwargs = _build_eval_runner_kwargs(
        root_agent=root_agent,
        app_name=app_name,
        app=app,
        internal_eval_plugins=[
            request_intercepter_plugin,
            ensure_retry_options_plugin,
        ],
    )

    async with Runner(
        **runner_kwargs,
        artifact_service=artifact_service,
        session_service=session_service,
        memory_service=memory_service,
    ) as runner:
      events: list[Event] = []
      while True:
        next_user_message = await user_simulator.get_next_user_message(
            copy.deepcopy(events)
        )
        if next_user_message.status == UserSimulatorStatus.SUCCESS:
          user_message = next_user_message.user_message
          if user_message is None:
            raise RuntimeError(
                "A successful user-simulator result must include a message."
            )
          async for (
              event
          ) in EvaluationGenerator._generate_inferences_for_single_user_invocation(
              runner, user_id, session_id, user_message
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
      final_response: Optional[Content] = None
      final_event: Optional[Event] = None
      user_content = Content(parts=[])
      invocation_timestamp: float = 0
      app_details = None
      if (
          app_details_per_invocation
          and invocation_id in app_details_per_invocation
      ):
        app_details = app_details_per_invocation[invocation_id]

      events_to_add: list[Event] = []
      for event in events:
        current_author = (event.author or _DEFAULT_AUTHOR).lower()

        if current_author == _USER_AUTHOR:
          # If the author is the user, then we just identify it and move on
          # to the next event.
          if event.content is not None:
            user_content = event.content
            invocation_timestamp = event.timestamp
          continue

        if event.content and event.content.parts:
          if event.is_final_response():
            # A live response is both audio and a text transcript; keep the
            # text one as the gradable response.
            final_has_text = final_response is not None and any(
                p.text for p in final_response.parts or []
            )
            event_has_text = any(p.text for p in event.content.parts or [])
            if not final_has_text or event_has_text:
              final_response = event.content
              final_event = event

          should_add_event = event.grounding_metadata is not None
          for p in event.content.parts:
            if (
                p.function_call
                or p.function_response
                or p.text
                or p.inline_data
            ):
              should_add_event = True
              break
          if should_add_event:
            events_to_add.append(event)
        elif event.grounding_metadata is not None:
          events_to_add.append(event)

      invocation_events = []
      for e in events_to_add:
        # Keep the final event only when it carries tool calls (so the judge
        # still sees the function call) or grounding metadata; every other
        # event is always included.
        if (
            final_event is not None
            and e is final_event
            and not e.get_function_calls()
            and not e.grounding_metadata
        ):
          continue
        invocation_events.append(
            InvocationEvent(
                author=e.author,
                content=(
                    e.content
                    if e is not final_event or e.get_function_calls()
                    else None
                ),
                grounding_metadata=e.grounding_metadata,
            )
        )
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
  def _normalize_live_transcriptions(events: list[Event]) -> list[Event]:
    """Rewrites native-audio Live transcription events into text content events."""
    # Only consolidated (non-partial) transcription events are rewritten,
    # mirroring `contents.py`; every other event passes through untouched.
    normalized: list[Event] = []
    for event in events:
      if event.content is not None or event.partial:
        normalized.append(event)
        continue

      if event.input_transcription and event.input_transcription.text:
        transcription = event.input_transcription
        role = "user"
      elif event.output_transcription and event.output_transcription.text:
        transcription = event.output_transcription
        role = "model"
      else:
        normalized.append(event)
        continue

      rewritten = event.model_copy(deep=True)
      rewritten.input_transcription = None
      rewritten.output_transcription = None
      rewritten.content = Content(
          role=role, parts=[types.Part(text=transcription.text)]
      )
      normalized.append(rewritten)

    return normalized

  @staticmethod
  def _collect_events_by_invocation_id(
      events: list[Event],
  ) -> dict[str, list[Event]]:
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
  def _process_query_with_session(
      session_data: Session,
      data: list[dict[str, object]],
  ) -> list[dict[str, object]]:
    """Process the queries using the existing session data without invoking the runner."""
    responses = data.copy()

    # Iterate through the provided queries and align them with the session
    # events
    for index, eval_entry in enumerate(responses):
      query = eval_entry.get("query")
      if not isinstance(query, str):
        raise ValueError("Each evaluation entry must contain a string query.")
      actual_tool_uses: list[dict[str, object]] = []
      response: Optional[str] = None

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
              content = subsequent_event.content
              if content is None or not content.parts:
                continue
              first_part = content.parts[0]
              # Extract tool usage
              if first_part.function_call:
                call = first_part.function_call
                actual_tool_uses.append(
                    {"tool_name": call.name, "tool_input": call.args}
                )
              # Extract final response
              elif subsequent_event.author != "user":
                response = first_part.text

      # Update the results for the current query
      responses[index]["actual_tool_use"] = actual_tool_uses
      responses[index]["response"] = response
    return responses
