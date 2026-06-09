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

import asyncio
import contextlib
import copy
from typing import Any
from typing import AsyncGenerator
from typing import Generator
from typing import Optional

from google.adk.agents.context import Context as WorkflowContext
from google.adk.agents.invocation_context import InvocationContext as BaseInvocationContext
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.base_llm_connection import BaseLlmConnection
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.runners import InMemoryRunner as AfInMemoryRunner
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from google.genai.types import Part
from typing_extensions import override


def create_test_agent(name: str = 'test_agent') -> LlmAgent:
  """Create a simple test agent for use in unit tests.

  Args:
    name: The name of the test agent.

  Returns:
    A configured LlmAgent instance suitable for testing.
  """
  return LlmAgent(name=name)


class UserContent(types.Content):

  def __init__(self, text_or_part: str):
    parts = [
        types.Part.from_text(text=text_or_part)
        if isinstance(text_or_part, str)
        else text_or_part
    ]
    super().__init__(role='user', parts=parts)


class ModelContent(types.Content):

  def __init__(self, parts: list[types.Part]):
    super().__init__(role='model', parts=parts)


async def create_invocation_context(
    agent: Agent,
    user_content: str = '',
    run_config: RunConfig = None,
    plugins: list[BasePlugin] = [],
):
  invocation_id = 'test_id'
  artifact_service = InMemoryArtifactService()
  session_service = InMemorySessionService()
  memory_service = InMemoryMemoryService()
  invocation_context = BaseInvocationContext(
      artifact_service=artifact_service,
      session_service=session_service,
      memory_service=memory_service,
      plugin_manager=PluginManager(plugins=plugins),
      invocation_id=invocation_id,
      agent=agent,
      session=await session_service.create_session(
          app_name='test_app', user_id='test_user'
      ),
      user_content=types.Content(
          role='user', parts=[types.Part.from_text(text=user_content)]
      ),
      run_config=run_config or RunConfig(),
  )
  if user_content:
    append_user_content(
        invocation_context, [types.Part.from_text(text=user_content)]
    )
  return invocation_context


async def create_workflow_context(
    agent,
    user_content='',
) -> WorkflowContext:
  """Create a WorkflowContext for isolated node testing.

  Constructs the minimal InvocationContext and wraps it in a
  WorkflowContext so that individual nodes can be tested in
  isolation without running the full _SingleLlmAgent pipeline.
  """
  invocation_context = await create_invocation_context(agent, user_content)
  return WorkflowContext(
      invocation_context=invocation_context,
      node_path='test',
      run_id='test-execution',
  )


def append_user_content(
    invocation_context: BaseInvocationContext, parts: list[types.Part]
) -> Event:
  session = invocation_context.session
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author='user',
      content=types.Content(role='user', parts=parts),
  )
  session.events.append(event)
  return event


# Extracts the contents from the events and transform them into a list of
# (author, simplified_content) tuples.
def simplify_events(events: list[Event]) -> list[tuple[str, types.Part]]:
  res = []
  for event in events:
    if event.content:
      author = event.author
      res.append((author, simplify_content(event.content)))
  return res


END_OF_AGENT = 'end_of_agent'


# Extracts the contents from the events and transform them into a list of
# (author, simplified_content OR AgentState OR "end_of_agent") tuples.
#
# Could be used to compare events for testing resumability.
def simplify_resumable_app_events(
    events: list[Event],
) -> list[(str, types.Part | str)]:
  results = []
  for event in events:
    if event.content:
      results.append((event.author, simplify_content(event.content)))
    elif event.actions.end_of_agent:
      results.append((event.author, END_OF_AGENT))
    elif event.actions.agent_state is not None:
      agent_state = event.actions.agent_state
      if isinstance(agent_state, dict):
        nodes = agent_state.get('nodes', {})
        agent_state = {
            'node_states': {
                node_name: node_state.get('status')
                for node_name, node_state in nodes.items()
            }
        }
      results.append((event.author, agent_state))
  return results


# Simplifies the contents into a list of (author, simplified_content) tuples.
def simplify_contents(contents: list[types.Content]) -> list[(str, types.Part)]:
  return [(content.role, simplify_content(content)) for content in contents]


# Simplifies the content so it's easier to assert.
# - If there is only one part, return part
# - If the only part is pure text, return stripped_text
# - If there are multiple parts, return parts
# - remove function_call_id if it exists
def simplify_content(
    content: types.Content,
) -> str | types.Part | list[types.Part]:
  content = copy.deepcopy(content)
  for part in content.parts:
    if part.function_call and part.function_call.id:
      part.function_call.id = None
    if part.function_response and part.function_response.id:
      part.function_response.id = None
  if len(content.parts) == 1:
    if content.parts[0].text:
      return content.parts[0].text.strip()
    else:
      return content.parts[0]
  return content.parts


def get_user_content(message: types.ContentUnion) -> types.Content:
  return message if isinstance(message, types.Content) else UserContent(message)


class TestInMemoryRunner(AfInMemoryRunner):
  """InMemoryRunner that is tailored for tests, features async run method.

  app_name is hardcoded as InMemoryRunner in the parent class.
  """

  async def run_async_with_new_session(
      self, new_message: types.ContentUnion
  ) -> list[Event]:

    collected_events: list[Event] = []
    async for event in self.run_async_with_new_session_agen(new_message):
      collected_events.append(event)

    return collected_events

  async def run_async_with_new_session_agen(
      self, new_message: types.ContentUnion
  ) -> AsyncGenerator[Event, None]:
    session = await self.session_service.create_session(
        app_name='InMemoryRunner', user_id='test_user'
    )
    agen = self.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=get_user_content(new_message),
    )
    async with Aclosing(agen):
      async for event in agen:
        yield event


class InMemoryRunner:
  """InMemoryRunner that is tailored for tests."""

  def __init__(
      self,
      root_agent: Optional[Agent | LlmAgent] = None,
      response_modalities: list[str] = None,
      plugins: list[BasePlugin] = [],
      app: Optional[App] = None,
      node: Any = None,
  ):
    """Initializes the InMemoryRunner.

    Args:
      root_agent: The root agent to run, won't be used if app is provided.
      response_modalities: The response modalities of the runner.
      plugins: The plugins to use in the runner, won't be used if app is
        provided.
      app: The app to use in the runner.
      node: The root node to run.
    """
    self._app = app
    if node:
      self.app_name = node.name
      self.root_agent = None
      self.runner = Runner(
          node=node,
          artifact_service=InMemoryArtifactService(),
          session_service=InMemorySessionService(),
          memory_service=InMemoryMemoryService(),
          plugins=plugins,
      )
    elif not app:
      self.app_name = 'test_app'
      self.root_agent = root_agent
      self.runner = Runner(
          app_name='test_app',
          agent=root_agent,
          artifact_service=InMemoryArtifactService(),
          session_service=InMemorySessionService(),
          memory_service=InMemoryMemoryService(),
          plugins=plugins,
      )
    else:
      self.app_name = app.name
      self.root_agent = app.root_agent
      self.runner = Runner(
          app=app,
          artifact_service=InMemoryArtifactService(),
          session_service=InMemorySessionService(),
          memory_service=InMemoryMemoryService(),
      )
    self.session_id = None

  @property
  def session(self) -> Session:
    if not self.session_id:
      session = self.runner.session_service.create_session_sync(
          app_name=self.app_name, user_id='test_user'
      )
      self.session_id = session.id
      return session
    return self.runner.session_service.get_session_sync(
        app_name=self.app_name, user_id='test_user', session_id=self.session_id
    )

  def run(self, new_message: types.ContentUnion) -> list[Event]:
    return list(
        self.runner.run(
            user_id=self.session.user_id,
            session_id=self.session.id,
            new_message=get_user_content(new_message),
        )
    )

  @property
  def is_resumable(self) -> bool:
    """Returns whether the app is configured for resumable HITL."""
    if hasattr(self, '_app') and self._app:
      cfg = getattr(self._app, 'resumability_config', None)
      return cfg is not None and cfg.is_resumable
    return False

  async def run_async(
      self,
      new_message: Optional[types.ContentUnion] = None,
      invocation_id: Optional[str] = None,
  ) -> list[Event]:
    # For non-resumable apps, don't reuse invocation_id on resume.
    # State reconstruction relies on scanning events from *previous*
    # invocations, so the resume call must get a fresh invocation_id.
    if invocation_id and not self.is_resumable:
      invocation_id = None
    events = []
    async for event in self.runner.run_async(
        user_id=self.session.user_id,
        session_id=self.session.id,
        invocation_id=invocation_id,
        new_message=get_user_content(new_message) if new_message else None,
    ):
      events.append(event)
    return events

  def run_live(
      self, live_request_queue: LiveRequestQueue, run_config: RunConfig = None
  ) -> list[Event]:
    collected_responses = []

    async def consume_responses(session: Session):
      run_res = self.runner.run_live(
          session=session,
          live_request_queue=live_request_queue,
          run_config=run_config or RunConfig(),
      )

      async for response in run_res:
        collected_responses.append(response)
        # When we have enough response, we should return
        if len(collected_responses) >= 1:
          return

    try:
      session = self.session
      asyncio.run(consume_responses(session))
    except asyncio.TimeoutError:
      print('Returning any partial results collected so far.')

    return collected_responses


class MockModel(BaseLlm):
  model: str = 'mock'

  requests: list[LlmRequest] = []
  live_blobs: list[types.Blob] = []
  live_contents: list[types.Content] = []
  responses: list[LlmResponse]
  error: Exception | None = None
  response_index: int = -1

  # Whether the mock model should wait for realtime input (blobs or content)
  # to be sent before yielding pre-defined responses in live mode.
  wait_for_realtime_input: bool = False

  @classmethod
  def create(
      cls,
      responses: (
          list[types.Part]
          | list[LlmResponse]
          | list[str]
          | list[list[types.Part]]
      ),
      error: Exception | None = None,
      wait_for_realtime_input: bool = False,
  ):
    if error and not responses:
      return cls(
          responses=[],
          error=error,
          wait_for_realtime_input=wait_for_realtime_input,
      )
    if not responses:
      return cls(responses=[], wait_for_realtime_input=wait_for_realtime_input)
    elif isinstance(responses[0], LlmResponse):
      # responses is list[LlmResponse]
      return cls(
          responses=responses, wait_for_realtime_input=wait_for_realtime_input
      )
    else:
      responses = [
          LlmResponse(content=ModelContent(item))
          if isinstance(item, list) and isinstance(item[0], types.Part)
          # responses is list[list[Part]]
          else LlmResponse(
              content=ModelContent(
                  # responses is list[str] or list[Part]
                  [Part(text=item) if isinstance(item, str) else item]
              )
          )
          for item in responses
          if item
      ]

      return cls(
          responses=responses, wait_for_realtime_input=wait_for_realtime_input
      )

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return ['mock']

  def generate_content(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> Generator[LlmResponse, None, None]:
    if self.error is not None:
      raise self.error
    # Increasement of the index has to happen before the yield.
    self.response_index += 1
    self.requests.append(llm_request)
    # yield LlmResponse(content=self.responses[self.response_index])
    yield self.responses[self.response_index]

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    if self.error is not None:
      raise self.error
    # Increasement of the index has to happen before the yield.
    self.response_index += 1
    self.requests.append(llm_request)
    yield self.responses[self.response_index]

  @contextlib.asynccontextmanager
  async def connect(self, llm_request: LlmRequest) -> BaseLlmConnection:
    """Creates a live connection to the LLM."""
    self.requests.append(llm_request)
    yield MockLlmConnection(
        self.responses,
        self,
        wait_for_realtime_input=self.wait_for_realtime_input,
    )


class MockLlmConnection(BaseLlmConnection):

  def __init__(
      self,
      llm_responses: list[LlmResponse],
      mock_model: MockModel,
      wait_for_realtime_input: bool = False,
  ):
    self.llm_responses = llm_responses
    self.mock_model = mock_model
    self.wait_for_realtime_input = wait_for_realtime_input
    self._input_event = asyncio.Event()

  async def send_history(self, history: list[types.Content]):
    pass

  async def send_content(self, content: types.Content):
    self.mock_model.live_contents.append(content)
    self._input_event.set()

  async def send(self, data):
    pass

  async def send_realtime(self, blob: types.Blob):
    self.mock_model.live_blobs.append(blob)
    self._input_event.set()

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    """Yield each of the pre-defined LlmResponses."""
    if self.wait_for_realtime_input:
      await self._input_event.wait()

    for response in self.llm_responses:
      yield response

  async def close(self):
    pass
