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

"""Unit tests for _runner_utils helper module."""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.live import _runner_utils
from google.adk.live import LiveRequestQueue
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.genai import types
import pytest


class _MockLiveAgent(BaseAgent):

  def __init__(self, name: str = "mock_agent"):
    super().__init__(name=name)

  async def _run_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(author=self.name)

  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text="live response")]
        ),
    )


@pytest.mark.asyncio
async def test_new_invocation_context_for_live_subagents_audio_transcription():
  parent_agent = _MockLiveAgent(name="parent")
  sub_agent = _MockLiveAgent(name="child")
  parent_agent.sub_agents = [sub_agent]

  runner = Runner(
      app_name="test_app",
      agent=parent_agent,
      session_service=InMemorySessionService(),
  )
  session = await runner.session_service.create_session(
      user_id="u1", session_id="s1", app_name=runner.app_name
  )
  queue = LiveRequestQueue()
  run_config = RunConfig(response_modalities=[types.Modality.AUDIO])

  ic = _runner_utils.new_invocation_context_for_live(
      runner,
      session,
      live_request_queue=queue,
      run_config=run_config,
  )

  assert ic.live_request_queue is queue
  assert ic.run_config.output_audio_transcription is not None
  assert ic.run_config.input_audio_transcription is not None


@pytest.mark.asyncio
async def test_run_live_validates_required_arguments():
  agent = _MockLiveAgent()
  runner = Runner(
      app_name="test_app", agent=agent, session_service=InMemorySessionService()
  )
  queue = LiveRequestQueue()

  with pytest.raises(
      ValueError,
      match="Either session or user_id and session_id must be provided.",
  ):
    async for _ in _runner_utils.run_live(
        runner,
        live_request_queue=queue,
    ):
      pass

  with pytest.raises(
      ValueError, match="live_request_queue is required for run_live."
  ):
    async for _ in _runner_utils.run_live(
        runner,
        user_id="u1",
        session_id="s1",
        live_request_queue=None,  # pytype: disable=wrong-arg-types
    ):
      pass


@pytest.mark.asyncio
async def test_run_live_yields_events_and_delegates_to_agent():
  agent = _MockLiveAgent(name="live_agent")
  runner = Runner(
      app_name="test_app", agent=agent, session_service=InMemorySessionService()
  )
  await runner.session_service.create_session(
      user_id="u1", session_id="s1", app_name=runner.app_name
  )
  queue = LiveRequestQueue()

  events = [
      event
      async for event in runner.run_live(
          user_id="u1",
          session_id="s1",
          live_request_queue=queue,
      )
  ]

  assert len(events) == 1
  assert events[0].author == "live_agent"
  assert events[0].content.parts[0].text == "live response"


class _BranchRecordingAgent(BaseAgent):
  """Records the branch of the context it is run under."""

  seen_branch: Any = None

  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    type(self).seen_branch = ctx.branch
    yield Event(author=self.name)


@pytest.mark.asyncio
async def test_run_live_restores_the_branch_of_a_resumed_sub_agent():
  """A live run resumed on a sub-agent continues on that agent's branch.

  `run_live` resolves the agent to run from history and must then recover the
  branch that agent last ran on; without it the sub-agent silently continues on
  the root branch. Guarded here because this body moved into `_runner_utils`,
  where the step is easy to drop.
  """
  child = _BranchRecordingAgent(name="child")
  _BranchRecordingAgent.seen_branch = None
  parent = _MockLiveAgent(name="parent")
  parent.sub_agents = [child]

  runner = Runner(
      app_name="test_app",
      agent=parent,
      session_service=InMemorySessionService(),
  )
  session = await runner.session_service.create_session(
      user_id="u1", session_id="s1", app_name=runner.app_name
  )
  # The sub-agent's last turn, recorded on its own sub-branch.
  await runner.session_service.append_event(
      session,
      Event(
          invocation_id="inv_prev",
          author="child",
          branch="parent.child",
          content=types.Content(role="model", parts=[types.Part(text="hi")]),
      ),
  )

  # run_live resolves the agent from history; pin it so the test is about the
  # branch recovery that follows, not about resolution.
  runner._find_agent_to_run = lambda _session, _root: child

  async for _ in runner.run_live(
      user_id="u1", session_id="s1", live_request_queue=LiveRequestQueue()
  ):
    break

  assert _BranchRecordingAgent.seen_branch == "parent.child"


class _FailingNode(BaseNode):
  """A non-agent root node whose run raises."""

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    raise RuntimeError("root node exploded")
    yield  # pylint: disable=unreachable


class _RecordingPlugin(BasePlugin):

  def __init__(self):
    super().__init__(name="recording")
    self.errors: list[Exception] = []

  async def on_run_error_callback(
      self, *, invocation_context: InvocationContext, error: Exception
  ) -> None:
    self.errors.append(error)


@pytest.mark.asyncio
async def test_run_node_live_notifies_plugins_when_the_root_node_fails():
  """A root-node failure reaches on_run_error_callback, then propagates.

  The failure surfaces from `_cleanup_root_task`, which re-raises it after the
  event queue has drained normally. That only reaches the plugins while the
  cleanup runs inside the region the notifying `except` covers, so this pins
  the nesting rather than just the presence of the handler.
  """
  plugin = _RecordingPlugin()
  runner = Runner(
      app=App(
          name="test_app",
          root_agent=_FailingNode(name="root_node"),
          plugins=[plugin],
      ),
      session_service=InMemorySessionService(),
  )
  session = await runner.session_service.create_session(
      user_id="u1", session_id="s1", app_name=runner.app_name
  )

  with pytest.raises(RuntimeError, match="root node exploded"):
    async for _ in _runner_utils.run_node_live(
        runner,
        session=session,
        live_request_queue=LiveRequestQueue(),
    ):
      pass

  assert [str(e) for e in plugin.errors] == ["root node exploded"]
