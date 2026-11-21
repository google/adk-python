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

"""Pause and resume on every session backend registered in _conformance.

Each turn builds a new App and Runner over the same session service, so a
resumed turn knows only what the backend stored, as after a process restart.
"""

from __future__ import annotations

from collections import Counter
import itertools
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.utils._function_call_names import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

from .. import testing_utils
from ..workflow.workflow_testing_utils import create_runner
from ..workflow.workflow_testing_utils import RequestInputNode
from ._conformance import session_service  # noqa: F401

_APP_NAME = 'resume_app'
_USER_ID = 'user'


async def _run_turn(
    session_service: BaseSessionService,
    app: App,
    session_id: str,
    message: types.Content,
    invocation_id: str | None = None,
) -> list[Event]:
  async with create_runner(app=app, session_service=session_service) as runner:
    return [
        event
        async for event in runner.run_async(
            user_id=_USER_ID,
            session_id=session_id,
            invocation_id=invocation_id,
            new_message=message,
        )
    ]


def _answer(
    events: list[Event], call_name: str, response: dict[str, Any]
) -> types.Content:
  calls = itertools.chain.from_iterable(
      event.get_function_calls() for event in events
  )
  [call_id] = [call.id for call in calls if call.name == call_name]
  return testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=call_id, name=call_name, response=response
          )
      )
  )


def _agent_app(tool: BaseTool, *responses: types.Part | str) -> App:
  return App(
      name=_APP_NAME,
      root_agent=LlmAgent(
          name='root_agent',
          model=testing_utils.MockModel.create(responses=list(responses)),
          tools=[tool],
      ),
      resumability_config=ResumabilityConfig(is_resumable=True),
  )


def _workflow_app(
    runs: list[Any], *questions: str, prepare_output: str | None = 'prepared'
) -> App:
  """Builds START, prepare, one input request per question, then finish."""

  def prepare() -> str | None:
    runs.append('prepare')
    return prepare_output

  def finish(node_input: Any) -> None:
    runs.append(('finish', node_input))

  asks = [
      RequestInputNode(name=f'ask_{index}', message=question)
      for index, question in enumerate(questions)
  ]
  return App(
      name=_APP_NAME,
      root_agent=Workflow(
          name='approval', edges=[(START, prepare, *asks, finish)]
      ),
      resumability_config=ResumabilityConfig(is_resumable=True),
  )


async def test_confirmed_tool_runs_once_after_reload(
    session_service: BaseSessionService,
):
  """An approval sent to a new Runner runs the paused tool exactly once."""
  deleted = []

  def delete_file(path: str) -> dict[str, str]:
    deleted.append(path)
    return {'deleted': path}

  session = await session_service.create_session(
      app_name=_APP_NAME, user_id=_USER_ID
  )

  paused = await _run_turn(
      session_service,
      _agent_app(
          FunctionTool(delete_file, require_confirmation=True),
          types.Part.from_function_call(
              name='delete_file', args={'path': 'a.txt'}
          ),
      ),
      session.id,
      testing_utils.UserContent('delete a.txt'),
  )
  resumed = await _run_turn(
      session_service,
      _agent_app(
          FunctionTool(delete_file, require_confirmation=True), 'a.txt deleted'
      ),
      session.id,
      _answer(
          paused, REQUEST_CONFIRMATION_FUNCTION_CALL_NAME, {'confirmed': True}
      ),
      invocation_id=paused[0].invocation_id,
  )

  assert deleted == ['a.txt']
  assert testing_utils.simplify_events(resumed) == [
      (
          'root_agent',
          types.Part.from_function_response(
              name='delete_file', response={'deleted': 'a.txt'}
          ),
      ),
      ('root_agent', 'a.txt deleted'),
  ]


async def test_long_running_tool_result_reaches_model_after_reload(
    session_service: BaseSessionService,
):
  """A final result sent to a new Runner for a pending call reaches the model."""
  started = []

  def start_export(table: str) -> dict[str, str]:
    started.append(table)
    return {'status': 'pending'}

  session = await session_service.create_session(
      app_name=_APP_NAME, user_id=_USER_ID
  )
  resumed_app = _agent_app(LongRunningFunctionTool(start_export), 'exported')

  pending = await _run_turn(
      session_service,
      _agent_app(
          LongRunningFunctionTool(start_export),
          types.Part.from_function_call(
              name='start_export', args={'table': 'orders'}
          ),
          'export started',
      ),
      session.id,
      testing_utils.UserContent('export orders'),
  )
  resumed = await _run_turn(
      session_service,
      resumed_app,
      session.id,
      _answer(pending, 'start_export', {'status': 'done'}),
      invocation_id=pending[0].invocation_id,
  )

  assert started == ['orders']
  assert testing_utils.simplify_events(resumed) == [('root_agent', 'exported')]
  [request] = resumed_app.root_agent.model.requests
  assert request.contents[-1].parts[0].function_response.response == {
      'status': 'done'
  }


@pytest.mark.parametrize(
    'prepare_output',
    [
        'prepared',
        pytest.param(
            None,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    'A node that finishes without output, route or state'
                    ' change leaves no event, so a resume runs it again.'
                ),
            ),
        ),
    ],
)
async def test_workflow_resumes_at_paused_node_after_reload(
    session_service: BaseSessionService, prepare_output: str | None
):
  """A new Runner resumes at the paused node and does not rerun earlier ones."""
  runs = []
  session = await session_service.create_session(
      app_name=_APP_NAME, user_id=_USER_ID
  )

  paused = await _run_turn(
      session_service,
      _workflow_app(runs, 'Approve?', prepare_output=prepare_output),
      session.id,
      testing_utils.UserContent('start'),
  )
  await _run_turn(
      session_service,
      _workflow_app(runs, 'Approve?', prepare_output=prepare_output),
      session.id,
      _answer(paused, REQUEST_INPUT_FUNCTION_CALL_NAME, {'approved': True}),
      invocation_id=paused[0].invocation_id,
  )

  assert runs == ['prepare', ('finish', {'approved': True})]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        'Each resume stores the output of every node that finished before it'
        ' again, so the session grows with every resume of an invocation.'
    ),
)
async def test_workflow_resumed_twice_stores_each_output_once(
    session_service: BaseSessionService,
):
  """A second resume does not store earlier nodes' outputs again."""
  runs = []
  session = await session_service.create_session(
      app_name=_APP_NAME, user_id=_USER_ID
  )

  first_pause = await _run_turn(
      session_service,
      _workflow_app(runs, 'Approve?', 'Really?'),
      session.id,
      testing_utils.UserContent('start'),
  )
  invocation_id = first_pause[0].invocation_id
  second_pause = await _run_turn(
      session_service,
      _workflow_app(runs, 'Approve?', 'Really?'),
      session.id,
      _answer(
          first_pause, REQUEST_INPUT_FUNCTION_CALL_NAME, {'approved': True}
      ),
      invocation_id=invocation_id,
  )
  await _run_turn(
      session_service,
      _workflow_app(runs, 'Approve?', 'Really?'),
      session.id,
      _answer(second_pause, REQUEST_INPUT_FUNCTION_CALL_NAME, {'sure': True}),
      invocation_id=invocation_id,
  )

  session = await session_service.get_session(
      app_name=_APP_NAME, user_id=_USER_ID, session_id=session.id
  )
  assert runs == ['prepare', ('finish', {'sure': True})]
  assert Counter(
      event.node_info.name
      for event in session.events
      if event.output is not None
  ) == {'prepare': 1, 'ask_0': 1, 'ask_1': 1}
