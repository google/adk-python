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

"""Testing utils for the Workflow."""

import copy
import inspect
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import List
from typing import Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.invocation_context import InvocationContext as BaseInvocationContext
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import BaseNode
from google.adk.workflow._graph import RouteValue
from google.adk.workflow.utils._workflow_hitl_utils import has_auth_request_function_call
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from .testing_utils import END_OF_AGENT
from .testing_utils import simplify_content


async def run_workflow(wf, message='start'):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name=wf.name, node=wf, session_service=ss)
  session = await ss.create_session(app_name=wf.name, user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


# Emulates a node that outputs an Event & a route.
# If output is not None, the output is set as the data field in the event.
# If route is not None, the route is set in the node output event.
# The route can be set without the output. This means didn't produce any output
# but wants to signal a route to take.
class TestingNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  output: Optional[Any] = None
  route: (
      RouteValue | list[RouteValue] | Callable[[Context, Any], Any] | None
  ) = None
  received_inputs: List[Any] = Field(default_factory=list)

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    if self.output is not None or self.route is not None:
      route = None
      if callable(self.route):
        if inspect.iscoroutinefunction(self.route):
          route = await self.route(ctx, node_input)
        else:
          route = self.route(ctx, node_input)
      else:
        route = self.route

      self.received_inputs.append(node_input)
      yield Event(
          output=self.output,
          route=route,
      )


class TestingNodeWithIntermediateContent(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  intermediate_content: list[types.Content] = Field(default_factory=list)
  output: Optional[Any] = None
  route: Optional[str] = None

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    for content in self.intermediate_content:
      yield Event(
          author=self.name,
          invocation_id=ctx.invocation_id,
          content=content,
      )

    if self.output is not None:
      yield Event(
          output=self.output,
          route=self.route,
      )


class InputCapturingNode(BaseNode):
  """A node that captures the inputs it receives."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  received_inputs: List[Any] = Field(default_factory=list)

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    self.received_inputs.append(node_input)
    yield Event(
        output={'received': node_input},
    )


class RequestInputNode(BaseNode):
  """A simple node that requests input from the user."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  message: str = Field(default='')
  response_schema: Optional[dict[str, Any]] = None

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    yield RequestInput(
        message=self.message,
        response_schema=self.response_schema,
    )


async def create_parent_invocation_context(
    test_name: str, agent: BaseAgent, resumable: bool = False
) -> InvocationContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id=f'{test_name}_invocation_id',
      agent=agent,
      session=session,
      session_service=session_service,
      resumability_config=ResumabilityConfig(is_resumable=resumable),
  )


def simplify_event_with_node(
    event: Event,
    include_state_delta: bool = False,
) -> Any | None:
  if isinstance(event, Event):
    if (
        'output' not in event.model_fields_set
        and not (include_state_delta and event.actions.state_delta)
        and not event.content
    ):
      return None

    # If the event has content, return the simplified content.
    if event.content:
      return simplify_content(event.content)

    simplified_event = {}

    # Also simplify event.output if it contains Content.
    # The tests assume that Content found in event data should be simplified
    # (IDs stripped) just like event.content. This ensures consistent
    # assertion behavior.
    output = event.output
    if isinstance(output, types.Content):
      output = copy.deepcopy(output)
      for part in output.parts:
        if part.function_call and part.function_call.id:
          part.function_call.id = None
        if part.function_response and part.function_response.id:
          part.function_response.id = None
    simplified_event['output'] = output

    if include_state_delta and event.actions.state_delta:
      simplified_event['state_delta'] = event.actions.state_delta

    return simplified_event
  elif event.content:
    return simplify_content(event.content)


def simplify_events_with_node(
    events: list[Event],
    *,
    include_state_delta: bool = False,
    include_workflow_output: bool = False,
) -> list[tuple[str, Any]]:
  results = []

  # Second pass: Simplify events
  for event in events:
    # Optionally skip top-level workflow output events (events emitted
    # by the Workflow in _finalize_workflow). These events have a
    # top-level node_path (no '/' separator) and carry output data.
    if (
        not include_workflow_output
        and isinstance(event, Event)
        and event.output is not None
        and '/' not in (event.node_info.path or '')
    ):
      continue

    simplified_event = simplify_event_with_node(event, include_state_delta)
    if simplified_event:
      # Map the author to the source node name if it exists.
      if hasattr(event, 'node_info') and event.node_info.path:
        author = event.node_info.path
      else:
        author = event.author
      results.append((author, simplified_event))
  return results


def simplify_events_with_node_and_agent_state(
    events: list[Event],
    *,
    include_state_delta: bool = False,
    include_inputs_and_triggers: bool = False,
    include_resume_inputs: bool = False,
    include_workflow_output: bool = False,
):
  fields_to_exclude = {'run_id'}
  if not include_inputs_and_triggers:
    fields_to_exclude.add('input')
  if not include_resume_inputs:
    fields_to_exclude.add('resume_inputs')

  results = []

  for event in events:
    # Optionally skip top-level workflow output events.
    if (
        not include_workflow_output
        and isinstance(event, Event)
        and event.output is not None
        and '/' not in (event.node_info.path or '')
    ):
      continue
    simplified_event = simplify_event_with_node(event, include_state_delta)

    # Map the author to the source node name if it exists.
    if hasattr(event, 'node_info') and event.node_info.path:
      author = event.node_info.path
    else:
      author = event.author

    if simplified_event:
      results.append((author, simplified_event))
    elif event.actions.end_of_agent:
      results.append((author, END_OF_AGENT))
    elif event.actions.agent_state is not None:
      agent_state = event.actions.agent_state
      nodes = agent_state.get('nodes', {})
      simplified_nodes = {}
      for node_name, node_state in nodes.items():
        simplified_nodes[node_name] = {
            k: v
            for k, v in node_state.items()
            if k not in fields_to_exclude
            and (k != 'interrupts' or v)  # Exclude empty interrupts
            and (k != 'resume_inputs' or v)  # Exclude empty resume_inputs
        }
      results.append((author, {'nodes': simplified_nodes}))
  return results


def get_request_input_events(events: list[Any]) -> list[Any]:
  """Returns a list of request input events from the given list of events."""
  return [e for e in events if has_request_input_function_call(e)]


def get_auth_request_events(events: list[Any]) -> list[Any]:
  """Returns a list of auth credential request events from the given list."""
  return [e for e in events if has_auth_request_function_call(e)]


def get_output_events(events: list[Any], output: Any = None) -> list[Any]:
  """Returns a list of events that have output populated."""
  return [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and (output is None or e.output == output)
  ]


def get_outputs(events: list[Event]) -> list[Any]:
  """Extracts output values from events, skipping non-output events."""
  return [
      e.output for e in events if isinstance(e, Event) and e.output is not None
  ]


def strip_checkpoint_events(
    simplified_events: list[tuple[str, Any]],
) -> list[tuple[str, Any]]:
  """Strips agent_state checkpoint and end_of_agent events.

  In non-resumable mode, the workflow does not emit checkpoint events
  or end_of_agent events.  Use this to derive the expected simplified
  output for non-resumable tests from the resumable expected output.
  """
  return [
      (author, data)
      for author, data in simplified_events
      if not (isinstance(data, dict) and 'nodes' in data)
      and data != END_OF_AGENT
  ]


def find_function_call_event(
    events: list[Any], name: str | None = None
) -> Any | None:
  """Finds the first event containing a function call."""
  for e in events:
    if hasattr(e, 'content') and e.content and e.content.parts:
      for part in e.content.parts:
        if part.function_call:
          if name is None or part.function_call.name == name:
            return e
  return None


class CustomRetryableError(Exception):
  """A custom error meant to be retried."""


class CustomNonRetryableError(Exception):
  """A custom error not meant to be retried."""


class _FlakyNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  message: str = Field(default='')
  succeed_on_iteration: int = Field(default=0)
  tracker: dict[str, Any] = Field(default_factory=dict)
  exception_to_raise: Exception = Field(...)

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    iteration_count = self.tracker.get('iteration_count', 0) + 1
    self.tracker['iteration_count'] = iteration_count
    self.tracker.setdefault('attempt_counts', []).append(ctx.attempt_count)

    if iteration_count < self.succeed_on_iteration:
      raise self.exception_to_raise

    yield Event(
        output=self.message,
    )
