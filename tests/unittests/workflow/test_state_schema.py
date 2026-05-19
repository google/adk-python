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

"""Tests for Workflow state_schema runtime enforcement."""

from __future__ import annotations

from typing import Optional

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.sessions.state import State
from google.adk.sessions.state import StateSchemaError
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from pydantic import BaseModel
import pytest

from .. import testing_utils
from .workflow_testing_utils import create_parent_invocation_context

# ── Schema models for testing ────────────────────────────────────────


class _PipelineSchema(BaseModel):
  counter: int
  name: str
  optional_field: Optional[str] = None


class _NodeSchema(BaseModel):
  x: int
  y: int


# ── Unit tests: State validation ─────────────────────────────────────


def test_state_rejects_unknown_key() -> None:
  """State with schema raises on unknown key."""
  state = State(value={}, delta={}, schema=_PipelineSchema)
  with pytest.raises(StateSchemaError, match='bad_key'):
    state['bad_key'] = 'value'


def test_state_accepts_declared_key() -> None:
  """State with schema accepts keys that exist in the schema."""
  state = State(value={}, delta={}, schema=_PipelineSchema)
  state['counter'] = 5
  state['name'] = 'hello'
  assert state['counter'] == 5
  assert state['name'] == 'hello'


def test_state_rejects_wrong_type() -> None:
  """State with schema raises when value type doesn't match annotation."""
  state = State(value={}, delta={}, schema=_PipelineSchema)
  with pytest.raises(StateSchemaError, match='counter'):
    state['counter'] = 'not_an_int'


def test_state_accepts_optional_none() -> None:
  """Optional fields accept None."""
  state = State(value={}, delta={}, schema=_PipelineSchema)
  state['optional_field'] = None
  assert state['optional_field'] is None


def test_state_allows_prefixed_keys() -> None:
  """Prefixed keys (app:, user:, temp:) bypass schema validation."""
  state = State(value={}, delta={}, schema=_PipelineSchema)
  state['app:anything'] = 'value'
  state['user:pref'] = 42
  state['temp:cache'] = [1, 2, 3]
  assert state['app:anything'] == 'value'


def test_state_update_validates_all_keys() -> None:
  """State.update validates each key-value pair."""
  state = State(value={}, delta={}, schema=_PipelineSchema)
  with pytest.raises(StateSchemaError, match='unknown'):
    state.update({'counter': 1, 'unknown': 'x'})


def test_state_no_schema_allows_all() -> None:
  """Without schema, any key/value is accepted (backward compat)."""
  state = State(value={}, delta={})
  state['anything'] = 'goes'
  state['whatever'] = 42
  assert state['anything'] == 'goes'


# ── Startup validation tests ─────────────────────────────────────────


def test_startup_rejects_mismatched_param() -> None:
  """FunctionNode param not in state_schema raises at construction."""

  def node_with_bad_param(ctx: Context, unknown_param: str) -> str:
    return 'done'

  with pytest.raises(StateSchemaError, match='unknown_param'):
    Workflow(
        name='wf',
        edges=[(START, node_with_bad_param)],
        state_schema=_PipelineSchema,
    )


def test_startup_accepts_matching_params() -> None:
  """FunctionNode params matching schema fields pass construction."""

  def node_with_good_params(ctx: Context, counter: int, name: str) -> str:
    return 'done'

  wf = Workflow(
      name='wf',
      edges=[(START, node_with_good_params)],
      state_schema=_PipelineSchema,
  )
  assert wf.state_schema is _PipelineSchema


def test_startup_skips_ctx_and_node_input() -> None:
  """Framework params (ctx, node_input) are not checked against schema."""

  def node_with_framework_params(ctx: Context, node_input: str) -> str:
    return 'done'

  Workflow(
      name='wf',
      edges=[(START, node_with_framework_params)],
      state_schema=_PipelineSchema,
  )


def test_startup_no_validation_when_schema_none() -> None:
  """No startup validation when state_schema is not set."""

  def node_with_any_param(ctx: Context, anything: str) -> str:
    return 'done'

  Workflow(
      name='wf',
      edges=[(START, node_with_any_param)],
  )


def test_workflow_state_schema_field_exists() -> None:
  """Workflow accepts a state_schema parameter."""

  def produce_done():
    return Event(output='done')

  wf = Workflow(
      name='wf',
      edges=[(START, produce_done)],
      state_schema=_PipelineSchema,
  )
  assert wf.state_schema is _PipelineSchema


def test_workflow_state_schema_defaults_to_none() -> None:
  """state_schema defaults to None when not provided."""

  def produce_done():
    return Event(output='done')

  wf = Workflow(
      name='wf',
      edges=[(START, produce_done)],
  )
  assert wf.state_schema is None


# ── Runtime enforcement tests (workflow execution) ───────────────────


@pytest.mark.asyncio
async def test_workflow_valid_state_writes_succeed(
    request: pytest.FixtureRequest,
) -> None:
  """A workflow with valid state writes runs without errors."""

  def write_state(ctx: Context) -> str:
    ctx.state['counter'] = 5
    ctx.state['name'] = 'hello'
    return 'done'

  wf = Workflow(
      name='wf',
      edges=[(START, write_state)],
      state_schema=_PipelineSchema,
  )
  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'done' for e in data_events)


@pytest.mark.asyncio
async def test_workflow_rejects_unknown_key_via_ctx_state(
    request: pytest.FixtureRequest,
) -> None:
  """ctx.state write with unknown key raises StateSchemaError."""

  def write_bad_key(ctx: Context) -> str:
    ctx.state['unknown_key'] = 'value'
    return 'done'

  wf = Workflow(
      name='wf',
      edges=[(START, write_bad_key)],
      state_schema=_PipelineSchema,
  )
  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(StateSchemaError, match='unknown_key'):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_workflow_rejects_unknown_key_via_event_state(
    request: pytest.FixtureRequest,
) -> None:
  """Event(state={...}) with unknown key raises StateSchemaError."""

  def emit_bad_state() -> Event:
    return Event(state={'arbitrary_key': 'value'}, output='done')

  wf = Workflow(
      name='wf',
      edges=[(START, emit_bad_state)],
      state_schema=_PipelineSchema,
  )
  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(StateSchemaError, match='arbitrary_key'):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_workflow_accepts_valid_event_state(
    request: pytest.FixtureRequest,
) -> None:
  """Event(state={...}) with valid keys succeeds."""

  def emit_valid_state() -> Event:
    return Event(state={'counter': 10, 'name': 'ok'}, output='done')

  wf = Workflow(
      name='wf',
      edges=[(START, emit_valid_state)],
      state_schema=_PipelineSchema,
  )
  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'done' for e in data_events)


@pytest.mark.asyncio
async def test_workflow_allows_prefixed_keys_at_runtime(
    request: pytest.FixtureRequest,
) -> None:
  """Prefixed keys bypass schema validation during workflow execution."""

  def write_prefixed(ctx: Context) -> str:
    ctx.state['temp:debug'] = True
    ctx.state['app:config'] = 'val'
    return 'done'

  wf = Workflow(
      name='wf',
      edges=[(START, write_prefixed)],
      state_schema=_PipelineSchema,
  )
  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'done' for e in data_events)


@pytest.mark.asyncio
async def test_workflow_without_schema_allows_anything(
    request: pytest.FixtureRequest,
) -> None:
  """When state_schema=None, any key/value is accepted (backward compat)."""

  def write_anything(ctx: Context) -> str:
    ctx.state['any_key'] = 'any_value'
    ctx.state['another'] = 42
    return 'done'

  wf = Workflow(
      name='wf',
      edges=[(START, write_anything)],
  )
  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'done' for e in data_events)


# ── Per-node state_schema tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_node_level_schema_validates_writes(
    request: pytest.FixtureRequest,
) -> None:
  """A FunctionNode with its own state_schema validates state writes."""

  def write_bad_key(ctx: Context) -> str:
    ctx.state['bad_key'] = 'value'
    return 'done'

  node = FunctionNode(
      name='guarded',
      func=write_bad_key,
      state_schema=_NodeSchema,
  )
  wf = Workflow(
      name='wf',
      edges=[(START, node)],
  )
  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(StateSchemaError, match='bad_key'):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_node_level_schema_accepts_valid_writes(
    request: pytest.FixtureRequest,
) -> None:
  """A FunctionNode with its own state_schema accepts declared keys."""

  def write_good_keys(ctx: Context) -> str:
    ctx.state['x'] = 1
    ctx.state['y'] = 2
    return 'done'

  node = FunctionNode(
      name='guarded',
      func=write_good_keys,
      state_schema=_NodeSchema,
  )
  wf = Workflow(
      name='wf',
      edges=[(START, node)],
  )
  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'done' for e in data_events)


@pytest.mark.asyncio
async def test_node_schema_overrides_workflow_schema(
    request: pytest.FixtureRequest,
) -> None:
  """Node-level state_schema takes precedence over workflow-level schema."""

  def write_node_key(ctx: Context) -> str:
    ctx.state['x'] = 10
    return 'done'

  node = FunctionNode(
      name='guarded',
      func=write_node_key,
      state_schema=_NodeSchema,
  )
  wf = Workflow(
      name='wf',
      edges=[(START, node)],
      state_schema=_PipelineSchema,
  )
  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  # 'x' is in _NodeSchema but NOT in _PipelineSchema — should succeed
  # because node schema overrides workflow schema
  events = await runner.run_async(testing_utils.get_user_content('start'))
  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'done' for e in data_events)


@pytest.mark.asyncio
async def test_node_without_schema_inherits_workflow_schema(
    request: pytest.FixtureRequest,
) -> None:
  """Node without state_schema inherits validation from parent workflow."""

  def write_bad_key(ctx: Context) -> str:
    ctx.state['unknown'] = 'value'
    return 'done'

  wf = Workflow(
      name='wf',
      edges=[(START, write_bad_key)],
      state_schema=_PipelineSchema,
  )
  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(StateSchemaError, match='unknown'):
    await runner.run_async(testing_utils.get_user_content('start'))
