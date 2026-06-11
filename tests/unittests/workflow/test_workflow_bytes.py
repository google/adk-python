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

"""Tests for Workflow handling of bytes and serialization."""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import BaseNode
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest

# --- Helpers ---


class _BytesOutputNode(BaseNode):
  """Yields bytes content or raw bytes."""

  raw_bytes: bool = False

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    data = b"\x89PNG\r\n\x1a\n"
    if self.raw_bytes:
      yield data
    else:
      yield Event(
          content=types.Content(
              parts=[types.Part.from_bytes(data=data, mime_type="image/png")]
          ),
          output="bytes_sent",
      )


class _InputCapturingNode(BaseNode):
  """Captures node_input for later assertion."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  received_inputs: list[Any] = Field(default_factory=list)

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    self.received_inputs.append(node_input)
    yield {"received": node_input}


async def _run_workflow(wf, message="start"):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name="test", node=wf, session_service=ss)
  session = await ss.create_session(app_name="test", user_id="u")
  msg = types.Content(parts=[types.Part(text=message)], role="user")
  events = []
  async for event in runner.run_async(
      user_id="u", session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


# --- Tests ---


@pytest.mark.asyncio
async def test_bytes_in_content_output():
  """Content with bytes propagates to downstream node."""
  a = _BytesOutputNode(name="a", raw_bytes=False)
  b = _InputCapturingNode(name="b")
  wf = Workflow(name="wf", edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)

  assert b.received_inputs == ["bytes_sent"]


@pytest.mark.asyncio
async def test_raw_bytes_output():
  """Raw bytes output propagates to downstream node."""
  a = _BytesOutputNode(name="a", raw_bytes=True)
  b = _InputCapturingNode(name="b")
  wf = Workflow(name="wf", edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)

  assert len(b.received_inputs) == 1
  assert isinstance(b.received_inputs[0], bytes)


@pytest.mark.xfail(reason="Checkpoint/resume not yet in new Workflow.")
@pytest.mark.asyncio
async def test_bytes_in_node_input_serialization():
  """Bytes in node input survive checkpoint/resume."""
  assert False, "TODO"


@pytest.mark.xfail(reason="Checkpoint/resume not yet in new Workflow.")
@pytest.mark.asyncio
async def test_bytes_in_typed_model_input():
  """Bytes in Pydantic model input survive round-trip."""
  assert False, "TODO"


@pytest.mark.xfail(reason="Checkpoint/resume not yet in new Workflow.")
@pytest.mark.asyncio
async def test_bytes_in_trigger_buffer():
  """Bytes in trigger buffer survive serialization."""
  assert False, "TODO"


@pytest.mark.xfail(reason="Checkpoint/resume not yet in new Workflow.")
@pytest.mark.asyncio
async def test_bytes_full_workflow_resume():
  """Full resume with bytes data end-to-end."""
  assert False, "TODO"
