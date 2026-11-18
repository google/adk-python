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
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.workflow import BaseNode
from google.adk.workflow import node
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


from .._invariants import InvariantPlugin


async def _run_workflow(wf, message="start", *, check_invariants: bool = True):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name="test", node=wf, session_service=ss)
  if check_invariants:
    runner.plugin_manager.plugins.insert(0, InvariantPlugin())
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


def _produce_bytes() -> bytes:
  return b"\x00\x01"


def _produce_url_safe_bytes() -> bytes:
  return b"\xfb\xff\xfe"


async def _stream_bytes() -> AsyncGenerator[bytes, None]:
  yield b"\x00\x01"


def _produce_optional_bytes() -> bytes | None:
  return b"\x00\x01"


class _BytesSchemaNode(BaseNode):
  """Yields raw bytes and declares them through ``output_schema``."""

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield b"\x00\x01"


def _produce_str() -> str:
  return "test"


def _produce_non_ascii_text() -> bytes:
  return "\u00e9"  # type: ignore[return-value]  # A mislabeled producer.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("produce", "expected"),
    [
        (_produce_bytes, b"\x00\x01"),
        (_produce_url_safe_bytes, b"\xfb\xff\xfe"),
        (_stream_bytes, b"\x00\x01"),
        (_produce_optional_bytes, b"\x00\x01"),
        (
            _BytesSchemaNode(name="produce", output_schema=bytes),
            b"\x00\x01",
        ),
        (_produce_str, b"test"),
        (_produce_non_ascii_text, "\u00e9".encode()),
    ],
)
async def test_output_survives_persisted_session_resume(
    tmp_path, produce, expected
):
  """A node finished before a pause hands the resumed node its fresh value."""
  received = []

  def consume(node_input: bytes):
    received.append(node_input)

  wf = Workflow(
      name="wf",
      edges=[(START, produce, node(consume, require_confirmation=True))],
  )
  ss = SqliteSessionService(str(tmp_path / "sessions.db"))
  runner = Runner(app_name="test", node=wf, session_service=ss)
  session = await ss.create_session(app_name="test", user_id="u")

  async def send(part: types.Part) -> list[types.FunctionCall]:
    calls = []
    async for event in runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=types.Content(parts=[part], role="user"),
    ):
      calls.extend(event.get_function_calls())
    return calls

  calls = await send(types.Part(text="go"))
  await send(
      types.Part(
          function_response=types.FunctionResponse(
              id=calls[-1].id,
              name=calls[-1].name,
              response={"confirmed": True},
          )
      )
  )

  assert received == [expected]


@pytest.mark.asyncio
async def test_text_to_bytes_param_is_utf8_encoded():
  """Fresh text is UTF-8 encoded for a bytes parameter, even if it is base64."""
  received = []

  def consume(node_input: bytes):
    received.append(node_input)

  wf = Workflow(name="wf", edges=[(START, consume)])

  await _run_workflow(wf, message="test")

  assert received == [b"test"]
