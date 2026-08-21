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

"""Tests for reaching the LiveKit call from inside an ADK tool.

Covers the ambient call handle (`current_call()`) that `LiveKitRunner`
publishes, and the prebuilt tools that use it to hang up, transfer, and drive
a phone keypad.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.runners import Runner
from google.genai import types
import pytest

pytest.importorskip("livekit")

from google.adk.integrations.livekit import _call_context
from google.adk.integrations.livekit import current_call
from google.adk.integrations.livekit import end_call
from google.adk.integrations.livekit import LiveKitCall
from google.adk.integrations.livekit import LiveKitToolset
from google.adk.integrations.livekit import send_dtmf
from google.adk.integrations.livekit import transfer_call

from tests.unittests.integrations.livekit.conftest import make_lk_runner
from tests.unittests.integrations.livekit.conftest import make_room
from tests.unittests.integrations.livekit.conftest import sip_participant
from tests.unittests.integrations.livekit.conftest import webrtc_participant
from tests.unittests.testing_utils import MockModel

# --- Fixtures (minimal, one purpose each) ---


def _make_call(room=None, hang_up_callback=None) -> LiveKitCall:
  return LiveKitCall(
      room=room or make_room(),
      user_id="u1",
      session_id="s1",
      hang_up_callback=hang_up_callback or (lambda: None),
  )


def _idle_runner() -> Runner:
  """A Runner whose `run_live` never finishes, like a real idle call."""
  runner = MagicMock(spec=Runner)
  runner.app_name = "test_app"
  runner.session_service = MagicMock()
  runner.session_service.get_session = AsyncMock(return_value=MagicMock())

  async def run_live(**kwargs):
    await asyncio.Event().wait()
    yield  # pragma: no cover - unreachable, keeps this an async generator

  runner.run_live = run_live
  return runner


@contextlib.asynccontextmanager
async def _patched_livekit_api():
  """Patches LiveKit's server API and yields the client the code will use."""
  from livekit import api

  client = MagicMock()
  client.sip.transfer_sip_participant = AsyncMock()
  client.room.delete_room = AsyncMock()

  @contextlib.asynccontextmanager
  async def _session(*args, **kwargs):
    del args, kwargs
    yield client

  with patch.object(api, "LiveKitAPI", _session):
    yield client


def _transfer_request(client):
  return client.sip.transfer_sip_participant.await_args.args[0]


# --- Reaching the call from a tool ---


def test_a_tool_outside_a_call_is_told_so():
  """An agent run without LiveKit must fail loudly, not silently no-op."""
  with pytest.raises(RuntimeError, match="No LiveKit call is in progress"):
    current_call()


async def test_real_tools_reach_the_call_during_a_live_session():
  """Real FunctionTools on a real agent can act on the room.

  Setup: an agent with one sync and one async tool, both of which read
    `current_call()`, driven through a real `Runner.run_live`.
  Act: the model answers the first user turn by calling both tools.
  Assert: each tool saw the session's own call handle.

  Sync and async tools are dispatched differently -- one onto a thread pool
  through a copied context, one as a task -- so both are exercised.
  """
  seen: dict[str, str] = {}

  def read_call_from_sync_tool() -> str:
    """Reads the ambient call from a sync tool."""
    seen["sync"] = current_call().session_id
    return "ok"

  async def read_call_from_async_tool() -> str:
    """Reads the ambient call from an async tool."""
    seen["async"] = current_call().session_id
    return "ok"

  model = MockModel.create(
      responses=[
          types.Part.from_function_call(
              name="read_call_from_sync_tool", args={}
          ),
          types.Part.from_function_call(
              name="read_call_from_async_tool", args={}
          ),
          "done",
      ]
  )
  runner = InMemoryRunner(
      agent=LlmAgent(
          name="probe",
          model=model,
          tools=[read_call_from_sync_tool, read_call_from_async_tool],
      ),
      app_name="probe_app",
  )
  lk_runner = make_lk_runner(runner, make_room())

  session = asyncio.create_task(lk_runner.start())
  await asyncio.sleep(0)
  lk_runner._queue.send_content(
      types.Content(role="user", parts=[types.Part(text="go")])
  )
  try:
    for _ in range(100):
      if len(seen) == 2:
        break
      await asyncio.sleep(0.05)
  finally:
    session.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await session

  assert seen == {"sync": "s1", "async": "s1"}


async def test_the_call_does_not_leak_past_the_session():
  """A second call in the same process must not see the first one's room."""
  with _call_context._use_call(_make_call()):
    pass

  with pytest.raises(RuntimeError):
    current_call()


# --- Caller identity ---


def test_the_caller_number_is_readable_on_a_phone_call():
  """Tools look up customers by number, so it has to be reachable."""
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  call = _make_call(make_room({"sip_caller": participant}))

  assert call.caller_phone_number == "+15105550100"


def test_a_browser_caller_has_no_phone_number():
  """WebRTC callers are not phone calls; nothing should be invented."""
  call = _make_call(make_room({"browser": webrtc_participant()}))

  assert call.caller_phone_number is None


def test_sip_attributes_exclude_unrelated_participant_metadata():
  """Only LiveKit's telephony attributes describe the call."""
  participant = sip_participant(
      {"sip.phoneNumber": "+15105550100", "app.theme": "dark"}
  )
  call = _make_call(make_room({"sip_caller": participant}))

  assert call.sip_attributes() == {"sip.phoneNumber": "+15105550100"}


# --- Hanging up ---


async def test_hanging_up_a_phone_call_drops_the_phone_leg():
  """A SIP caller is held up by the SIP service, not by a client.

  Leaving the room is enough for a browser, which disconnects itself when the
  agent goes. Do the same to a phone caller and they are left on an open line
  listening to silence, so the room has to go.
  """
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  ended = asyncio.Event()
  call = _make_call(
      make_room({"sip_caller": participant}), hang_up_callback=ended.set
  )

  async with _patched_livekit_api() as client:
    with _call_context._use_call(call):
      result = await end_call()

  assert client.room.delete_room.await_args.args[0].room == "test-room"
  assert ended.is_set()
  assert "ending" in result


async def test_hanging_up_a_browser_call_only_leaves_the_room():
  """The browser tears its own side down, so no server call is needed.

  Deleting the room here would work too, but it would make every hangup
  depend on server API credentials that a WebRTC-only app has no other use
  for.
  """
  ended = asyncio.Event()
  call = _make_call(
      make_room({"browser": webrtc_participant()}), hang_up_callback=ended.set
  )

  async with _patched_livekit_api() as client:
    with _call_context._use_call(call):
      await end_call()

  client.room.delete_room.assert_not_awaited()
  assert ended.is_set()


async def test_a_failed_hangup_still_ends_the_session():
  """A room that will not delete must not also strand the model connection.

  The tool reports the failure instead of raising, so the model can say
  something rather than the turn dying.
  """
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  ended = asyncio.Event()
  call = _make_call(
      make_room({"sip_caller": participant}), hang_up_callback=ended.set
  )

  async with _patched_livekit_api() as client:
    client.room.delete_room = AsyncMock(side_effect=RuntimeError("no auth"))
    with _call_context._use_call(call):
      result = await end_call()

  assert ended.is_set()
  assert "no auth" in result


async def test_a_sync_tool_can_end_the_call_from_its_worker_thread():
  """`end_call` reaches the runner even from off the event loop.

  Setup: a live session, and a sync tool body invoked the way ADK invokes one
    -- through a copied context on a worker thread.
  Act: that body calls `end_call`.
  Assert: the session actually stops.

  ADK runs sync tools off the loop and `asyncio.Event` is not thread-safe, so
  this is the harder of the two dispatch paths.
  """
  lk_runner = make_lk_runner(_idle_runner(), make_room())

  session = asyncio.create_task(lk_runner.start())
  await asyncio.sleep(0)

  def _sync_tool_body():
    asyncio.run(end_call())

  with _call_context._use_call(lk_runner.call):
    context = contextvars.copy_context()
    await asyncio.get_running_loop().run_in_executor(
        None, lambda: context.run(_sync_tool_body)
    )

  await asyncio.wait_for(session, timeout=5)


# --- Keypad output ---


async def test_pressing_keys_publishes_dtmf_tones():
  """Driving a downstream IVR means sending real tones, not speaking digits."""
  room = make_room()

  with _call_context._use_call(_make_call(room)):
    await send_dtmf("12#")

  sent = [
      (c.kwargs["code"], c.kwargs["digit"])
      for c in room.local_participant.publish_dtmf.await_args_list
  ]
  assert sent == [(1, "1"), (2, "2"), (11, "#")]


async def test_non_keypad_characters_are_skipped():
  """A model that hallucinates a letter must not break the call."""
  room = make_room()

  with _call_context._use_call(_make_call(room)):
    await send_dtmf("1z2")

  assert room.local_participant.publish_dtmf.await_count == 2


# --- Transfers ---


async def test_transferring_hands_the_caller_to_another_number():
  """Escalating to a human is the most-requested telephony behavior."""
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  call = _make_call(make_room({"sip_caller": participant}))

  async with _patched_livekit_api() as client:
    with _call_context._use_call(call):
      await transfer_call("+15105550111")

  request = _transfer_request(client)
  assert request.transfer_to == "tel:+15105550111"
  assert request.participant_identity == "sip_caller"


async def test_transferring_a_browser_call_explains_itself():
  """The model should hear why, so it can tell the user, not fail the turn."""
  call = _make_call(make_room({"browser": webrtc_participant()}))

  with _call_context._use_call(call):
    result = await transfer_call("+15105550111")

  assert "Could not transfer" in result


async def test_a_sip_uri_destination_is_passed_through():
  """Not every transfer target is a phone number."""
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  call = _make_call(make_room({"sip_caller": participant}))

  async with _patched_livekit_api() as client:
    with _call_context._use_call(call):
      await transfer_call("sip:support@example.com")

  assert _transfer_request(client).transfer_to == "sip:support@example.com"


# --- App-specific data ---


async def test_a_tool_can_push_data_to_clients():
  """In-game actions and robot commands ride the room's data track."""
  room = make_room()

  await _make_call(room).send_data(b'{"action":"open_door"}', topic="game")

  (payload,) = room.local_participant.publish_data.await_args.args
  assert payload == b'{"action":"open_door"}'
  assert (
      room.local_participant.publish_data.await_args.kwargs["topic"] == "game"
  )


async def test_a_tool_can_call_the_client_and_read_its_reply():
  """RPC is the round trip `send_data` cannot do.

  A tool has to be able to reach the client and use its answer, which is what
  forwarding an LLM function call to a game or app client depends on.
  """
  room = make_room({"player": webrtc_participant("player")})
  room.local_participant.perform_rpc = AsyncMock(return_value="door opened")

  reply = await _make_call(room).perform_rpc(
      method="open_door", payload="north"
  )

  assert reply == "door opened"
  kwargs = room.local_participant.perform_rpc.await_args.kwargs
  assert kwargs["destination_identity"] == "player"
  assert kwargs["method"] == "open_door"
  assert kwargs["payload"] == "north"


async def test_rpc_refuses_to_guess_between_two_participants():
  """Picking a destination silently would send game actions to a bystander.

  An explicit destination is honored in the same room, so the refusal is
  about the guess rather than about multi-party rooms.
  """
  room = make_room({
      "player": webrtc_participant("player"),
      "spectator": webrtc_participant("spectator"),
  })
  room.local_participant.perform_rpc = AsyncMock(return_value="ok")
  call = _make_call(room)

  with pytest.raises(RuntimeError, match="2 remote participants"):
    await call.perform_rpc(method="open_door", payload="north")

  await call.perform_rpc(
      method="open_door", payload="north", destination_identity="player"
  )
  assert (
      room.local_participant.perform_rpc.await_args.kwargs[
          "destination_identity"
      ]
      == "player"
  )


# --- The call toolset ---


async def _toolset_names(toolset, call=None):
  """Resolves the toolset with `call` published, as ADK would at runtime."""
  if call is None:
    return [tool.name for tool in await toolset.get_tools()]
  with _call_context._use_call(call):
    return [tool.name for tool in await toolset.get_tools()]


async def test_the_toolset_offers_nothing_without_a_call():
  """Under `adk web` the agent must not be offered a hangup it cannot do."""
  assert await _toolset_names(LiveKitToolset()) == []


async def test_the_toolset_offers_only_hangup_on_a_webrtc_call():
  """A browser caller cannot be transferred or sent tones."""
  call = _make_call(make_room({"browser": webrtc_participant()}))

  assert await _toolset_names(LiveKitToolset(), call) == ["end_call"]


async def test_the_toolset_adds_telephony_tools_on_a_phone_call():
  """Only a SIP peer can be transferred or sent tones."""
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  call = _make_call(make_room({"sip_caller": participant}))

  assert await _toolset_names(LiveKitToolset(), call) == [
      "end_call",
      "transfer_call",
      "send_dtmf",
  ]


async def test_a_tool_filter_withholds_a_tool():
  """Some agents must never decide the conversation is over."""
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  call = _make_call(make_room({"sip_caller": participant}))
  toolset = LiveKitToolset(tool_filter=["transfer_call", "send_dtmf"])

  assert await _toolset_names(toolset, call) == ["transfer_call", "send_dtmf"]


async def test_the_toolset_goes_in_an_agents_tools_list():
  """The point of the toolset: no model_copy, no rebuilt Runner."""
  agent = LlmAgent(
      name="support_agent",
      model="gemini-live-2.5-flash",
      tools=[LiveKitToolset()],
  )
  call = _make_call(make_room({"browser": webrtc_participant()}))

  with _call_context._use_call(call):
    tools = await agent.canonical_tools()

  assert [tool.name for tool in tools] == ["end_call"]
