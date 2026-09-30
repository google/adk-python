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

"""Tests for the call handle and for reaching it from a tool's context.

`LiveKitRunner` registers the call under the session it drives, and
`current_call()` reads it back off the `ToolContext` ADK hands a tool. The
tools built on top of that live in `test_livekit_tools.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock

from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

pytest.importorskip("livekit.rtc")

from google.adk.integrations.livekit import _livekit_call
from google.adk.integrations.livekit import current_call

from tests.unittests.integrations.livekit.conftest import idle_runner
from tests.unittests.integrations.livekit.conftest import make_call
from tests.unittests.integrations.livekit.conftest import make_lk_runner
from tests.unittests.integrations.livekit.conftest import make_room
from tests.unittests.integrations.livekit.conftest import make_tool_context
from tests.unittests.integrations.livekit.conftest import patched_livekit_api
from tests.unittests.integrations.livekit.conftest import sip_participant
from tests.unittests.integrations.livekit.conftest import transfer_request
from tests.unittests.integrations.livekit.conftest import webrtc_participant
from tests.unittests.testing_utils import MockModel

# --- Reaching the call from a tool's context ---


def test_a_tool_outside_a_call_is_told_so():
  """An agent run without LiveKit must fail loudly, not silently no-op."""
  with pytest.raises(RuntimeError, match="No LiveKit call is in progress"):
    current_call(make_tool_context())


def test_a_tool_on_another_session_does_not_see_this_call():
  """The registry is keyed by session, so a lookup must not fall through.

  Two calls in one process is the normal case for a thread-executor worker.
  Answering with whichever call happens to be registered would hand one
  caller's room to another caller's agent.
  """
  with _livekit_call._register_call(make_call()):
    with pytest.raises(RuntimeError, match="No LiveKit call is in progress"):
      current_call(make_tool_context(session_id="someone-else"))


def test_two_calls_cannot_share_a_session():
  """Silently replacing the first call would leave it unreachable mid-call."""
  with _livekit_call._register_call(make_call()):
    with pytest.raises(RuntimeError, match="already in progress"):
      with _livekit_call._register_call(make_call()):
        pass  # pragma: no cover - the registration above raises


def test_concurrent_calls_each_reach_their_own_room():
  """One process can serve two calls, and neither may see the other's room."""
  first = make_call(session_id="call-one")
  second = make_call(session_id="call-two")

  with _livekit_call._register_call(first):
    with _livekit_call._register_call(second):
      assert current_call(make_tool_context(session_id="call-one")) is first
      assert current_call(make_tool_context(session_id="call-two")) is second


def test_the_call_does_not_leak_past_the_session():
  """A second call in the same process must not see the first one's room."""
  with _livekit_call._register_call(make_call()):
    pass

  assert _livekit_call._ACTIVE_CALLS == {}
  with pytest.raises(RuntimeError):
    current_call(make_tool_context())


async def test_a_failed_call_still_deregisters():
  """A setup failure must not strand an entry that blocks the next call."""
  room = make_room()
  room.local_participant.publish_track = AsyncMock(
      side_effect=RuntimeError("no track")
  )
  lk_runner = make_lk_runner(idle_runner(), room)

  with pytest.raises(RuntimeError, match="no track"):
    await lk_runner.start()

  assert _livekit_call._ACTIVE_CALLS == {}


async def test_real_tools_reach_the_call_during_a_live_session():
  """Real FunctionTools on a real agent can act on the room.

  Setup: an agent with one sync and one async tool, both of which resolve
    `current_call()`, driven through a real `Runner.run_live`.
  Act: the model answers the first user turn by calling both tools.
  Assert: each tool saw the session's own call handle.

  Sync and async tools are dispatched differently -- one onto a thread pool,
  one as a task -- so both are exercised.
  """
  seen: dict[str, str] = {}

  def read_call_from_sync_tool(tool_context: ToolContext) -> str:
    """Reads the call from a sync tool."""
    seen["sync"] = current_call(tool_context).session_id
    return "ok"

  async def read_call_from_async_tool(tool_context: ToolContext) -> str:
    """Reads the call from an async tool."""
    seen["async"] = current_call(tool_context).session_id
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


# --- Caller identity ---


def test_the_caller_number_is_readable_on_a_phone_call():
  """Tools look up customers by number, so it has to be reachable."""
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  call = make_call(make_room({"sip_caller": participant}))

  assert call.caller_phone_number == "+15105550100"


def test_a_browser_caller_has_no_phone_number():
  """WebRTC callers are not phone calls; nothing should be invented."""
  call = make_call(make_room({"browser": webrtc_participant()}))

  assert call.caller_phone_number is None


def test_sip_attributes_exclude_unrelated_participant_metadata():
  """Only LiveKit's telephony attributes describe the call."""
  participant = sip_participant(
      {"sip.phoneNumber": "+15105550100", "app.theme": "dark"}
  )
  call = make_call(make_room({"sip_caller": participant}))

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
  call = make_call(
      make_room({"sip_caller": participant}), hang_up_callback=ended.set
  )

  async with patched_livekit_api() as client:
    await call.hang_up()

  assert client.room.delete_room.await_args.args[0].room == "test-room"
  assert ended.is_set()


async def test_hanging_up_a_browser_call_only_leaves_the_room():
  """The browser tears its own side down, so no server call is needed.

  Deleting the room here would work too, but it would make every hangup
  depend on server API credentials that a WebRTC-only app has no other use
  for.
  """
  ended = asyncio.Event()
  call = make_call(
      make_room({"browser": webrtc_participant()}), hang_up_callback=ended.set
  )

  async with patched_livekit_api() as client:
    await call.hang_up()

  client.room.delete_room.assert_not_awaited()
  assert ended.is_set()


async def test_a_room_that_will_not_delete_still_ends_the_adk_side():
  """The model connection must not be stranded by a failed room delete."""
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  ended = asyncio.Event()
  call = make_call(
      make_room({"sip_caller": participant}), hang_up_callback=ended.set
  )

  async with patched_livekit_api() as client:
    client.room.delete_room = AsyncMock(side_effect=RuntimeError("no auth"))
    with pytest.raises(RuntimeError, match="no auth"):
      await call.hang_up()

  assert ended.is_set()


# --- Keypad output ---


async def test_pressing_keys_publishes_dtmf_tones():
  """Driving a downstream IVR means sending real tones, not speaking digits."""
  room = make_room()

  await make_call(room).send_dtmf("12#")

  sent = [
      (c.kwargs["code"], c.kwargs["digit"])
      for c in room.local_participant.publish_dtmf.await_args_list
  ]
  assert sent == [(1, "1"), (2, "2"), (11, "#")]


async def test_non_keypad_characters_are_skipped():
  """A model that hallucinates a letter must not break the call."""
  room = make_room()

  await make_call(room).send_dtmf("1z2")

  assert room.local_participant.publish_dtmf.await_count == 2


# --- Transfers ---


async def test_transferring_hands_the_caller_to_another_number():
  """Escalating to a human is the most-requested telephony behavior."""
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  call = make_call(make_room({"sip_caller": participant}))

  async with patched_livekit_api() as client:
    await call.transfer("tel:+15105550111")

  request = transfer_request(client)
  assert request.transfer_to == "tel:+15105550111"
  assert request.participant_identity == "sip_caller"


async def test_transferring_a_browser_call_is_refused():
  """There is no phone leg to hand over, so this cannot silently succeed."""
  call = make_call(make_room({"browser": webrtc_participant()}))

  with pytest.raises(RuntimeError, match="no SIP participant"):
    await call.transfer("tel:+15105550111")


# --- App-specific data ---


async def test_a_tool_can_push_data_to_clients():
  """In-game actions and robot commands ride the room's data track."""
  room = make_room()

  await make_call(room).send_data(b'{"action":"open_door"}', topic="game")

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

  reply = await make_call(room).perform_rpc(method="open_door", payload="north")

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
  call = make_call(room)

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
