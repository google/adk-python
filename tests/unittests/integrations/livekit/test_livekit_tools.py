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

"""Tests for the prebuilt tools a live agent uses to act on its call.

These wrap `LiveKitCall`, whose own behavior is covered in
`test_livekit_call.py`. What is tested here is the tool layer: resolving the
call from the `ToolContext`, and returning something the model can act on
rather than raising and killing the turn.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("livekit.rtc")

from google.adk.integrations.livekit import _livekit_call
from google.adk.integrations.livekit import end_call
from google.adk.integrations.livekit import send_dtmf
from google.adk.integrations.livekit import transfer_call

from tests.unittests.integrations.livekit.conftest import idle_runner
from tests.unittests.integrations.livekit.conftest import make_call
from tests.unittests.integrations.livekit.conftest import make_lk_runner
from tests.unittests.integrations.livekit.conftest import make_room
from tests.unittests.integrations.livekit.conftest import make_tool_context
from tests.unittests.integrations.livekit.conftest import patched_livekit_api
from tests.unittests.integrations.livekit.conftest import sip_participant
from tests.unittests.integrations.livekit.conftest import transfer_request
from tests.unittests.integrations.livekit.conftest import webrtc_participant


def _phone_room():
  return make_room(
      {"sip_caller": sip_participant({"sip.phoneNumber": "+15105550100"})}
  )


# --- Resolving the call ---


@pytest.mark.parametrize(
    "invoke",
    [
        pytest.param(lambda ctx: end_call(ctx), id="end_call"),
        pytest.param(
            lambda ctx: transfer_call("+15105550111", ctx), id="transfer_call"
        ),
        pytest.param(lambda ctx: send_dtmf("1", ctx), id="send_dtmf"),
    ],
)
async def test_every_tool_refuses_to_run_outside_a_call(invoke):
  """Under `adk web` there is no room, and no tool may pretend otherwise."""
  with pytest.raises(RuntimeError, match="No LiveKit call is in progress"):
    await invoke(make_tool_context())


# --- Ending the call ---


async def test_ending_the_call_hangs_up_and_says_so():
  """The model needs a result it can narrate, not a bare None."""
  ended = asyncio.Event()
  call = make_call(
      make_room({"browser": webrtc_participant()}), hang_up_callback=ended.set
  )

  async with patched_livekit_api():
    with _livekit_call._register_call(call):
      result = await end_call(make_tool_context())

  assert ended.is_set()
  assert "ending" in result


async def test_a_failed_hangup_is_reported_rather_than_raised():
  """A room that will not delete must not also kill the turn.

  The tool reports the failure so the model can say something, and the ADK
  side has already ended either way.
  """
  ended = asyncio.Event()
  call = make_call(_phone_room(), hang_up_callback=ended.set)

  async with patched_livekit_api() as client:
    client.room.delete_room = AsyncMock(side_effect=RuntimeError("no auth"))
    with _livekit_call._register_call(call):
      result = await end_call(make_tool_context())

  assert ended.is_set()
  assert "no auth" in result


async def test_a_sync_tool_can_end_the_call_from_its_worker_thread():
  """`end_call` reaches the runner even from off the event loop.

  Setup: a live session, and a sync tool body invoked the way ADK invokes one
    -- on a worker thread.
  Act: that body calls `end_call`.
  Assert: the session actually stops.

  ADK runs sync tools off the loop and `asyncio.Event` is not thread-safe, so
  this is the harder of the two dispatch paths. The call is resolved from the
  tool's own context rather than an inherited one, so unlike a `ContextVar`
  this does not depend on ADK copying the ambient context across the hand-off.
  """
  lk_runner = make_lk_runner(idle_runner(), make_room())

  session = asyncio.create_task(lk_runner.start())
  await asyncio.sleep(0)

  def _sync_tool_body():
    asyncio.run(end_call(make_tool_context()))

  await asyncio.get_running_loop().run_in_executor(None, _sync_tool_body)

  await asyncio.wait_for(session, timeout=5)


# --- Transferring ---


async def test_a_bare_number_is_dialled_as_a_phone_number():
  """A model asked for a transfer says `+1510...`, not `tel:+1510...`."""
  call = make_call(_phone_room())

  async with patched_livekit_api() as client:
    with _livekit_call._register_call(call):
      await transfer_call("+15105550111", make_tool_context())

  assert transfer_request(client).transfer_to == "tel:+15105550111"


async def test_a_sip_uri_destination_is_passed_through():
  """Not every transfer target is a phone number."""
  call = make_call(_phone_room())

  async with patched_livekit_api() as client:
    with _livekit_call._register_call(call):
      await transfer_call("sip:support@example.com", make_tool_context())

  assert transfer_request(client).transfer_to == "sip:support@example.com"


async def test_transferring_a_browser_call_explains_itself():
  """The model should hear why, so it can tell the user, not fail the turn."""
  call = make_call(make_room({"browser": webrtc_participant()}))

  with _livekit_call._register_call(call):
    result = await transfer_call("+15105550111", make_tool_context())

  assert "Could not transfer" in result


# --- Keypad ---


async def test_pressing_keys_sends_the_tones_and_confirms_them():
  """Driving a downstream IVR means sending real tones, not speaking digits."""
  room = make_room()

  with _livekit_call._register_call(make_call(room)):
    result = await send_dtmf("12#", make_tool_context())

  sent = [
      c.kwargs["digit"]
      for c in room.local_participant.publish_dtmf.await_args_list
  ]
  assert sent == ["1", "2", "#"]
  assert "12#" in result
