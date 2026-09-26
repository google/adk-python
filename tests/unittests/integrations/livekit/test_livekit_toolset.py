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

"""Tests for the toolset that offers the tools the current call can honor."""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent
import pytest

pytest.importorskip("livekit.rtc")

from google.adk.integrations.livekit import _livekit_call
from google.adk.integrations.livekit import LiveKitToolset

from tests.unittests.integrations.livekit.conftest import make_call
from tests.unittests.integrations.livekit.conftest import make_room
from tests.unittests.integrations.livekit.conftest import make_tool_context
from tests.unittests.integrations.livekit.conftest import sip_participant
from tests.unittests.integrations.livekit.conftest import webrtc_participant


async def _toolset_names(toolset, call=None):
  """Resolves the toolset the way ADK would, with `call` registered or not."""
  context = make_tool_context()
  if call is None:
    return [tool.name for tool in await toolset.get_tools(context)]
  with _livekit_call._register_call(call):
    return [tool.name for tool in await toolset.get_tools(context)]


async def test_the_toolset_offers_nothing_without_a_call():
  """Under `adk web` the agent must not be offered a hangup it cannot do."""
  assert await _toolset_names(LiveKitToolset()) == []


async def test_the_toolset_offers_nothing_outside_an_invocation():
  """Config loading resolves tools with no context, and must not blow up."""
  assert await LiveKitToolset().get_tools() == []


async def test_the_toolset_offers_only_hangup_on_a_webrtc_call():
  """A browser caller cannot be transferred or sent tones."""
  call = make_call(make_room({"browser": webrtc_participant()}))

  assert await _toolset_names(LiveKitToolset(), call) == ["end_call"]


async def test_the_toolset_adds_telephony_tools_on_a_phone_call():
  """Only a SIP peer can be transferred or sent tones."""
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  call = make_call(make_room({"sip_caller": participant}))

  assert await _toolset_names(LiveKitToolset(), call) == [
      "end_call",
      "transfer_call",
      "send_dtmf",
  ]


async def test_a_toolset_serves_only_the_call_on_its_own_session():
  """One agent definition, two concurrent calls, no crossed wires."""
  toolset = LiveKitToolset()
  sip_call = make_call(
      make_room(
          {"sip_caller": sip_participant({"sip.phoneNumber": "+15105550100"})}
      ),
      session_id="phone",
  )
  web_call = make_call(
      make_room({"browser": webrtc_participant()}), session_id="browser"
  )

  with _livekit_call._register_call(sip_call):
    with _livekit_call._register_call(web_call):
      phone_tools = await toolset.get_tools(make_tool_context("u1", "phone"))
      web_tools = await toolset.get_tools(make_tool_context("u1", "browser"))

  assert [tool.name for tool in phone_tools] == [
      "end_call",
      "transfer_call",
      "send_dtmf",
  ]
  assert [tool.name for tool in web_tools] == ["end_call"]


async def test_a_tool_filter_withholds_a_tool():
  """Some agents must never decide the conversation is over."""
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  call = make_call(make_room({"sip_caller": participant}))
  toolset = LiveKitToolset(tool_filter=["transfer_call", "send_dtmf"])

  assert await _toolset_names(toolset, call) == ["transfer_call", "send_dtmf"]


async def test_a_tool_name_prefix_keeps_the_tools_apart():
  """Two toolsets in one agent must not collide on `end_call`."""
  call = make_call(make_room({"browser": webrtc_participant()}))
  toolset = LiveKitToolset(tool_name_prefix="voice")

  with _livekit_call._register_call(call):
    tools = await toolset.get_tools_with_prefix(make_tool_context())

  assert [tool.name for tool in tools] == ["voice_end_call"]


async def test_the_toolset_goes_in_an_agents_tools_list():
  """The point of the toolset: no model_copy, no rebuilt Runner."""
  agent = LlmAgent(
      name="support_agent",
      model="gemini-live-2.5-flash",
      tools=[LiveKitToolset()],
  )
  call = make_call(make_room({"browser": webrtc_participant()}))

  with _livekit_call._register_call(call):
    tools = await agent.canonical_tools(make_tool_context())

  assert [tool.name for tool in tools] == ["end_call"]
