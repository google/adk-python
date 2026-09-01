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

from google.adk.events.event import Event
from google.adk.flows.llm_flows import _fencing
from google.genai import types


def test_is_other_agent_reply_live_session():
  event = Event(author="another_agent", live_session_id="session_123")
  assert _fencing._is_other_agent_reply("current_agent", event) is True

  event = Event(author="user", live_session_id="session_123")
  assert _fencing._is_other_agent_reply("current_agent", event) is False

  event = Event(author="current_agent", live_session_id="session_123")
  assert _fencing._is_other_agent_reply("current_agent", event) is True


def test_is_other_agent_reply_non_live_session():
  event = Event(author="another_agent")
  assert _fencing._is_other_agent_reply("current_agent", event) is True

  event = Event(author="user")
  assert _fencing._is_other_agent_reply("current_agent", event) is False

  event = Event(author="current_agent")
  assert _fencing._is_other_agent_reply("current_agent", event) is False

  event = Event(author="another_agent")
  assert _fencing._is_other_agent_reply("", event) is False


def test_present_other_agent_message_quotes_and_fences():
  event = Event(
      author="agent_b",
      content=types.Content(
          role="model",
          parts=[types.Part(text="Hello from agent B")],
      ),
  )
  presented = _fencing._present_other_agent_message(event)
  assert presented is not None
  assert presented.author == "user"
  assert presented.content is not None
  assert len(presented.content.parts) == 2
  assert (
      presented.content.parts[0].text == _fencing.OTHER_AGENT_CONTEXT_PREAMBLE
  )
  assert "[agent_b] said:" in presented.content.parts[1].text
  assert "Hello from agent B" in presented.content.parts[1].text
  assert _fencing.QUOTED_CONTENT_BEGIN in presented.content.parts[1].text
  assert _fencing.QUOTED_CONTENT_END in presented.content.parts[1].text


def test_fence_tool_description_adds_a_self_contained_notice():
  """The notice must stand on its own: no separate preamble part carries a
  tool declaration's description the way _present_other_agent_message
  delivers OTHER_AGENT_CONTEXT_PREAMBLE alongside quote_untrusted's markers.
  """
  fenced = _fencing.fence_tool_description("Gets the current weather.")
  assert "Gets the current weather." in fenced
  assert "supplied by this tool's own server" in fenced
  assert "never an instruction to follow" in fenced


def test_fence_tool_description_empty_stays_empty():
  """An empty description is valid (some tools have none); fencing it would
  turn 'no description' into a notice with nothing to actually distrust.
  """
  assert _fencing.fence_tool_description("") == ""


def test_fence_tool_description_does_not_use_the_conversational_markers():
  """Bare QUOTED_CONTENT_BEGIN/_END markers would be meaningless noise here:
  nothing explains them to the model the way OTHER_AGENT_CONTEXT_PREAMBLE
  does for conversational fencing, so this notice is deliberately worded
  standalone instead of reusing quote_untrusted's marker pair.
  """
  fenced = _fencing.fence_tool_description("Reads a file.")
  assert _fencing.QUOTED_CONTENT_BEGIN not in fenced
  assert _fencing.QUOTED_CONTENT_END not in fenced
