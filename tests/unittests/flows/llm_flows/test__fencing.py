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
