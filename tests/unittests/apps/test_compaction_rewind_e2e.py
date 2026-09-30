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

"""End-to-end test: rewind + sliding-window event compaction.

Specifies the expected behavior at the boundary between rewind and
sliding-window compaction. Rewind ("undo a past invocation") only appends a
marker event (`rewind_before_invocation_id`); it does not delete events. The
expected behavior is that rewound content never reaches the agent LLM: when
building a prompt the contents processor honors the marker and drops the rewound
events, and the post-invocation sliding-window compactor honors it too, so a
compaction summary covers only live events.

Both paths share the `_apply_rewinds` helper as the single source of truth for
"which events are live" after rewinds.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.apps.app import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai.types import Content
from google.genai.types import Part
import pytest

from .. import testing_utils

# Unique text carried only by the invocation that the user rewinds (undoes).
_REWOUND_TEXT = "SECRET_REWOUND_CONTENT"

# Prefix the echo summarizer stamps on every summary, so the test can confirm a
# compaction summary actually reached the agent's prompt.
_SUMMARY_MARKER = "COMPACTED_SUMMARY_MARKER"


def _all_request_text(requests):
  """Returns every text part across the given LLM requests, joined."""
  texts = []
  for request in requests:
    for content in request.contents or []:
      for part in content.parts or []:
        if part.text:
          texts.append(part.text)
  return "\n".join(texts)


class _EchoSummarizerModel(testing_utils.MockModel):
  """A summarizer model whose summary echoes the text it was asked to compact."""

  async def generate_content_async(self, llm_request, stream=False):
    self.requests.append(llm_request)
    echoed = _all_request_text([llm_request])
    yield LlmResponse(
        content=testing_utils.ModelContent(
            [Part(text=f"{_SUMMARY_MARKER} {echoed}")]
        )
    )


@pytest.mark.asyncio
async def test_rewind_with_sliding_window_compaction():
  """Rewound content must not be fed to the agent LLM."""
  summarizer_model = _EchoSummarizerModel.create(responses=["unused"])
  agent_model = testing_utils.MockModel.create(
      responses=["answer one", "answer two"]
  )
  agent = Agent(name="agent", model=agent_model)
  app = App(
      name="test_app",
      root_agent=agent,
      events_compaction_config=EventsCompactionConfig(
          compaction_interval=2,
          overlap_size=0,
          summarizer=LlmEventSummarizer(llm=summarizer_model),
      ),
  )
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="u1", session_id="s1"
  )

  events = [
      Event(
          timestamp=1.0,
          invocation_id="inv1",
          author="user",
          content=Content(role="user", parts=[Part(text="hello")]),
      ),
      Event(
          timestamp=2.0,
          invocation_id="inv1",
          author="agent",
          content=Content(role="model", parts=[Part(text="hi")]),
      ),
      # This invocation is undone below via a rewind marker.
      Event(
          timestamp=3.0,
          invocation_id="inv_to_rewind",
          author="user",
          content=Content(role="user", parts=[Part(text=_REWOUND_TEXT)]),
      ),
      Event(
          timestamp=4.0,
          invocation_id="inv_to_rewind",
          author="agent",
          content=Content(role="model", parts=[Part(text="acknowledged")]),
      ),
      # Rewind marker undoing inv_to_rewind.
      Event(
          timestamp=5.0,
          invocation_id="rewind_inv",
          author="user",
          actions=EventActions(rewind_before_invocation_id="inv_to_rewind"),
      ),
  ]
  for event in events:
    await session_service.append_event(session=session, event=event)

  runner = Runner(app=app, session_service=session_service)

  # Turn 1: the agent runs, then post-invocation sliding-window compaction
  # fires over the accumulated invocations -- should drop the rewound events.
  async for _ in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=Content(role="user", parts=[Part(text="next")]),
  ):
    pass

  # Turn 2: the agent now builds its prompt from the compacted history, which
  # substitutes the compaction summary for the raw events.
  async for _ in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=Content(role="user", parts=[Part(text="again")]),
  ):
    pass

  final_agent_prompt = _all_request_text([agent_model.requests[-1]])

  # Sanity check: compaction ran and its summary was
  # substituted into the agent's prompt. Without this, the contract assertion
  # below could pass trivially -- e.g. if compaction never fired, or its
  # summary never reached the agent -- giving a false green.
  assert summarizer_model.requests, "sliding-window compaction did not run"
  assert (
      _SUMMARY_MARKER in final_agent_prompt
  ), "compaction summary did not reach the agent prompt"

  # Content the user rewound must NEVER reach the summarizer LLM: the compactor
  # drops rewound events before summarizing.
  summarizer_input = _all_request_text(summarizer_model.requests)
  assert _REWOUND_TEXT not in summarizer_input

  # ...and, consequently, must never reach the agent LLM.
  assert _REWOUND_TEXT not in final_agent_prompt


@pytest.mark.asyncio
async def test_rewind_with_token_threshold_compaction():
  """Token-threshold compaction must not feed rewound content to the summarizer.

  Mirrors the sliding-window scenario but configures token-threshold compaction
  (``token_threshold`` + ``event_retention_size``, no sliding-window config) so
  the token-threshold path runs. The rewound invocation must be dropped before
  summarization, so it reaches neither the summarizer nor the next prompt.
  """
  summarizer_model = _EchoSummarizerModel.create(responses=["unused"])
  agent_model = testing_utils.MockModel.create(
      responses=["answer one", "answer two"]
  )
  agent = Agent(name="agent", model=agent_model)
  app = App(
      name="test_app",
      root_agent=agent,
      events_compaction_config=EventsCompactionConfig(
          # High interval so the sliding-window path never triggers; only the
          # token-threshold path (threshold=1) can fire in this test.
          compaction_interval=1000,
          overlap_size=0,
          token_threshold=1,
          event_retention_size=0,
          summarizer=LlmEventSummarizer(llm=summarizer_model),
      ),
  )
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="u1", session_id="s1"
  )

  events = [
      Event(
          timestamp=1.0,
          invocation_id="inv1",
          author="user",
          content=Content(role="user", parts=[Part(text="hello")]),
      ),
      Event(
          timestamp=2.0,
          invocation_id="inv1",
          author="agent",
          content=Content(role="model", parts=[Part(text="hi")]),
      ),
      # This invocation is undone below via a rewind marker.
      Event(
          timestamp=3.0,
          invocation_id="inv_to_rewind",
          author="user",
          content=Content(role="user", parts=[Part(text=_REWOUND_TEXT)]),
      ),
      Event(
          timestamp=4.0,
          invocation_id="inv_to_rewind",
          author="agent",
          content=Content(role="model", parts=[Part(text="acknowledged")]),
      ),
      # Rewind marker undoing inv_to_rewind.
      Event(
          timestamp=5.0,
          invocation_id="rewind_inv",
          author="user",
          actions=EventActions(rewind_before_invocation_id="inv_to_rewind"),
      ),
  ]
  for event in events:
    await session_service.append_event(session=session, event=event)

  runner = Runner(app=app, session_service=session_service)

  # Turn 1: the agent runs, then post-invocation token-threshold compaction
  # fires over the accumulated invocations.
  async for _ in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=Content(role="user", parts=[Part(text="next")]),
  ):
    pass

  # Turn 2: the agent now builds its prompt from the compacted history.
  async for _ in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=Content(role="user", parts=[Part(text="again")]),
  ):
    pass

  final_agent_prompt = _all_request_text([agent_model.requests[-1]])

  # Sanity: token-threshold compaction ran and its summary reached the agent.
  assert summarizer_model.requests, "token-threshold compaction did not run"
  assert (
      _SUMMARY_MARKER in final_agent_prompt
  ), "compaction summary did not reach the agent prompt"

  # Content the user rewound must NEVER reach the summarizer LLM...
  summarizer_input = _all_request_text(summarizer_model.requests)
  assert _REWOUND_TEXT not in summarizer_input

  # ...and, consequently, must never reach the agent LLM.
  assert _REWOUND_TEXT not in final_agent_prompt
