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

"""The compaction scenarios: a turn taken over a session that already has one.

Two graphs, one per kind of entrypoint the runner's post-invocation compaction
can report under. ``conversational_agent_compaction`` enters at the canonical
agent; ``workflow_compaction`` at the canonical workflow, whose root node runs
inside the entrypoint span the runner opens for it, in schema v1 as well.

Both run the canonical conversation against a seeded history, under an app
configured to compact.
"""

from __future__ import annotations

import contextlib
from typing import Literal

from google.adk.apps.app import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import InMemoryRunner
from google.genai.types import Content
from google.genai.types import GenerateContentResponseUsageMetadata
from google.genai.types import Part
from typing_extensions import assert_never

from ....testing_utils import MockModel
from .agent import build_test_agent
from .agent import build_test_workflow
from .conversation import AGENT_NAME
from .conversation import USER_PROMPT

# How a compaction case's app compacts. ``untriggered`` sets a threshold this
# turn never reaches, so the app can compact and this turn does not.
CompactionSetup = Literal["token_threshold", "sliding_window", "untriggered"]

_APP_NAME = "compaction_app"
_USER_ID = "compaction_user"
_SUMMARY = "a summary of the earlier turns"
_SEEDED_INVOCATION_COUNT = 3

# The seeded history reports a prompt far over the threshold while both of the
# turn's model responses report well under it (100 and 150). Compaction
# therefore fires on the first model call and on nothing after it, which is
# what keeps the recording a fixed size rather than a function of how many
# times the threshold is re-tested during the turn.
_TOKEN_THRESHOLD = 300
_SEEDED_PROMPT_TOKEN_COUNT = 500
# Far above anything the seeded history or the model responses report.
_UNREACHABLE_TOKEN_THRESHOLD = 1_000_000


def _seeded_history() -> list[Event]:
  """Returns the conversation the session already holds when a case starts.

  Timestamps are fixed rather than wall-clock so that what gets compacted is a
  property of the scenario and not of when it ran.
  """
  events: list[Event] = []
  for index in range(_SEEDED_INVOCATION_COUNT):
    invocation_id = f"seeded-invocation-{index}"
    events.append(
        Event(
            timestamp=float(index * 2 + 1),
            invocation_id=invocation_id,
            author="user",
            content=Content(
                role="user", parts=[Part.from_text(text=f"question {index}")]
            ),
        )
    )
    events.append(
        Event(
            timestamp=float(index * 2 + 2),
            invocation_id=invocation_id,
            author=AGENT_NAME,
            content=Content(
                role="model", parts=[Part.from_text(text=f"answer {index}")]
            ),
            # Only the newest of these is read, and only until the turn's first
            # model response reports its own usage.
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=_SEEDED_PROMPT_TOKEN_COUNT
            ),
        )
    )
  return events


def _compaction_config(setup: CompactionSetup) -> EventsCompactionConfig:
  """Returns the config an app compacts under for ``setup``."""
  summarizer = LlmEventSummarizer(
      llm=MockModel.create(responses=[Part.from_text(text=_SUMMARY)])
  )
  match setup:
    case "sliding_window":
      return EventsCompactionConfig(
          summarizer=summarizer,
          compaction_interval=_SEEDED_INVOCATION_COUNT,
          overlap_size=0,
      )
    case "token_threshold":
      token_threshold = _TOKEN_THRESHOLD
    case "untriggered":
      token_threshold = _UNREACHABLE_TOKEN_THRESHOLD
    case _:
      assert_never(setup)
  return EventsCompactionConfig(
      summarizer=summarizer,
      token_threshold=token_threshold,
      event_retention_size=0,
  )


def _build_compaction_app(
    model: BaseLlm, setup: CompactionSetup, *, workflow_root: bool
) -> App:
  """Builds an app that compacts under ``setup``."""
  return App(
      name=_APP_NAME,
      root_agent=(
          build_test_workflow(model)
          if workflow_root
          else build_test_agent(model)
      ),
      events_compaction_config=_compaction_config(setup),
  )


async def run_compaction_scenario(
    model: BaseLlm,
    *,
    setup: CompactionSetup,
    workflow_root: bool,
    event_sink: list[Event] | None = None,
) -> list[Event]:
  """Runs the canonical turn over a seeded session, and compacts it.

  Builds its own runner and session rather than going through
  ``run_agent_scenario``: the history has to be in place before the turn
  starts, and ``TestInMemoryRunner`` names the session's app for itself.

  Args:
    model: The model the agent runs against.
    setup: How the app compacts.
    workflow_root: Whether the app is rooted at the canonical workflow rather
      than the canonical agent.
    event_sink: Collects the events as they are drained, so a run that raises
      still reports what it emitted first.

  Returns:
    The events the turn emitted.
  """
  runner = InMemoryRunner(
      app=_build_compaction_app(model, setup, workflow_root=workflow_root)
  )
  session = await runner.session_service.create_session(
      app_name=_APP_NAME, user_id=_USER_ID
  )
  for event in _seeded_history():
    await runner.session_service.append_event(session=session, event=event)

  collected_events: list[Event] = event_sink if event_sink is not None else []
  async with contextlib.aclosing(
      runner.run_async(
          user_id=_USER_ID,
          session_id=session.id,
          new_message=Content(
              parts=[Part.from_text(text=USER_PROMPT)], role="user"
          ),
      )
  ) as agen:
    async for event in agen:
      collected_events.append(event)

  return collected_events
