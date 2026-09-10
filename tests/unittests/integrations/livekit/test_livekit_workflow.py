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

"""Tests for driving a multi-stage `Workflow` over a LiveKit room.

A `Workflow` root takes a different path through `Runner.run_live` than a
plain agent -- a producer task feeding an event queue rather than an inline
generator -- and opens one model connection per stage. Everything the bridge
does with the resulting events is covered in `test_livekit_runner.py`; what is
left to check is that it still sees them across that boundary.
"""

from __future__ import annotations

import asyncio
import json

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.genai import types
import pytest

pytest.importorskip("livekit.rtc")

from tests.unittests.integrations.livekit.conftest import final_transcripts
from tests.unittests.integrations.livekit.conftest import make_lk_runner
from tests.unittests.integrations.livekit.conftest import make_room
from tests.unittests.testing_utils import MockModel


def _stage(name: str, *, says: str) -> LlmAgent:
  """One task-mode live stage that speaks once, then completes.

  A real live model reports its own speech as a transcription rather than as
  text content, so the stage emits the fragments and the finished utterance
  the way `gemini_llm_connection` does -- otherwise nothing would exercise the
  caption path.
  """
  head, tail = says[: len(says) // 2], says[len(says) // 2 :]
  model = MockModel.create(
      responses=[
          LlmResponse(
              partial=True,
              output_transcription=types.Transcription(
                  text=head, finished=False
              ),
          ),
          LlmResponse(
              partial=True,
              output_transcription=types.Transcription(
                  text=tail, finished=False
              ),
          ),
          LlmResponse(
              partial=False,
              output_transcription=types.Transcription(
                  text=says, finished=True
              ),
          ),
          LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[
                      types.Part.from_function_call(
                          name="finish_task", args={"result": f"{name}_done"}
                      )
                  ],
              )
          ),
      ]
  )
  return LlmAgent(name=name, model=model, mode="task", instruction=name)


async def test_every_stage_of_a_workflow_reaches_the_room():
  """A three-stage call runs to completion and each stage is heard.

  Setup: a greet -> verify -> deliver workflow, each stage speaking once and
    then calling `finish_task`. The run_config replaces the connector's
    default wholesale, so transcription is set explicitly.
  Act: run the session to completion.
  Assert: all three stages ran, and each published its own caption segment
    rather than extending the previous stage's.
  """
  room = make_room()
  workflow = Workflow(
      name="care_call",
      edges=[
          (START, greeter := _stage("greeter", says="Am I speaking with Jo?")),
          (greeter, verifier := _stage("verifier", says="Date of birth?")),
          (verifier, _stage("goals", says="Your appointment is Tuesday.")),
      ],
  )
  lk_runner = make_lk_runner(
      InMemoryRunner(agent=workflow, app_name="care_call"),
      room,
      run_config=RunConfig(
          response_modalities=[types.Modality.AUDIO],
          input_audio_transcription=types.AudioTranscriptionConfig(),
          output_audio_transcription=types.AudioTranscriptionConfig(),
      ),
  )

  await asyncio.wait_for(lk_runner.start(), timeout=10)

  finished = [
      json.loads(call.args[0])["name"]
      for call in room.local_participant.publish_data.await_args_list
      if json.loads(call.args[0])["type"] == "function_call"
  ]
  assert finished.count("finish_task") == 3
  captions = final_transcripts(room)
  assert [writer.text for writer in captions] == [
      "Am I speaking with Jo?",
      "Date of birth?",
      "Your appointment is Tuesday.",
  ]
  assert len({writer.attributes["lk.segment_id"] for writer in captions}) == 3
