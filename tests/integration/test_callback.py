# Copyright 2025 Google LLC
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

from pytest import mark

from ..unittests.testing_utils import simplify_events
from .fixture import callback_agent
from .utils import assert_agent_says
from .utils import TestRunner


@mark.parametrize(
    "agent_runner",
    [{"agent": callback_agent.agent.before_agent_callback_agent}],
    indirect=True,
)
def test_before_agent_call(agent_runner: TestRunner):
  agent_runner.run("Hi.")

  # Assert the response content
  assert_agent_says(
      "End invocation event before agent call.",
      agent_name="before_agent_callback_agent",
      agent_runner=agent_runner,
  )


@mark.parametrize(
    "agent_runner",
    [{"agent": callback_agent.agent.before_model_callback_agent}],
    indirect=True,
)
def test_before_model_call(agent_runner: TestRunner):
  agent_runner.run("Hi.")

  # Assert the response content
  assert_agent_says(
      "End invocation event before model call.",
      agent_name="before_model_callback_agent",
      agent_runner=agent_runner,
  )


# TODO: re-enable vertex by removing below line after fixing.
@mark.parametrize("llm_backend", ["GOOGLE_AI"], indirect=True)
@mark.parametrize(
    "agent_runner",
    [{"agent": callback_agent.agent.after_model_callback_agent}],
    indirect=True,
)
def test_after_model_call(agent_runner: TestRunner):
  events = agent_runner.run("Hi.")

  # Assert the response content
  simplified_events = simplify_events(events)
  assert simplified_events[0][0] == "after_model_callback_agent"
  assert simplified_events[0][1].endswith(
      "Update response event after model call."
  )
