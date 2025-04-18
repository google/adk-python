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

from typing import TypedDict

from .test_runner import TestRunner


class Message(TypedDict):
  agent_name: str
  expected_text: str


def assert_current_agent_is(agent_name: str, *, agent_runner: TestRunner):
  assert agent_runner.get_current_agent_name() == agent_name


def assert_agent_says(
    expected_text: str, *, agent_name: str, agent_runner: TestRunner
):
  for event in reversed(agent_runner.get_events()):
    if event.author == agent_name and event.content.parts[0].text:
      assert event.content.parts[0].text.strip() == expected_text
      return


def assert_agent_says_in_order(
    expected_conversation: list[Message], agent_runner: TestRunner
):
  expected_conversation_idx = len(expected_conversation) - 1
  for event in reversed(agent_runner.get_events()):
    if event.content.parts and event.content.parts[0].text:
      assert (
          event.author
          == expected_conversation[expected_conversation_idx]['agent_name']
      )
      assert (
          event.content.parts[0].text.strip()
          == expected_conversation[expected_conversation_idx]['expected_text']
      )
      expected_conversation_idx -= 1
      if expected_conversation_idx < 0:
        return


def assert_agent_transfer_path(
    expected_path: list[str], *, agent_runner: TestRunner
):
  events = agent_runner.get_events()
  idx_in_expected_path = len(expected_path) - 1
  # iterate events in reverse order
  for event in reversed(events):
    function_calls = event.get_function_calls()
    if (
        len(function_calls) == 1
        and function_calls[0].name == 'transfer_to_agent'
    ):
      assert (
          function_calls[0].args['agent_name']
          == expected_path[idx_in_expected_path]
      )
      idx_in_expected_path -= 1
      if idx_in_expected_path < 0:
        return
