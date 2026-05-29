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
from google.adk.events.event import NodeInfo
from google.adk.events.request_input import RequestInput


def test_event_constructor_with_state():
  """Tests that the event constructor handles the state argument."""
  my_event = Event(state={"key": "value"})
  assert my_event.actions is not None
  assert my_event.actions.state_delta == {"key": "value"}


def test_event_constructor_without_state():
  """Tests that the event constructor works without the state argument."""
  my_event = Event()
  assert my_event.actions is not None
  assert my_event.actions.state_delta == {}


def test_event_serialization_always_camel_case():
  """Tests that Event serialization produces camelCase keys."""
  request_input = RequestInput(interrupt_id="fc-1", message="test")

  # Create an event with fields that would produce snake_case if not dumped by alias
  event = Event(
      invocation_id="i-1",
      node_info=NodeInfo(
          path="a/b",
          output_for=["c"],
          message_as_output=True,
      ),
      output=request_input,
  )

  dumped = event.model_dump(by_alias=True)

  def check_no_snake_case_keys(data):
    if isinstance(data, dict):
      for key, value in data.items():
        assert "_" not in key, f"Found snake_case key: {key} in {data}"
        check_no_snake_case_keys(value)
    elif isinstance(data, list):
      for item in data:
        check_no_snake_case_keys(item)

  check_no_snake_case_keys(dumped)

  # Also verify that expected keys are indeed camelCased
  assert "invocationId" in dumped
  assert "nodeInfo" in dumped
  assert "outputFor" in dumped["nodeInfo"]
  assert "messageAsOutput" in dumped["nodeInfo"]

  # Verify RequestInput fields are camelCased
  assert "output" in dumped
  assert "interruptId" in dumped["output"]


def test_event_isolation_scope():
  """Tests Event.isolation_scope default value and serialization."""
  ev = Event()
  assert ev.isolation_scope is None

  ev2 = Event(isolation_scope="task:fc-123")
  dumped = ev2.model_dump(mode="json", by_alias=True, exclude_none=True)
  assert dumped["isolationScope"] == "task:fc-123"
