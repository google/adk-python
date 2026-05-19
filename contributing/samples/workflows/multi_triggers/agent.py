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

from typing import Any

from google.adk import Event
from google.adk import Workflow


def make_uppercase(node_input: str):
  return node_input.upper()


def count_characters(node_input: str):
  return len(node_input)


def reverse_string(node_input: str):
  return node_input[::-1]


async def send_message(node_input: Any):
  yield Event(message=f"Triggered for input: {node_input}")


root_agent = Workflow(
    name="root_agent",
    edges=[(
        "START",
        (make_uppercase, count_characters, reverse_string),
        send_message,
    )],
    input_schema=str,
)
