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

"""Sample agent demonstrating nested state access and optional chaining."""

from __future__ import annotations

from google.adk import Agent
from google.adk import Context


def inject_nested_state(callback_context: Context):
  callback_context.state["user"] = {
      "name": "Jainish",
      "profile": {"age": 24, "role": "Software Engineer"},
  }


agent = Agent(
    name="nested_state",
    instruction=(
        "Current user is {user?.name?} and {user?.profile?.role?}. Please"
        " greet them by name and designation."
    ),
    before_agent_callback=[inject_nested_state],
)

root_agent = agent
