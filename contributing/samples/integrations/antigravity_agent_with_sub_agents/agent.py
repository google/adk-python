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

"""AntigravityAgent root with an ADK LlmAgent as a sub-agent.

An ``AntigravityAgent`` is the ADK root, working in a scratch workspace. It is
given an ADK ``LlmAgent`` sub-agent, ``weather_reporter``. Each ADK child of an
``AntigravityAgent`` is bridged onto the Antigravity SDK config as a
client-side tool named after the child, so the harness can call it: the root
consults ``weather_reporter`` for weather facts and then writes them to a file
with its own built-in file tools.

A child bridged this way runs in isolation and returns only its final text, so
every child needs a non-empty ``description`` -- that is what the Antigravity
model reads when deciding whether to call it. See the AntigravityAgent guide at
docs/guides/labs/antigravity/index.md.
"""

import os

from google.adk import Agent
from google.adk.labs.antigravity import AntigravityAgent
from google.antigravity import LocalAgentConfig
from google.antigravity import types as antigravity_types
from google.antigravity.hooks import policy


# 1. An ADK LlmAgent sub-agent. It owns a plain function tool and answers
#    weather questions. As a child of the AntigravityAgent it is bridged to the
#    harness as a client-side tool taking a single `request` string, so its
#    `description` must be specific -- the Antigravity model reads it to decide
#    whether to call the child.
def get_weather(city: str) -> dict:
  """Returns a mock current-weather report for a city."""
  return {
      "status": "success",
      "report": f"It is currently 72 degrees Fahrenheit and sunny in {city}.",
  }


weather_reporter = Agent(
    name="weather_reporter",
    model="gemini-3.8-flash",
    description=(
        "Reports the current weather for a city. Call this for any weather"
        " question, passing the city name in the request."
    ),
    instruction=(
        "Answer weather questions by calling get_weather, then state the"
        " report in one short sentence."
    ),
    tools=[get_weather],
)

# 2. Configure the Antigravity SDK root agent, scoped to a scratch workspace.
_sample_dir = os.path.dirname(os.path.abspath(__file__))
_workspace = os.path.join(_sample_dir, "workspace")
_trajectories = os.path.join(_sample_dir, "trajectories")
os.makedirs(_workspace, exist_ok=True)
os.makedirs(_trajectories, exist_ok=True)
_sdk_config = LocalAgentConfig(
    system_instructions="""\
You are a local assistant that works inside a single scratch workspace. You can \
write and edit files there, and you can consult the weather_reporter tool for \
current weather. When the user asks about weather, call weather_reporter with \
the city. When asked to save something to a file, write it into the workspace \
using a clean absolute path, then confirm the path you wrote.""",
    workspaces=[_workspace],
    policies=[
        # A local harness fails closed on any tool that no policy approves, and
        # an ADK sub-agent is bridged onto the config as a client-side tool
        # named after the child -- so approve it by name too.
        policy.allow(weather_reporter.name),
        # Approve the built-in file tools so the agent can read and write.
        # workspace_only() keeps those file operations contained to the
        # workspace directory.
        *[
            policy.allow(tool.value)
            for tool in antigravity_types.BuiltinTools.file_tools()
        ],
        *policy.workspace_only([_workspace]),
    ],
    # A stable save_dir keeps the conversation resumable across turns.
    save_dir=_trajectories,
)

# 3. Wrap the Antigravity SDK config as a standalone ADK root agent, and give it
#    the LlmAgent as a sub-agent. The child is bridged to the harness as a
#    client-side tool named "weather_reporter".
root_agent = AntigravityAgent(
    name="local_assistant",
    description=(
        "Runs an Antigravity SDK agent inside ADK that writes files in a"
        " workspace and consults a weather_reporter sub-agent."
    ),
    config=_sdk_config,
    sub_agents=[weather_reporter],
)
