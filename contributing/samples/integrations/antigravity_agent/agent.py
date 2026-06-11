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

"""Game-developer agent that writes browser games as self-contained HTML.

Wraps a Google Antigravity SDK agent as an ADK agent. See the package README
for setup and details.
"""

import os

from google.adk.labs.antigravity import AntigravityAgent
from google.antigravity import LocalAgentConfig
from google.antigravity.hooks import policy

# 1. Configure the Google Antigravity SDK game-developer agent. The
#    workspace-scoped policy lets it create and edit files inside the game_repo
#    workspace (built-in file tools are allowed there) while keeping writes
#    contained.
_sample_dir = os.path.dirname(os.path.abspath(__file__))
_workspace = os.path.join(_sample_dir, "game_repo")
_trajectories = os.path.join(_sample_dir, "trajectories")
os.makedirs(_workspace, exist_ok=True)
os.makedirs(_trajectories, exist_ok=True)
_sdk_config = LocalAgentConfig(
    system_instructions="""\
You are a senior web game developer. You build small, runnable games on request \
as a single self-contained HTML file with inline CSS and JavaScript (no external \
assets or third-party dependencies). Write the HTML file into the allowed \
workspace using a clean absolute filesystem path.

Build the file incrementally: first create it with a minimal skeleton (HTML \
structure, canvas, and empty script), then add CSS and the game logic over a \
few substantial edits. Group each edit around a complete feature (e.g. all \
styling, then rendering, then input handling) rather than many tiny changes, \
but do not attempt to write the entire game in one step.

After the file is complete, briefly explain how to play it (open the .html file \
in a browser).""",
    workspaces=[_workspace],
    policies=[*policy.workspace_only([_workspace])],
    save_dir=_trajectories,
)

# 2. Wrap the SDK config as a standalone ADK root agent.
root_agent = AntigravityAgent(
    name="antigravity_game_developer",
    description=(
        "Builds small, runnable games inside the game_repo workspace via the"
        " Antigravity SDK."
    ),
    config=_sdk_config,
)
