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

"""Coordinator LlmAgent that delegates coding tasks to an AntigravityAgent.

An ADK ``LlmAgent`` is the root. It answers general questions itself and hands
hands-on file/coding work to an ``AntigravityAgent`` sub-agent that runs in a
scratch workspace.

The sub-agent sets ``mode='single_turn'``, which is what lets an
``AntigravityAgent`` have an ADK parent at all: the parent exposes it as an
inline tool taking a ``request`` string, composes each request itself, and does
not forward session history. See the AntigravityAgent guide at
docs/guides/labs/antigravity/index.md.
"""

import os

from google.adk import Agent
from google.adk.labs.antigravity import AntigravityAgent
from google.antigravity import LocalAgentConfig
from google.antigravity import types as antigravity_types
from google.antigravity.hooks import policy

# 1. Configure the Antigravity SDK coding agent, scoped to a scratch workspace.
_sample_dir = os.path.dirname(os.path.abspath(__file__))
_workspace = os.path.join(_sample_dir, "workspace")
os.makedirs(_workspace, exist_ok=True)
_sdk_config = LocalAgentConfig(
    system_instructions="""\
You are a coding assistant working inside a single scratch workspace. When asked \
to create or edit code, write real files into the workspace using clean absolute \
paths, keep each change small and self-contained, and finish by naming the files \
you touched and briefly summarizing what each one does.""",
    workspaces=[_workspace],
    policies=[
        # Approve the built-in file tools so the agent can read and write; a
        # local harness fails closed on any tool that no policy approves.
        # workspace_only() keeps those file operations contained to the
        # workspace directory.
        *[
            policy.allow(tool.value)
            for tool in antigravity_types.BuiltinTools.file_tools()
        ],
        *policy.workspace_only([_workspace]),
    ],
)

# 2. Wrap it as a single_turn ADK sub-agent. `mode='single_turn'` is required
#    for an AntigravityAgent to have an ADK parent: the parent composes a
#    self-contained request, and each call is an independent conversation.
code_writer = AntigravityAgent(
    name="code_writer",
    description=(
        "Writes and edits real code files in a scratch workspace. Delegate any"
        " request to create, modify, or inspect files on disk to this agent,"
        " passing a complete, self-contained description of the coding task."
    ),
    config=_sdk_config,
    mode="single_turn",
)

# 3. The ADK root: an LlmAgent that answers general questions itself and hands
#    hands-on coding work to the code_writer sub-agent. Because code_writer sets
#    mode='single_turn', the root exposes it as an inline tool rather than as an
#    LLM-transfer target.
root_agent = Agent(
    name="coordinator",
    model="gemini-3.8-flash",
    description="Coordinates general Q&A and delegates coding tasks.",
    instruction="""\
You are a helpful coordinator.
- Answer general questions, explanations, and planning yourself.
- When the user wants code written, edited, or files created on disk, delegate
  to the code_writer sub-agent. Compose a clear, self-contained task description
  for it rather than forwarding the user's words verbatim.
- After code_writer replies, summarize for the user what it built.""",
    sub_agents=[code_writer],
)
