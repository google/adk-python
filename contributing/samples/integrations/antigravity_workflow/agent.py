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

"""An ADK Workflow with an LlmAgent and an AntigravityAgent as nodes.

A two-node sequential ``Workflow``:

1. ``planner`` -- an ADK ``LlmAgent`` node that turns the user's request into a
   short, concrete build spec.
2. ``builder`` -- an ``AntigravityAgent`` node that receives that spec as its
   input and implements it by writing files into a scratch workspace.

A ``Workflow`` wires nodes through edges (output -> next node's input), not
through ADK ``sub_agents``, so the ``AntigravityAgent`` node is not adopted by an
ADK parent and does not need ``mode='single_turn'`` for that reason. It sets
``mode='single_turn'`` anyway because a workflow node is invoked once per run
with a freshly composed input, which is exactly single-turn semantics -- and it
keeps the harness from minting a throwaway save_dir per run. See the
AntigravityAgent guide at docs/guides/labs/antigravity/index.md.
"""

import os

from google.adk import Agent
from google.adk import Workflow
from google.adk.labs.antigravity import AntigravityAgent
from google.antigravity import LocalAgentConfig
from google.antigravity import types as antigravity_types
from google.antigravity.hooks import policy

# 1. The first node: an ADK LlmAgent that plans the build. As a workflow node,
#    an LlmAgent defaults to single_turn mode; its final text becomes this
#    node's output, which the workflow threads in as the next node's input.
planner = Agent(
    name="planner",
    model="gemini-3.8-flash",
    description=(
        "Turns a build request into a concrete spec for a coding agent."
    ),
    instruction="""\
You turn a build request into a short, concrete spec for a coding agent to
implement. Output a numbered list of the files to create and, for each, exactly
what it should contain (functions, behavior, edge cases). Be specific and
concise. Output only the spec, no preamble.""",
)

# 2. Configure the Antigravity SDK builder agent, scoped to a scratch workspace.
_sample_dir = os.path.dirname(os.path.abspath(__file__))
_workspace = os.path.join(_sample_dir, "workspace")
os.makedirs(_workspace, exist_ok=True)
_sdk_config = LocalAgentConfig(
    system_instructions="""\
You are a coding agent. You receive a build spec and implement it by writing the
described files into your workspace using clean absolute paths. Keep each file
small and self-contained. When done, name the files you created and summarize
what each one does.""",
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

# 3. The second node: an AntigravityAgent. It receives the planner's spec as its
#    node input and builds the files.
builder = AntigravityAgent(
    name="builder",
    description="Implements a build spec by writing files into the workspace.",
    config=_sdk_config,
    mode="single_turn",
)

# 4. The workflow: START -> planner -> builder. Each node's output is threaded in
#    as the next node's input.
root_agent = Workflow(
    name="antigravity_workflow",
    edges=[("START", planner, builder)],
)
