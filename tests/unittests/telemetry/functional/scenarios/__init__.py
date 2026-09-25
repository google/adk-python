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

"""The end-to-end scenarios the functional tests record.

One per graph shape, listed in ``Scenario``, each with its own
``run_*_scenario`` and each recorded under both inference instrumentations.

This package names the scenarios; the pieces they are built from live in it,
one module each:

* ``telemetry_setup``: ``install_telemetry`` points ADK's telemetry globals
  at in-memory exporters.
* ``conversation``: the names, token usage and canned turns every scenario
  shares.
* ``agent``: the canonical agent and the workflows wrapping it.
* ``inference``: ``inference_under_test`` hands out the model to run with,
  its instrumentation already active.
* ``mcp``: the same agent, with its tool served over (fake) MCP.
* ``skill``: the skill-loading agent.
"""

from __future__ import annotations

from typing import Literal

# Which end-to-end scenario a test case drives. The last three are variants of
# `agent` and `node`, named rather than flagged: which graph a case drives is
# what the case is, so it belongs here and not in a boolean on the case.
Scenario = Literal[
    "agent",
    "node",
    "mcp",
    "skill",
    "multi_agent",
    "agent_tool",
    "nested_agents_in_workflow",
    "streaming",
]

__all__ = ["Scenario"]
