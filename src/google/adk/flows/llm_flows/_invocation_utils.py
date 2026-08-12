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

"""Runtime invariants shared by LLM-flow processors."""

from __future__ import annotations

from typing import cast
from typing import TYPE_CHECKING

from ...agents.base_agent import BaseAgent
from ...agents.invocation_context import InvocationContext
from ...agents.run_config import RunConfig

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent


def require_agent(invocation_context: InvocationContext) -> BaseAgent:
  """Returns the agent required by processors that walk the agent tree."""
  agent = invocation_context.agent
  if not isinstance(agent, BaseAgent):
    raise TypeError('LLM flow requires a BaseAgent in InvocationContext.')
  return agent


def as_llm_agent(invocation_context: InvocationContext) -> LlmAgent:
  """Returns the invocation's agent, narrowed to what LLM flows read from it.

  Flows also drive agents defined outside this package that provide the
  LlmAgent surface without subclassing it, so this narrows statically; call
  sites that read an attribute those agents may not define still guard it.
  """
  agent = invocation_context.agent
  if agent is None:
    raise TypeError('LLM flow requires an agent in InvocationContext.')
  return cast('LlmAgent', agent)


def require_agent_name(invocation_context: InvocationContext) -> str:
  """Returns the name shared by agent and workflow-node invocations."""
  agent = invocation_context.agent
  if agent is None:
    raise TypeError('LLM flow requires an agent in InvocationContext.')
  return agent.name


def require_run_config(invocation_context: InvocationContext) -> RunConfig:
  """Returns the run configuration required by model execution."""
  run_config = invocation_context.run_config
  if run_config is None:
    raise ValueError('LLM flow requires a RunConfig in InvocationContext.')
  return run_config
