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

"""Unit tests for flows.llm_flows.core._function_call_postprocessor."""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.flows.llm_flows.core import _function_call_postprocessor
import pytest

from .... import testing_utils


def _make_agent_tree():
  root = Agent(name='root')
  child1 = Agent(name='child1')
  child2 = Agent(name='child2')

  child1.parent_agent = root
  child2.parent_agent = root
  root.sub_agents = [child1, child2]
  return root, child1, child2


@pytest.mark.asyncio
async def test_transfer_to_sibling_disallowed_raises_value_error():
  """Transfer to sibling raises ValueError when disallow_transfer_to_peers is True."""
  _, child1, _ = _make_agent_tree()
  caller = child1
  caller.disallow_transfer_to_peers = True
  ctx = await testing_utils.create_invocation_context(caller)

  with pytest.raises(
      ValueError, match='child1 is not allowed to transfer to agent child2'
  ):
    _function_call_postprocessor.get_agent_to_run(ctx, 'child2')


@pytest.mark.asyncio
async def test_transfer_to_sibling_allowed_returns_agent():
  """Transfer to sibling returns the agent when disallow_transfer_to_peers is False."""
  _, child1, _ = _make_agent_tree()
  caller = child1
  caller.disallow_transfer_to_peers = False
  ctx = await testing_utils.create_invocation_context(caller)

  agent = _function_call_postprocessor.get_agent_to_run(ctx, 'child2')

  assert agent is not None
  assert agent.name == 'child2'


@pytest.mark.asyncio
async def test_transfer_to_unknown_agent_raises_value_error():
  """Transfer to unknown agent name raises ValueError."""
  _, child1, _ = _make_agent_tree()
  caller = child1
  ctx = await testing_utils.create_invocation_context(caller)

  with pytest.raises(ValueError, match='not found in the agent tree'):
    _function_call_postprocessor.get_agent_to_run(ctx, 'not_in_tree')


@pytest.mark.asyncio
async def test_transfer_to_self_allowed_when_peers_disallowed():
  """Transfer to self is allowed even when disallow_transfer_to_peers is True."""
  _, child1, _ = _make_agent_tree()
  caller = child1
  caller.disallow_transfer_to_peers = True
  ctx = await testing_utils.create_invocation_context(caller)

  agent = _function_call_postprocessor.get_agent_to_run(ctx, 'child1')

  assert agent is not None
  assert agent.name == 'child1'


@pytest.mark.asyncio
async def test_transfer_to_sibling_from_non_llm_agent_allowed():
  """Transfer to sibling is allowed when the caller is not an LlmAgent."""
  root = Agent(name='root')
  child1 = LoopAgent(name='child1')
  child2 = Agent(name='child2')

  child1.parent_agent = root
  child2.parent_agent = root
  root.sub_agents = [child1, child2]

  ctx = await testing_utils.create_invocation_context(child1)

  agent = _function_call_postprocessor.get_agent_to_run(ctx, 'child2')

  assert agent is not None
  assert agent.name == 'child2'


@pytest.mark.asyncio
async def test_transfer_to_unoffered_agent_raises_value_error():
  """Transfer to an agent that is only reachable through the tree is rejected."""
  _, child1, child2 = _make_agent_tree()
  grandchild2 = Agent(name='grandchild2')
  grandchild2.parent_agent = child2
  child2.sub_agents = [grandchild2]
  ctx = await testing_utils.create_invocation_context(child1)

  with pytest.raises(
      ValueError, match='child1 is not allowed to transfer to agent grandchild2'
  ):
    _function_call_postprocessor.get_agent_to_run(ctx, 'grandchild2')


@pytest.mark.asyncio
async def test_transfer_to_duplicate_name_returns_declared_target():
  """Transfer resolves the declared target, not a same-named agent elsewhere."""
  undeclared = Agent(name='shared_name')
  other_branch = Agent(name='other_branch', sub_agents=[undeclared])
  declared = Agent(name='shared_name')
  caller = Agent(
      name='caller',
      sub_agents=[declared],
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  Agent(name='root', sub_agents=[other_branch, caller])
  ctx = await testing_utils.create_invocation_context(caller)

  agent = _function_call_postprocessor.get_agent_to_run(ctx, 'shared_name')

  assert agent is declared


@pytest.mark.asyncio
async def test_transfer_to_self_returns_caller_when_name_is_duplicated():
  """Transfer to self returns the caller, not a same-named agent elsewhere."""
  namesake = Agent(name='caller')
  other_branch = Agent(name='other_branch', sub_agents=[namesake])
  caller = Agent(name='caller')
  Agent(name='root', sub_agents=[other_branch, caller])
  ctx = await testing_utils.create_invocation_context(caller)

  agent = _function_call_postprocessor.get_agent_to_run(ctx, 'caller')

  assert agent is caller


@pytest.mark.asyncio
async def test_transfer_to_parent_disallowed_raises_value_error():
  """Transfer to parent raises ValueError when disallow_transfer_to_parent is True."""
  _, child1, _ = _make_agent_tree()
  child1.disallow_transfer_to_parent = True
  ctx = await testing_utils.create_invocation_context(child1)

  with pytest.raises(
      ValueError, match='child1 is not allowed to transfer to agent root'
  ):
    _function_call_postprocessor.get_agent_to_run(ctx, 'root')


@pytest.mark.asyncio
async def test_transfer_to_parent_allowed_returns_agent():
  """Transfer to parent returns the agent when it is not disallowed."""
  _, child1, _ = _make_agent_tree()
  ctx = await testing_utils.create_invocation_context(child1)

  agent = _function_call_postprocessor.get_agent_to_run(ctx, 'root')

  assert agent is not None
  assert agent.name == 'root'
