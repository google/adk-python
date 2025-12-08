"""Tests for BaseLlmFlow._get_agent_to_run transfer-to-peers behavior."""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
import pytest

from ... import testing_utils


def make_agent_tree():
  root = LlmAgent(name='root')
  child1 = LlmAgent(name='child1')
  child2 = LlmAgent(name='child2')

  child1.parent_agent = root
  child2.parent_agent = root
  root.sub_agents = [child1, child2]
  return root, child1, child2


@pytest.mark.asyncio
async def test_transfer_to_sibling_disallowed_raises():
  root, child1, child2 = make_agent_tree()

  caller = child1
  caller.disallow_transfer_to_peers = True

  ctx = await testing_utils.create_invocation_context(caller)

  flow = BaseLlmFlow()

  with pytest.raises(ValueError) as exc:
    flow._get_agent_to_run(ctx, 'child2')


@pytest.mark.asyncio
async def test_transfer_to_sibling_allowed_returns_agent():
  root, child1, child2 = make_agent_tree()

  caller = child1
  caller.disallow_transfer_to_peers = False

  ctx = await testing_utils.create_invocation_context(caller)

  flow = BaseLlmFlow()
  agent = flow._get_agent_to_run(ctx, 'child2')
  assert agent is not None
  assert agent.name == 'child2'


@pytest.mark.asyncio
async def test_transfer_to_unknown_agent_raises():
  root, child1, child2 = make_agent_tree()

  caller = child1
  ctx = await testing_utils.create_invocation_context(caller)
  flow = BaseLlmFlow()

  with pytest.raises(ValueError) as exc:
    flow._get_agent_to_run(ctx, 'not_in_tree')
