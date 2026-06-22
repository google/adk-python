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

"""Tests for graph_serialization edge handling with routing maps."""

import json

from google.adk.agents import LlmAgent
from google.adk.cli.utils.graph_serialization import serialize_agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.base_toolset import BaseToolset
from google.adk.workflow import START
from google.adk.workflow import Workflow

from tests.unittests.workflow.workflow_testing_utils import TestingNode


def test_serialize_edges_with_routing_map() -> None:
  """Tests that routing map dicts in edges are serialized without error."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')

  agent = Workflow(
      name='test_workflow',
      edges=[
          (START, node_a),
          (node_a, {'route_b': node_b, 'route_c': node_c}),
      ],
  )

  result = serialize_agent(agent)

  serialized_edges = result['edges']
  assert len(serialized_edges) == 2

  # First edge: (START, node_a) — serialized as a 2-element list.
  assert len(serialized_edges[0]) == 2

  # Second edge: (node_a, {route: node}) — serialized as a 2-element list
  # where the second element is a dict with string keys.
  routing_map_edge = serialized_edges[1]
  assert len(routing_map_edge) == 2
  assert isinstance(routing_map_edge[1], dict)
  assert 'route_b' in routing_map_edge[1]
  assert 'route_c' in routing_map_edge[1]


def test_serialize_edges_with_routing_map_int_keys() -> None:
  """Tests that integer routing map keys are serialized as strings."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')

  agent = Workflow(
      name='test_workflow',
      edges=[
          (START, node_a),
          (node_a, {1: node_b}),
      ],
  )

  result = serialize_agent(agent)

  routing_map_edge = result['edges'][1]
  # Integer keys become string keys in the serialized output.
  assert '1' in routing_map_edge[1]


def test_serialize_edges_mixed_formats() -> None:
  """Tests serialization of edges mixing tuples, Edge objects, and routing maps."""
  from google.adk.workflow import Edge

  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  node_d = TestingNode(name='NodeD')

  agent = Workflow(
      name='test_workflow',
      edges=[
          (START, node_a),
          (node_a, {'route_b': node_b, 'route_c': node_c}),
          (node_b, node_d),
          Edge(from_node=node_c, to_node=node_d),
      ],
  )

  result = serialize_agent(agent)

  serialized_edges = result['edges']
  assert len(serialized_edges) == 4

  # Tuple edges are lists, Edge objects are dicts with from_node/to_node.
  assert isinstance(serialized_edges[0], list)  # (START, node_a)
  assert isinstance(serialized_edges[1], list)  # routing map
  assert isinstance(serialized_edges[2], list)  # (node_b, node_d)
  assert isinstance(serialized_edges[3], dict)  # Edge object
  assert 'from_node' in serialized_edges[3]


def test_serialize_agent_with_toolset() -> None:
  """Tests that toolsets are serialized using their class name."""

  class MockToolset(BaseToolset):

    async def get_tools(self, readonly_context=None):
      return []

  class FakeAgent:
    model_fields = {'tools': None}

    def __init__(self):
      self.tools = [MockToolset()]

  agent = FakeAgent()
  result = serialize_agent(agent)  # type: ignore

  assert 'tools' in result
  assert len(result['tools']) == 1
  assert result['tools'][0]['name'] == 'MockToolset'
  assert result['tools'][0]['type'] == 'tool'


def test_serialize_agent_with_litellm_model_is_json_safe() -> None:
  agent = LlmAgent(
      name='repro',
      model=LiteLlm(model='ollama_chat/llama3'),
  )

  result = serialize_agent(agent)

  assert result['model'] == 'ollama_chat/llama3'
  json.dumps(result)


def test_serialize_agent_skips_excluded_fields() -> None:
  """Fields marked Field(exclude=True) are omitted from serialization."""
  from typing import Any

  from google.adk.agents.base_agent import BaseAgent
  from pydantic import ConfigDict
  from pydantic import Field

  class _Agent(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    secret: Any = Field(default=None, exclude=True)

  agent = _Agent(name='a', secret=lambda: None)
  result = serialize_agent(agent)

  assert 'secret' not in result
  assert result['name'] == 'a'
