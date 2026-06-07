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

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent
from google.adk.cli.utils.state import create_empty_state
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow


def test_create_empty_state_reads_agent_tree():
  child = LlmAgent(name='child', instruction='Use {child_key}')
  root = LlmAgent(
      name='root',
      instruction='Use {root_key}',
      sub_agents=[child],
  )

  assert create_empty_state(root) == {
      'child_key': '',
      'root_key': '',
  }


def test_create_empty_state_reads_workflow_graph_nodes():
  node = LlmAgent(name='node', instruction='Use {workflow_key}')
  workflow = Workflow(name='workflow', edges=[(START, node)])

  assert create_empty_state(workflow) == {'workflow_key': ''}


def test_create_empty_state_skips_initialized_workflow_state():
  node = LlmAgent(name='node', instruction='Use {workflow_key} and {fresh_key}')
  workflow = Workflow(name='workflow', edges=[(START, node)])

  assert create_empty_state(workflow, {'workflow_key': 'set'}) == {
      'fresh_key': ''
  }
