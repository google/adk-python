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

"""Tests for the Agent Builder Assistant factory."""

from __future__ import annotations

import re
from unittest import mock

from google.adk.cli.built_in_agents.adk_agent_builder_assistant import AgentBuilderAssistant

# Every name that has ever been a tool on this assistant, live or retired.
# Retired names are kept deliberately: deleting a name here at the same time as
# its tool is exactly what lets the instruction go on advertising a capability
# that no longer exists, which is the drift this list exists to catch.
_EVERY_TOOL_NAME_EVER = frozenset({
    'google_search_agent',
    'url_context_agent',
    'read_config_files',
    'write_config_files',
    'explore_project',
    'read_files',
    'write_files',
    'delete_files',
    'cleanup_unused_files',
    'search_adk_source',
    # Retired.
    'search_adk_knowledge',
})


def test_create_agent_exposes_the_full_agent_building_tool_set():
  agent = AgentBuilderAssistant.create_agent(model='gemini-2.0-flash')

  assert agent.name == 'agent_builder_assistant'
  # Every capability the assistant needs to build an agent from a prompt:
  # config, file, and ADK-lookup tools. A missing entry silently disables a
  # capability. There are deliberately no web tools: knowledge comes from the
  # installed package, which cannot describe a version the user does not have.
  # search_adk_knowledge is deliberately absent: it called a service that no
  # longer answers, and search_adk_source covers the same ground locally.
  assert {tool.name for tool in agent.tools} == {
      'read_config_files',
      'write_config_files',
      'explore_project',
      'read_files',
      'write_files',
      'delete_files',
      'cleanup_unused_files',
      'search_adk_source',
  }
  assert agent.generate_content_config.max_output_tokens == 8192


def test_create_agent_instruction_provider_fills_model_and_project_folder(
    tmp_path,
):
  project_dir = tmp_path / 'my_agent_project'
  project_dir.mkdir()
  context = mock.MagicMock()
  context._invocation_context.session.state = {
      'root_directory': str(project_dir)
  }

  agent = AgentBuilderAssistant.create_agent(model='gemini-2.0-flash')
  instruction = agent.instruction(context)

  # The instruction is resolved per invocation so it can name the session's
  # project folder; the schema and model are baked in at build time.
  assert 'gemini-2.0-flash' in instruction
  assert 'my_agent_project' in instruction
  assert 'ADK AgentConfig quick reference' in instruction
  # The schema placeholder itself was substituted, not left in the prompt.
  assert '{schema_content}' not in instruction


def test_instruction_never_names_a_tool_the_assistant_does_not_have(tmp_path):
  """The prompt and the tool set must not drift apart.

  Nothing else pins this. The instruction describes the tools in prose, and
  prose does not fail to compile: search_adk_knowledge kept its place at the
  top of the research ladder for a release after the tool had stopped working,
  which sent the model to a dead service and then to open web search.
  """
  project_dir = tmp_path / 'my_agent_project'
  project_dir.mkdir()
  context = mock.MagicMock()
  context._invocation_context.session.state = {
      'root_directory': str(project_dir)
  }

  agent = AgentBuilderAssistant.create_agent(model='gemini-2.0-flash')
  instruction = agent.instruction(context)
  tool_names = {tool.name for tool in agent.tools}

  # The allowlist is the only thing bounding the search below, so it has to hold
  # the live tools as well as the retired ones. A tool added without being
  # listed here is a tool this test does not watch: retire it later with a
  # mention left behind in the prompt, and nothing notices.
  assert tool_names <= _EVERY_TOOL_NAME_EVER, (
      f'{sorted(tool_names - _EVERY_TOOL_NAME_EVER)} is a tool on this'
      ' assistant but is missing from _EVERY_TOOL_NAME_EVER'
  )

  # Search for each known name directly rather than for prose marked up as
  # `name` or **name**. The prompt names tools bare as well -- "Use
  # google_search_agent for ..." -- so a markup filter would pass a retirement
  # that cleaned up only the marked-up mentions while the prompt still pointed
  # the model at the tool. The precision is in the allowlist, not the markup.
  named = {
      name
      for name in _EVERY_TOOL_NAME_EVER
      if re.search(rf'(?<![\w.]){re.escape(name)}(?![\w.])', instruction)
  }
  dangling = sorted(named - tool_names)

  assert not dangling, (
      f'the instruction names {dangling}, which the assistant does not have:'
      ' drop the reference, or restore the tool'
  )
