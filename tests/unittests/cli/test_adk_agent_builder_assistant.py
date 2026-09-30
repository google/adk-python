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

from unittest import mock

from google.adk.cli.built_in_agents.adk_agent_builder_assistant import AgentBuilderAssistant


def test_create_agent_exposes_the_full_agent_building_tool_set():
  agent = AgentBuilderAssistant.create_agent(model='gemini-2.0-flash')

  assert agent.name == 'agent_builder_assistant'
  # Every capability the assistant needs to build an agent from a prompt:
  # the two built-in research agents (wrapped as tools) plus config, file,
  # and ADK-lookup tools. A missing entry silently disables a capability.
  assert {tool.name for tool in agent.tools} == {
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
      'search_adk_knowledge',
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
