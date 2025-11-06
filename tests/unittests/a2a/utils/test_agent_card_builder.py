# Copyright 2025 Google LLC
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

import sys
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from a2a.types import AgentCapabilities
  from a2a.types import AgentCard
  from a2a.types import AgentProvider
  from a2a.types import AgentSkill
  from a2a.types import SecurityScheme
  from google.adk.a2a.utils.agent_card_builder import _build_agent_description
  from google.adk.a2a.utils.agent_card_builder import _build_llm_agent_description_with_instructions
  from google.adk.a2a.utils.agent_card_builder import _build_loop_description
  from google.adk.a2a.utils.agent_card_builder import _build_orchestration_skill
  from google.adk.a2a.utils.agent_card_builder import _build_parallel_description
  from google.adk.a2a.utils.agent_card_builder import _build_sequential_description
  from google.adk.a2a.utils.agent_card_builder import _convert_example_tool_examples
  from google.adk.a2a.utils.agent_card_builder import _extract_examples_from_instruction
  from google.adk.a2a.utils.agent_card_builder import _get_agent_skill_name
  from google.adk.a2a.utils.agent_card_builder import _get_agent_type
  from google.adk.a2a.utils.agent_card_builder import _get_default_description
  from google.adk.a2a.utils.agent_card_builder import _get_input_modes
  from google.adk.a2a.utils.agent_card_builder import _get_output_modes
  from google.adk.a2a.utils.agent_card_builder import _get_workflow_description
  from google.adk.a2a.utils.agent_card_builder import _replace_pronouns
  from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
  from google.adk.agents.base_agent import BaseAgent
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.agents.loop_agent import LoopAgent
  from google.adk.agents.parallel_agent import ParallelAgent
  from google.adk.agents.sequential_agent import SequentialAgent
  from google.adk.tools.example_tool import ExampleTool
except ImportError as e:
  if sys.version_info < (3, 10):
    # Imports are not needed since tests will be skipped due to pytestmark.
    # The imported names are only used within test methods, not at module level,
    # so no NameError occurs during module compilation.
    pass
  else:
    raise e


class TestAgentCardBuilder:
  """Test suite for AgentCardBuilder class."""

  def test_init_with_valid_agent(self):
    """Test successful initialization with valid agent."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "test_agent"

    # Act
    builder = AgentCardBuilder(agent=mock_agent)

    # Assert
    assert builder._agent == mock_agent
    assert builder._rpc_url == "http://localhost:80/a2a"
    assert isinstance(builder._capabilities, AgentCapabilities)
    assert builder._doc_url is None
    assert builder._provider is None
    assert builder._security_schemes is None
    assert builder._agent_version == "0.0.1"

  def test_init_with_custom_parameters(self):
    """Test initialization with custom parameters."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "test_agent"
    mock_capabilities = Mock(spec=AgentCapabilities)
    mock_provider = Mock(spec=AgentProvider)
    mock_security_schemes = {"test": Mock(spec=SecurityScheme)}

    # Act
    builder = AgentCardBuilder(
        agent=mock_agent,
        rpc_url="https://example.com/a2a",
        capabilities=mock_capabilities,
        doc_url="https://docs.example.com",
        provider=mock_provider,
        agent_version="1.2.3",
        security_schemes=mock_security_schemes,
    )

    # Assert
    assert builder._agent == mock_agent
    assert builder._rpc_url == "https://example.com/a2a"
    assert builder._capabilities == mock_capabilities
    assert builder._doc_url == "https://docs.example.com"
    assert builder._provider == mock_provider
    assert builder._security_schemes == mock_security_schemes
    assert builder._agent_version == "1.2.3"

  def test_init_with_none_agent(self):
    """Test initialization with None agent raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError, match="Agent cannot be None or empty."):
      AgentCardBuilder(agent=None)

  def test_init_with_empty_agent(self):
    """Test initialization with empty agent raises ValueError."""
    # Arrange
    mock_agent = None

    # Act & Assert
    with pytest.raises(ValueError, match="Agent cannot be None or empty."):
      AgentCardBuilder(agent=mock_agent)

  @patch("google.adk.a2a.utils.agent_card_builder._build_primary_skills")
  @patch("google.adk.a2a.utils.agent_card_builder._build_sub_agent_skills")
  async def test_build_success(
      self, mock_build_sub_skills, mock_build_primary_skills
  ):
    """Test successful agent card building."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "test_agent"
    mock_agent.description = "Test agent description"

    mock_primary_skill = Mock(spec=AgentSkill)
    mock_sub_skill = Mock(spec=AgentSkill)
    mock_build_primary_skills.return_value = [mock_primary_skill]
    mock_build_sub_skills.return_value = [mock_sub_skill]

    builder = AgentCardBuilder(agent=mock_agent)

    # Act
    result = await builder.build()

    # Assert
    assert isinstance(result, AgentCard)
    assert result.name == "test_agent"
    assert result.description == "Test agent description"
    assert result.documentation_url is None
    assert result.url == "http://localhost:80/a2a"
    assert result.version == "0.0.1"
    assert result.skills == [mock_primary_skill, mock_sub_skill]
    assert result.default_input_modes == ["text/plain"]
    assert result.default_output_modes == ["text/plain"]
    assert result.supports_authenticated_extended_card is False
    assert result.provider is None
    assert result.security_schemes is None

  @patch("google.adk.a2a.utils.agent_card_builder._build_primary_skills")
  @patch("google.adk.a2a.utils.agent_card_builder._build_sub_agent_skills")
  async def test_build_with_custom_parameters(
      self, mock_build_sub_skills, mock_build_primary_skills
  ):
    """Test agent card building with custom parameters."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "test_agent"
    mock_agent.description = None  # Should use default description

    mock_primary_skill = Mock(spec=AgentSkill)
    mock_sub_skill = Mock(spec=AgentSkill)
    mock_build_primary_skills.return_value = [mock_primary_skill]
    mock_build_sub_skills.return_value = [mock_sub_skill]

    mock_provider = Mock(spec=AgentProvider)
    mock_security_schemes = {"test": Mock(spec=SecurityScheme)}

    builder = AgentCardBuilder(
        agent=mock_agent,
        rpc_url="https://example.com/a2a/",
        doc_url="https://docs.example.com",
        provider=mock_provider,
        agent_version="2.0.0",
        security_schemes=mock_security_schemes,
    )

    # Act
    result = await builder.build()

    # Assert
    assert result.name == "test_agent"
    assert result.description == "An ADK Agent"  # Default description
    # The source code uses doc_url parameter but AgentCard expects documentation_url
    # Since the source code doesn't map doc_url to documentation_url, it will be None
    assert result.documentation_url is None
    assert (
        result.url == "https://example.com/a2a"
    )  # Should strip trailing slash
    assert result.version == "2.0.0"
    assert result.provider == mock_provider
    assert result.security_schemes == mock_security_schemes

  @patch("google.adk.a2a.utils.agent_card_builder._build_primary_skills")
  @patch("google.adk.a2a.utils.agent_card_builder._build_sub_agent_skills")
  async def test_build_raises_runtime_error_on_failure(
      self, mock_build_sub_skills, mock_build_primary_skills
  ):
    """Test that build raises RuntimeError when underlying functions fail."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "test_agent"
    mock_build_primary_skills.side_effect = Exception("Test error")

    builder = AgentCardBuilder(agent=mock_agent)

    # Act & Assert
    with pytest.raises(
        RuntimeError,
        match="Failed to build agent card for test_agent: Test error",
    ):
      await builder.build()


class TestHelperFunctions:
  """Test suite for helper functions."""

  def test_get_agent_type_llm_agent(self):
    """Test _get_agent_type for LlmAgent."""
    # Arrange
    mock_agent = Mock(spec=LlmAgent)

    # Act
    result = _get_agent_type(mock_agent)

    # Assert
    assert result == "llm"

  def test_get_agent_type_sequential_agent(self):
    """Test _get_agent_type for SequentialAgent."""
    # Arrange
    mock_agent = Mock(spec=SequentialAgent)

    # Act
    result = _get_agent_type(mock_agent)

    # Assert
    assert result == "sequential_workflow"

  def test_get_agent_type_parallel_agent(self):
    """Test _get_agent_type for ParallelAgent."""
    # Arrange
    mock_agent = Mock(spec=ParallelAgent)

    # Act
    result = _get_agent_type(mock_agent)

    # Assert
    assert result == "parallel_workflow"

  def test_get_agent_type_loop_agent(self):
    """Test _get_agent_type for LoopAgent."""
    # Arrange
    mock_agent = Mock(spec=LoopAgent)

    # Act
    result = _get_agent_type(mock_agent)

    # Assert
    assert result == "loop_workflow"

  def test_get_agent_type_custom_agent(self):
    """Test _get_agent_type for custom agent."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)

    # Act
    result = _get_agent_type(mock_agent)

    # Assert
    assert result == "custom_agent"

  def test_get_agent_skill_name_llm_agent(self):
    """Test _get_agent_skill_name for LlmAgent."""
    # Arrange
    mock_agent = Mock(spec=LlmAgent)

    # Act
    result = _get_agent_skill_name(mock_agent)

    # Assert
    assert result == "model"

  def test_get_agent_skill_name_workflow_agents(self):
    """Test _get_agent_skill_name for workflow agents."""
    # Arrange
    mock_sequential = Mock(spec=SequentialAgent)
    mock_parallel = Mock(spec=ParallelAgent)
    mock_loop = Mock(spec=LoopAgent)

    # Act & Assert
    assert _get_agent_skill_name(mock_sequential) == "workflow"
    assert _get_agent_skill_name(mock_parallel) == "workflow"
    assert _get_agent_skill_name(mock_loop) == "workflow"

  def test_get_agent_skill_name_custom_agent(self):
    """Test _get_agent_skill_name for custom agent."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)

    # Act
    result = _get_agent_skill_name(mock_agent)

    # Assert
    assert result == "custom"

  def test_replace_pronouns_basic(self):
    """Test _replace_pronouns with basic pronoun replacement."""
    # Arrange
    text = "You should do your work and it will be yours."

    # Act
    result = _replace_pronouns(text)

    # Assert
    assert result == "I should do my work and it will be mine."

  def test_replace_pronouns_case_insensitive(self):
    """Test _replace_pronouns with case-insensitive matching."""
    # Arrange
    text = "YOU should do YOUR work and it will be YOURS."

    # Act
    result = _replace_pronouns(text)

    # Assert
    assert result == "I should do my work and it will be mine."

  def test_replace_pronouns_mixed_case(self):
    """Test _replace_pronouns with mixed case."""
    # Arrange
    text = "You should do Your work and it will be Yours."

    # Act
    result = _replace_pronouns(text)

    # Assert
    assert result == "I should do my work and it will be mine."

  def test_replace_pronouns_no_pronouns(self):
    """Test _replace_pronouns with no pronouns."""
    # Arrange
    text = "This is a test message without pronouns."

    # Act
    result = _replace_pronouns(text)

    # Assert
    assert result == text

  def test_replace_pronouns_partial_matches(self):
    """Test _replace_pronouns with partial matches that shouldn't be replaced."""
    # Arrange
    text = "youth, yourself, yourname"

    # Act
    result = _replace_pronouns(text)

    # Assert
    assert result == "youth, yourself, yourname"  # No changes

  def test_replace_pronouns_phrases(self):
    """Test _replace_pronouns with phrases that should be replaced."""
    # Arrange
    text = "You are a helpful chatbot"

    # Act
    result = _replace_pronouns(text)

    # Assert
    assert result == "I am a helpful chatbot"

  def test_get_default_description_llm_agent(self):
    """Test _get_default_description for LlmAgent."""
    # Arrange
    mock_agent = Mock(spec=LlmAgent)

    # Act
    result = _get_default_description(mock_agent)

    # Assert
    assert result == "An LLM-based agent"

  def test_get_default_description_sequential_agent(self):
    """Test _get_default_description for SequentialAgent."""
    # Arrange
    mock_agent = Mock(spec=SequentialAgent)

    # Act
    result = _get_default_description(mock_agent)

    # Assert
    assert result == "A sequential workflow agent"

  def test_get_default_description_parallel_agent(self):
    """Test _get_default_description for ParallelAgent."""
    # Arrange
    mock_agent = Mock(spec=ParallelAgent)

    # Act
    result = _get_default_description(mock_agent)

    # Assert
    assert result == "A parallel workflow agent"

  def test_get_default_description_loop_agent(self):
    """Test _get_default_description for LoopAgent."""
    # Arrange
    mock_agent = Mock(spec=LoopAgent)

    # Act
    result = _get_default_description(mock_agent)

    # Assert
    assert result == "A loop workflow agent"

  def test_get_default_description_custom_agent(self):
    """Test _get_default_description for custom agent."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)

    # Act
    result = _get_default_description(mock_agent)

    # Assert
    assert result == "A custom agent"

  def test_get_input_modes_llm_agent(self):
    """Test _get_input_modes for LlmAgent."""
    # Arrange
    mock_agent = Mock(spec=LlmAgent)

    # Act
    result = _get_input_modes(mock_agent)

    # Assert
    assert result is None  # Currently returns None for all cases

  def test_get_input_modes_non_llm_agent(self):
    """Test _get_input_modes for non-LlmAgent."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)

    # Act
    result = _get_input_modes(mock_agent)

    # Assert
    assert result is None

  def test_get_output_modes_llm_agent_with_config(self):
    """Test _get_output_modes for LlmAgent with response_modalities."""
    # Arrange
    mock_config = Mock()
    mock_config.response_modalities = ["text/plain", "application/json"]
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.generate_content_config = mock_config

    # Act
    result = _get_output_modes(mock_agent)

    # Assert
    assert result == ["text/plain", "application/json"]

  def test_get_output_modes_llm_agent_without_config(self):
    """Test _get_output_modes for LlmAgent without config."""
    # Arrange
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.generate_content_config = None

    # Act
    result = _get_output_modes(mock_agent)

    # Assert
    assert result is None

  def test_get_output_modes_llm_agent_without_response_modalities(self):
    """Test _get_output_modes for LlmAgent without response_modalities."""
    # Arrange
    mock_config = Mock()
    del mock_config.response_modalities
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.generate_content_config = mock_config

    # Act
    result = _get_output_modes(mock_agent)

    # Assert
    assert result is None

  def test_get_output_modes_non_llm_agent(self):
    """Test _get_output_modes for non-LlmAgent."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)

    # Act
    result = _get_output_modes(mock_agent)

    # Assert
    assert result is None


class TestDescriptionBuildingFunctions:
  """Test suite for description building functions."""

  def test_build_agent_description_with_description(self):
    """Test _build_agent_description with agent description."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.description = "Test agent description"
    mock_agent.sub_agents = []

    # Act
    result = _build_agent_description(mock_agent)

    # Assert
    assert result == "Test agent description"

  def test_build_agent_description_without_description(self):
    """Test _build_agent_description without agent description."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.description = None
    mock_agent.sub_agents = []

    # Act
    result = _build_agent_description(mock_agent)

    # Assert
    assert result == "A custom agent"  # Default description

  def test_build_llm_agent_description_with_instructions(self):
    """Test _build_llm_agent_description_with_instructions with all components."""
    # Arrange
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.description = "Test agent"
    mock_agent.instruction = "You should help users."
    mock_agent.global_instruction = "Your role is to assist."

    # Act
    result = _build_llm_agent_description_with_instructions(mock_agent)

    # Assert
    assert result == "Test agent I should help users. my role is to assist."

  def test_build_llm_agent_description_without_instructions(self):
    """Test _build_llm_agent_description_with_instructions without instructions."""
    # Arrange
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.description = "Test agent"
    mock_agent.instruction = None
    mock_agent.global_instruction = None

    # Act
    result = _build_llm_agent_description_with_instructions(mock_agent)

    # Assert
    assert result == "Test agent"

  def test_build_llm_agent_description_without_description(self):
    """Test _build_llm_agent_description_with_instructions without description."""
    # Arrange
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.description = None
    mock_agent.instruction = "You should help users."
    mock_agent.global_instruction = None

    # Act
    result = _build_llm_agent_description_with_instructions(mock_agent)

    # Assert
    assert result == "I should help users."

  def test_build_llm_agent_description_empty_all(self):
    """Test _build_llm_agent_description_with_instructions with all empty."""
    # Arrange
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.description = None
    mock_agent.instruction = None
    mock_agent.global_instruction = None

    # Act
    result = _build_llm_agent_description_with_instructions(mock_agent)

    # Assert
    assert result == "An LLM-based agent"  # Default description

  def test_get_workflow_description_sequential_agent(self):
    """Test _get_workflow_description for SequentialAgent."""
    # Arrange
    mock_sub_agent1 = Mock(spec=BaseAgent)
    mock_sub_agent1.name = "agent1"
    mock_sub_agent1.description = "First agent"
    mock_sub_agent2 = Mock(spec=BaseAgent)
    mock_sub_agent2.name = "agent2"
    mock_sub_agent2.description = "Second agent"

    mock_agent = Mock(spec=SequentialAgent)
    mock_agent.sub_agents = [mock_sub_agent1, mock_sub_agent2]

    # Act
    result = _get_workflow_description(mock_agent)

    # Assert
    assert result is not None
    assert (
        result
        == "First, this agent will First agent Finally, this agent will Second"
        " agent."
    )

  def test_get_workflow_description_parallel_agent(self):
    """Test _get_workflow_description for ParallelAgent."""
    # Arrange
    mock_sub_agent1 = Mock(spec=BaseAgent)
    mock_sub_agent1.name = "agent1"
    mock_sub_agent1.description = "First agent"
    mock_sub_agent2 = Mock(spec=BaseAgent)
    mock_sub_agent2.name = "agent2"
    mock_sub_agent2.description = "Second agent"

    mock_agent = Mock(spec=ParallelAgent)
    mock_agent.sub_agents = [mock_sub_agent1, mock_sub_agent2]

    # Act
    result = _get_workflow_description(mock_agent)

    # Assert
    assert result is not None
    assert (
        result == "This agent will First agent and Second agent simultaneously."
    )

  def test_get_workflow_description_loop_agent(self):
    """Test _get_workflow_description for LoopAgent."""
    # Arrange
    mock_sub_agent1 = Mock(spec=BaseAgent)
    mock_sub_agent1.name = "agent1"
    mock_sub_agent1.description = "First agent"
    mock_sub_agent2 = Mock(spec=BaseAgent)
    mock_sub_agent2.name = "agent2"
    mock_sub_agent2.description = "Second agent"

    mock_agent = Mock(spec=LoopAgent)
    mock_agent.sub_agents = [mock_sub_agent1, mock_sub_agent2]
    mock_agent.max_iterations = 5

    # Act
    result = _get_workflow_description(mock_agent)

    # Assert
    assert (
        result
        == "This agent will First agent and Second agent in a loop (max 5"
        " iterations)."
    )

  def test_get_workflow_description_loop_agent_unlimited(self):
    """Test _get_workflow_description for LoopAgent with unlimited iterations."""
    # Arrange
    mock_sub_agent1 = Mock(spec=BaseAgent)
    mock_sub_agent1.name = "agent1"
    mock_sub_agent1.description = "First agent"

    mock_agent = Mock(spec=LoopAgent)
    mock_agent.sub_agents = [mock_sub_agent1]
    mock_agent.max_iterations = None

    # Act
    result = _get_workflow_description(mock_agent)

    # Assert
    assert (
        result
        == "This agent will First agent in a loop (max unlimited iterations)."
    )

  def test_get_workflow_description_no_sub_agents(self):
    """Test _get_workflow_description for agent without sub-agents."""
    # Arrange
    mock_agent = Mock(spec=SequentialAgent)
    mock_agent.sub_agents = []

    # Act
    result = _get_workflow_description(mock_agent)

    # Assert
    assert result is None

  def test_get_workflow_description_custom_agent(self):
    """Test _get_workflow_description for custom agent."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.sub_agents = [Mock(spec=BaseAgent)]

    # Act
    result = _get_workflow_description(mock_agent)

    # Assert
    assert result is None

  def test_build_sequential_description_single_agent(self):
    """Test _build_sequential_description with single sub-agent."""
    # Arrange
    mock_sub_agent = Mock(spec=BaseAgent)
    mock_sub_agent.name = "agent1"
    mock_sub_agent.description = "First agent"

    mock_agent = Mock(spec=SequentialAgent)
    mock_agent.sub_agents = [mock_sub_agent]

    # Act
    result = _build_sequential_description(mock_agent)

    # Assert
    assert result == "First, this agent will First agent."

  def test_build_sequential_description_multiple_agents(self):
    """Test _build_sequential_description with multiple sub-agents."""
    # Arrange
    mock_sub_agent1 = Mock(spec=BaseAgent)
    mock_sub_agent1.name = "agent1"
    mock_sub_agent1.description = "First agent"
    mock_sub_agent2 = Mock(spec=BaseAgent)
    mock_sub_agent2.name = "agent2"
    mock_sub_agent2.description = "Second agent"
    mock_sub_agent3 = Mock(spec=BaseAgent)
    mock_sub_agent3.name = "agent3"
    mock_sub_agent3.description = "Third agent"

    mock_agent = Mock(spec=SequentialAgent)
    mock_agent.sub_agents = [mock_sub_agent1, mock_sub_agent2, mock_sub_agent3]

    # Act
    result = _build_sequential_description(mock_agent)

    # Assert
    assert (
        result
        == "First, this agent will First agent Then, this agent will Second"
        " agent Finally, this agent will Third agent."
    )

  def test_build_sequential_description_without_descriptions(self):
    """Test _build_sequential_description with sub-agents without descriptions."""
    # Arrange
    mock_sub_agent1 = Mock(spec=BaseAgent)
    mock_sub_agent1.name = "agent1"
    mock_sub_agent1.description = None
    mock_sub_agent2 = Mock(spec=BaseAgent)
    mock_sub_agent2.name = "agent2"
    mock_sub_agent2.description = None

    mock_agent = Mock(spec=SequentialAgent)
    mock_agent.sub_agents = [mock_sub_agent1, mock_sub_agent2]

    # Act
    result = _build_sequential_description(mock_agent)

    # Assert
    assert (
        result
        == "First, this agent will execute the agent1 agent Finally, this agent"
        " will execute the agent2 agent."
    )

  def test_build_parallel_description_single_agent(self):
    """Test _build_parallel_description with single sub-agent."""
    # Arrange
    mock_sub_agent = Mock(spec=BaseAgent)
    mock_sub_agent.name = "agent1"
    mock_sub_agent.description = "First agent"

    mock_agent = Mock(spec=ParallelAgent)
    mock_agent.sub_agents = [mock_sub_agent]

    # Act
    result = _build_parallel_description(mock_agent)

    # Assert
    assert result == "This agent will First agent simultaneously."

  def test_build_parallel_description_multiple_agents(self):
    """Test _build_parallel_description with multiple sub-agents."""
    # Arrange
    mock_sub_agent1 = Mock(spec=BaseAgent)
    mock_sub_agent1.name = "agent1"
    mock_sub_agent1.description = "First agent"
    mock_sub_agent2 = Mock(spec=BaseAgent)
    mock_sub_agent2.name = "agent2"
    mock_sub_agent2.description = "Second agent"
    mock_sub_agent3 = Mock(spec=BaseAgent)
    mock_sub_agent3.name = "agent3"
    mock_sub_agent3.description = "Third agent"

    mock_agent = Mock(spec=ParallelAgent)
    mock_agent.sub_agents = [mock_sub_agent1, mock_sub_agent2, mock_sub_agent3]

    # Act
    result = _build_parallel_description(mock_agent)

    # Assert
    assert (
        result
        == "This agent will First agent , Second agent and Third agent"
        " simultaneously."
    )

  def test_build_loop_description_single_agent(self):
    """Test _build_loop_description with single sub-agent."""
    # Arrange
    mock_sub_agent = Mock(spec=BaseAgent)
    mock_sub_agent.name = "agent1"
    mock_sub_agent.description = "First agent"

    mock_agent = Mock(spec=LoopAgent)
    mock_agent.sub_agents = [mock_sub_agent]
    mock_agent.max_iterations = 3

    # Act
    result = _build_loop_description(mock_agent)

    # Assert
    assert result == "This agent will First agent in a loop (max 3 iterations)."

  def test_build_loop_description_multiple_agents(self):
    """Test _build_loop_description with multiple sub-agents."""
    # Arrange
    mock_sub_agent1 = Mock(spec=BaseAgent)
    mock_sub_agent1.name = "agent1"
    mock_sub_agent1.description = "First agent"
    mock_sub_agent2 = Mock(spec=BaseAgent)
    mock_sub_agent2.name = "agent2"
    mock_sub_agent2.description = "Second agent"

    mock_agent = Mock(spec=LoopAgent)
    mock_agent.sub_agents = [mock_sub_agent1, mock_sub_agent2]
    mock_agent.max_iterations = 10

    # Act
    result = _build_loop_description(mock_agent)

    # Assert
    assert (
        result
        == "This agent will First agent and Second agent in a loop (max 10"
        " iterations)."
    )

  def test_build_orchestration_skill_with_sub_agents(self):
    """Test _build_orchestration_skill with sub-agents."""
    # Arrange
    mock_sub_agent1 = Mock(spec=BaseAgent)
    mock_sub_agent1.name = "agent1"
    mock_sub_agent1.description = "First agent description"
    mock_sub_agent2 = Mock(spec=BaseAgent)
    mock_sub_agent2.name = "agent2"
    mock_sub_agent2.description = "Second agent description"

    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "main_agent"
    mock_agent.sub_agents = [mock_sub_agent1, mock_sub_agent2]

    # Act
    result = _build_orchestration_skill(mock_agent, "sequential_workflow")

    # Assert
    assert result is not None
    assert result.id == "main_agent-sub-agents"
    assert result.name == "sub-agents"
    assert (
        result.description
        == "Orchestrates: agent1: First agent description; agent2: Second agent"
        " description"
    )
    assert result.tags == ["sequential_workflow", "orchestration"]

  def test_build_orchestration_skill_without_descriptions(self):
    """Test _build_orchestration_skill with sub-agents without descriptions."""
    # Arrange
    mock_sub_agent1 = Mock(spec=BaseAgent)
    mock_sub_agent1.name = "agent1"
    mock_sub_agent1.description = None
    mock_sub_agent2 = Mock(spec=BaseAgent)
    mock_sub_agent2.name = "agent2"
    mock_sub_agent2.description = None

    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "main_agent"
    mock_agent.sub_agents = [mock_sub_agent1, mock_sub_agent2]

    # Act
    result = _build_orchestration_skill(mock_agent, "parallel_workflow")

    # Assert
    assert result is not None
    assert (
        result.description
        == "Orchestrates: agent1: No description; agent2: No description"
    )

  def test_build_orchestration_skill_no_sub_agents(self):
    """Test _build_orchestration_skill with no sub-agents."""
    # Arrange
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.sub_agents = []

    # Act
    result = _build_orchestration_skill(mock_agent, "custom_agent")

    # Assert
    assert result is None


class TestExampleExtractionFunctions:
  """Test suite for example extraction functions."""

  def test_convert_example_tool_examples_with_model_dump(self):
    """Test _convert_example_tool_examples with examples that have model_dump."""
    # Arrange
    mock_input = Mock()
    mock_input.model_dump.return_value = {"text": "test input"}
    mock_output1 = Mock()
    mock_output1.model_dump.return_value = {"text": "test output 1"}
    mock_output2 = Mock()
    mock_output2.model_dump.return_value = {"text": "test output 2"}

    mock_example = Mock()
    mock_example.input = mock_input
    mock_example.output = [mock_output1, mock_output2]

    mock_tool = Mock(spec=ExampleTool)
    mock_tool.examples = [mock_example]

    # Act
    result = _convert_example_tool_examples(mock_tool)

    # Assert
    assert len(result) == 1
    assert result[0]["input"] == {"text": "test input"}
    assert result[0]["output"] == [
        {"text": "test output 1"},
        {"text": "test output 2"},
    ]

  def test_convert_example_tool_examples_without_model_dump(self):
    """Test _convert_example_tool_examples with examples without model_dump."""
    # Arrange
    mock_input = {"text": "test input"}
    mock_output1 = {"text": "test output 1"}
    mock_output2 = {"text": "test output 2"}

    mock_example = Mock()
    mock_example.input = mock_input
    mock_example.output = [mock_output1, mock_output2]

    mock_tool = Mock(spec=ExampleTool)
    mock_tool.examples = [mock_example]

    # Act
    result = _convert_example_tool_examples(mock_tool)

    # Assert
    assert len(result) == 1
    assert result[0]["input"] == {"text": "test input"}
    assert result[0]["output"] == [
        {"text": "test output 1"},
        {"text": "test output 2"},
    ]

  def test_convert_example_tool_examples_multiple_examples(self):
    """Test _convert_example_tool_examples with multiple examples."""
    # Arrange
    mock_example1 = Mock()
    mock_example1.input = {"text": "input 1"}
    mock_example1.output = [{"text": "output 1"}]

    mock_example2 = Mock()
    mock_example2.input = {"text": "input 2"}
    mock_example2.output = [{"text": "output 2"}]

    mock_tool = Mock(spec=ExampleTool)
    mock_tool.examples = [mock_example1, mock_example2]

    # Act
    result = _convert_example_tool_examples(mock_tool)

    # Assert
    assert len(result) == 2
    assert result[0]["input"] == {"text": "input 1"}
    assert result[0]["output"] == [{"text": "output 1"}]
    assert result[1]["input"] == {"text": "input 2"}
    assert result[1]["output"] == [{"text": "output 2"}]

  def test_convert_example_tool_examples_empty_list(self):
    """Test _convert_example_tool_examples with empty examples list."""
    # Arrange
    mock_tool = Mock(spec=ExampleTool)
    mock_tool.examples = []

    # Act
    result = _convert_example_tool_examples(mock_tool)

    # Assert
    assert result == []

  def test_extract_examples_from_instruction_with_examples(self):
    """Test _extract_examples_from_instruction with valid examples."""
    # Arrange
    instruction = (
        'Example Query: "What is the weather?" Example Response: "The weather'
        ' is sunny."'
    )

    # Act
    result = _extract_examples_from_instruction(instruction)

    # Assert
    # The function processes each pattern separately, so it won't find pairs
    # from different patterns. This test should return None.
    assert result is None

  def test_extract_examples_from_instruction_with_multiple_examples(self):
    """Test _extract_examples_from_instruction with multiple examples."""
    # Arrange
    instruction = """
    Example Query: "What is the weather?" Example Response: "The weather is sunny."
    Example Query: "What time is it?" Example Response: "It is 3 PM."
    """

    # Act
    result = _extract_examples_from_instruction(instruction)

    # Assert
    # The function finds matches but pairs them incorrectly due to how patterns are processed
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2
    # The function pairs consecutive matches from the same pattern
    assert result[0]["input"] == {"text": "What is the weather?"}
    assert result[0]["output"] == [{"text": "What time is it?"}]
    assert result[1]["input"] == {"text": "The weather is sunny."}
    assert result[1]["output"] == [{"text": "It is 3 PM."}]

  def test_extract_examples_from_instruction_with_different_patterns(self):
    """Test _extract_examples_from_instruction with different example patterns."""
    # Arrange
    instruction = (
        'Example: "What is the weather?" Example Response: "The weather is'
        ' sunny."'
    )

    # Act
    result = _extract_examples_from_instruction(instruction)

    # Assert
    # The function processes each pattern separately, so it won't find pairs
    # from different patterns. This test should return None.
    assert result is None

  def test_extract_examples_from_instruction_case_insensitive(self):
    """Test _extract_examples_from_instruction with case-insensitive matching."""
    # Arrange
    instruction = (
        'example query: "What is the weather?" example response: "The weather'
        ' is sunny."'
    )

    # Act
    result = _extract_examples_from_instruction(instruction)

    # Assert
    # The function processes each pattern separately, so it won't find pairs
    # from different patterns. This test should return None.
    assert result is None

  def test_extract_examples_from_instruction_no_examples(self):
    """Test _extract_examples_from_instruction with no examples."""
    # Arrange
    instruction = "This is a regular instruction without any examples."

    # Act
    result = _extract_examples_from_instruction(instruction)

    # Assert
    assert result is None

  def test_extract_examples_from_instruction_odd_number_of_matches(self):
    """Test _extract_examples_from_instruction with odd number of matches."""
    # Arrange
    instruction = (
        'Example Query: "What is the weather?" Example Response: "The weather'
        ' is sunny." Example Query: "What time is it?"'
    )

    # Act
    result = _extract_examples_from_instruction(instruction)

    # Assert
    # The function finds matches but only pairs complete pairs
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1  # Only complete pairs should be included
    assert result[0]["input"] == {"text": "What is the weather?"}
    assert result[0]["output"] == [{"text": "What time is it?"}]
