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

"""Tests for the Data Analysis Agent."""

import unittest
from unittest import mock

from google.adk.agents.templates.data_analysis import DataAnalysisAgent
from google.adk.agents.templates.data_analysis import DataAnalysisToolset
from google.adk.memory import MemoryStore


class DataAnalysisAgentTest(unittest.TestCase):
  """Tests for the Data Analysis Agent."""
  
  def test_init_with_defaults(self):
    """Test initialization with default values."""
    agent = DataAnalysisAgent(
        name="test_agent",
        model="gemini-2.0-flash",
    )
    
    self.assertEqual(agent.name, "test_agent")
    self.assertEqual(agent.model, "gemini-2.0-flash")
    self.assertEqual(agent.data_sources, [])
    self.assertEqual(agent.analysis_types, [])
    self.assertEqual(agent.visualization_types, [])
    self.assertIsNone(agent.memory_store)
  
  def test_init_with_custom_values(self):
    """Test initialization with custom values."""
    memory_store = MemoryStore()
    data_sources = ["csv", "database"]
    analysis_types = ["summary", "correlation"]
    visualization_types = ["bar", "line"]
    
    agent = DataAnalysisAgent(
        name="custom_agent",
        model="gemini-2.0-flash",
        description="Custom description",
        data_sources=data_sources,
        analysis_types=analysis_types,
        visualization_types=visualization_types,
        memory_store=memory_store,
    )
    
    self.assertEqual(agent.name, "custom_agent")
    self.assertEqual(agent.description, "Custom description")
    self.assertEqual(agent.data_sources, data_sources)
    self.assertEqual(agent.analysis_types, analysis_types)
    self.assertEqual(agent.visualization_types, visualization_types)
    self.assertEqual(agent.memory_store, memory_store)
  
  def test_default_instruction_creation(self):
    """Test the creation of default instructions."""
    data_sources = ["csv", "database"]
    analysis_types = ["summary", "correlation"]
    visualization_types = ["bar", "line"]
    
    agent = DataAnalysisAgent(
        name="instruction_test",
        model="gemini-2.0-flash",
        data_sources=data_sources,
        analysis_types=analysis_types,
        visualization_types=visualization_types,
    )
    
    # Check that the instruction contains the expected information
    instruction = agent.instruction
    self.assertIn("data analysis assistant", instruction)
    self.assertIn("Available data sources:", instruction)
    self.assertIn("csv", instruction)
    self.assertIn("database", instruction)
    self.assertIn("Available analysis types:", instruction)
    self.assertIn("summary", instruction)
    self.assertIn("correlation", instruction)
    self.assertIn("Available visualization types:", instruction)
    self.assertIn("bar", instruction)
    self.assertIn("line", instruction)
  
  def test_custom_instruction(self):
    """Test providing a custom instruction."""
    custom_instruction = "This is a custom instruction."
    
    agent = DataAnalysisAgent(
        name="custom_instruction_test",
        model="gemini-2.0-flash",
        instruction=custom_instruction,
    )
    
    self.assertEqual(agent.instruction, custom_instruction)
  
  def test_tools_initialization(self):
    """Test that tools are properly initialized."""
    agent = DataAnalysisAgent(
        name="tools_test",
        model="gemini-2.0-flash",
    )
    
    # Check that the agent has the DataAnalysisToolset
    toolsets = [tool for tool in agent.tools if isinstance(tool, DataAnalysisToolset)]
    self.assertEqual(len(toolsets), 1)
    
    # Check that additional tools can be added
    mock_tool = mock.MagicMock()
    agent = DataAnalysisAgent(
        name="tools_test",
        model="gemini-2.0-flash",
        tools=[mock_tool],
    )
    
    # Should have both the DataAnalysisToolset and the mock tool
    self.assertEqual(len(agent.tools), 2)


if __name__ == "__main__":
  unittest.main()

