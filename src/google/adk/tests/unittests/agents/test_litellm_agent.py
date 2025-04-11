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

"""Unit tests for LiteLlmAgent."""

from typing import Optional
from unittest import mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.litellm_agent import LiteLlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import BaseModel
import pytest


@pytest.fixture
def mock_litellm():
    """Create a mocked LiteLlm class."""
    with mock.patch('google.adk.agents.litellm_agent.LiteLlm') as mock_litellm_cls:
        mock_instance = mock.MagicMock()
        mock_litellm_cls.return_value = mock_instance
        yield mock_litellm_cls


@pytest.fixture
def mock_litellm_with_error():
    """Create a mocked LiteLlm class that raises an error."""
    with mock.patch('google.adk.agents.litellm_agent.LiteLlm') as mock_litellm_cls:
        mock_litellm_cls.side_effect = ValueError("Invalid model configuration")
        yield mock_litellm_cls


def _create_readonly_context(
    agent: LiteLlmAgent, state: Optional[dict] = None
) -> ReadonlyContext:
    """Helper function to create a readonly context for testing."""
    session_service = InMemorySessionService()
    state = state or {}
    session = session_service.create_session(
        app_name='test_app', user_id='test_user', state=state
    )
    invocation_context = InvocationContext(
        invocation_id='test_id',
        agent=agent,
        session=session,
        session_service=session_service,
    )
    return ReadonlyContext(invocation_context)


class TestLiteLlmAgentBasics:
    """Basic tests for LiteLlmAgent initialization and properties."""

    def test_initialization_with_string(self, mock_litellm):
        """Test initializing LiteLlmAgent with a model name string."""
        agent = LiteLlmAgent(
            name="test_agent",
            model="ollama/gemma3:12b",
            description="Test agent description"
        )
        
        # Verify LiteLlm was properly instantiated
        mock_litellm.assert_called_once_with(model="ollama/gemma3:12b")
        
        # Verify agent properties
        assert agent.name == "test_agent"
        assert agent.description == "Test agent description"
    
    def test_initialization_with_instance(self):
        """Test initializing LiteLlmAgent with a LiteLlm instance."""
        # Create a real LiteLlm instance but with mocked internals
        with mock.patch.object(LiteLlm, '__init__', return_value=None) as mock_init:
            lite_llm_instance = LiteLlm.__new__(LiteLlm)
            lite_llm_instance.model = "custom_model"
            
            agent = LiteLlmAgent(
                name="test_agent",
                model=lite_llm_instance,
                description="Test agent description"
            )
            
            # Verify LiteLlm wasn't instantiated again
            mock_init.assert_not_called()
            
            # Verify agent properties
            assert agent.name == "test_agent"
            assert agent.model is lite_llm_instance
            assert agent.model.model == "custom_model"
            assert agent.description == "Test agent description"
    
    def test_initialization_with_invalid_model_type(self):
        """Test initializing LiteLlmAgent with an invalid model type."""
        with pytest.raises(TypeError, match="Expected model to be a string or LiteLlm instance"):
            LiteLlmAgent(
                name="test_agent",
                model=123  # Invalid model type
            )
    
    def test_initialization_with_model_error(self, mock_litellm_with_error):
        """Test initializing LiteLlmAgent when LiteLlm initialization fails."""
        with pytest.raises(ValueError, match="Failed to initialize LiteLlm with model 'invalid/model'"):
            LiteLlmAgent(
                name="test_agent",
                model="invalid/model"
            )
    
    def test_repr(self, mock_litellm):
        """Test the __repr__ method of LiteLlmAgent."""
        agent = LiteLlmAgent(
            name="test_agent",
            model="ollama/gemma3:12b"
        )
        
        # Mock the model class name
        agent.model.__class__.__name__ = "LiteLlm"
        
        assert repr(agent) == "LiteLlmAgent(name='test_agent', model=LiteLlm)"


class TestLiteLlmAgentFeatures:
    """Tests for LiteLlmAgent features and configurations."""

    def test_model_inheritance(self, mock_litellm):
        """Test that child agents inherit the model from parent agent."""
        parent_agent = LiteLlmAgent(
            name="parent_agent",
            model="ollama/gemma3:12b"
        )
        
        # Child agent doesn't specify a model
        child_agent = LiteLlmAgent(
            name="child_agent",
            sub_agents=[]
        )
        
        # Manually set parent-child relationship
        child_agent.parent_agent = parent_agent
        
        # Mock canonical_model behavior
        with mock.patch('google.adk.agents.llm_agent.LlmAgent.canonical_model',
                       new_callable=mock.PropertyMock) as mock_canonical_model:
            mock_canonical_model.return_value = parent_agent.model
            # Child should inherit model from parent
            assert child_agent.canonical_model is parent_agent.model
    
    def test_agent_with_callbacks(self, mock_litellm):
        """Test LiteLlmAgent initialization with callbacks."""
        def before_model_callback(
            callback_context: CallbackContext,
            llm_request: LlmRequest,
        ) -> Optional[LlmResponse]:
            return None
        
        def after_model_callback(
            callback_context: CallbackContext,
            llm_response: LlmResponse,
        ) -> Optional[LlmResponse]:
            return None
        
        agent = LiteLlmAgent(
            name="test_agent",
            model="ollama/gemma3:12b",
            before_model_callback=before_model_callback,
            after_model_callback=after_model_callback
        )
        
        assert agent.before_model_callback is before_model_callback
        assert agent.after_model_callback is after_model_callback
    
    def test_agent_with_tools(self, mock_litellm):
        """Test LiteLlmAgent initialization with tools."""
        def test_tool():
            """A simple test tool."""
            return "test result"
        
        agent = LiteLlmAgent(
            name="test_agent",
            model="ollama/gemma3:12b",
            tools=[test_tool]
        )
        
        assert len(agent.tools) == 1
        assert agent.tools[0] == test_tool
    
    def test_agent_with_configuration(self, mock_litellm):
        """Test LiteLlmAgent with various configuration options."""
        agent = LiteLlmAgent(
            name="test_agent",
            model="ollama/gemma3:12b",
            description="Test description",
            instruction="Test instruction",
            global_instruction="Global instruction",
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True,
            include_contents="none",
            generate_content_config=types.GenerateContentConfig(temperature=0.7)
        )
        
        assert agent.name == "test_agent"
        assert agent.description == "Test description"
        assert agent.instruction == "Test instruction"
        assert agent.global_instruction == "Global instruction"
        assert agent.disallow_transfer_to_parent is True
        assert agent.disallow_transfer_to_peers is True
        assert agent.include_contents == "none"
        assert agent.generate_content_config.temperature == 0.7


class TestLiteLlmAgentAdvanced:
    """Advanced tests for LiteLlmAgent functionality."""

    def test_update_model_parameters(self):
        """Test updating model parameters at runtime."""
        with mock.patch.object(LiteLlm, '__init__', return_value=None) as _:
            lite_llm_instance = LiteLlm.__new__(LiteLlm)
            lite_llm_instance._additional_args = {"api_key": "test_key"}
            
            agent = LiteLlmAgent(
                name="test_agent",
                model=lite_llm_instance
            )
            
            agent.update_model_parameters(temperature=0.8, max_tokens=100)
            
            assert agent.model._additional_args == {
                "api_key": "test_key",
                "temperature": 0.8,
                "max_tokens": 100
            }
    
    def test_update_model_parameters_with_non_litellm_model(self, mock_litellm):
        """Test updating model parameters when model is not a LiteLlm instance."""
        agent = LiteLlmAgent(
            name="test_agent",
            model="ollama/gemma3:12b"
        )
        
        # Replace the model with something that's not a LiteLlm instance
        agent.model = "not_a_litellm_instance"
        
        with pytest.raises(TypeError, match="Agent's model is not a LiteLlm instance"):
            agent.update_model_parameters(temperature=0.8)
    
    def test_with_output_schema(self, mock_litellm):
        """Test LiteLlmAgent with output schema."""
        class TestSchema(BaseModel):
            result: str
            score: int
        
        agent = LiteLlmAgent(
            name="test_agent",
            model="ollama/gemma3:12b",
            output_schema=TestSchema
        )
        
        assert agent.output_schema == TestSchema
        # LlmAgent disables transfers when output_schema is set
        assert agent.disallow_transfer_to_parent is True
        assert agent.disallow_transfer_to_peers is True
    
    @pytest.mark.parametrize(
        "model_name", [
            "ollama/gemma3:12b",
            "anthropic/claude-3-sonnet",
            "openai/gpt-4-turbo",
        ]
    )
    def test_with_different_model_providers(self, mock_litellm, model_name):
        """Test LiteLlmAgent with different model providers."""
        agent = LiteLlmAgent(
            name="test_agent",
            model=model_name
        )
        
        mock_litellm.assert_called_once_with(model=model_name)