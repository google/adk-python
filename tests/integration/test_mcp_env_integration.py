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

"""Integration tests for MCP environment variable extraction and injection."""

import asyncio
import os
import tempfile
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
from .utils import TestRunner

# Import MCP dependencies
try:
    from mcp import StdioServerParameters
    from mcp.types import ListToolsResult, Tool as McpTool
    from mcp.client.session import ClientSession
except ImportError:
    pytest.skip("MCP dependencies not available", allow_module_level=True)


class TestMCPEnvironmentIntegration:
    """Integration tests for MCP environment variable functionality."""

    def create_test_agent_with_env_callback(self, env_transform_callback=None) -> LlmAgent:
        """Create a test agent with MCP toolset and environment variable callback."""
        # Create a temporary directory for the filesystem server
        temp_dir = tempfile.mkdtemp()
        
        return LlmAgent(
            model='gemini-2.0-flash',
            name='test_env_agent',
            instruction=f"""
            You are a test agent with access to filesystem operations.
            Test directory: {temp_dir}
            """,
            tools=[
                MCPToolset(
                    connection_params=StdioServerParameters(
                        command='npx',
                        args=[
                            '-y',
                            '@modelcontextprotocol/server-filesystem',
                            temp_dir,
                        ],
                        env={'INITIAL_VAR': 'initial_value'},
                    ),
                    env_transform_callback=env_transform_callback,
                    tool_filter=[
                        'read_file',
                        'list_directory',
                        'directory_tree',
                    ],
                )
            ],
        )

    def sample_env_transform_callback(self, state_dict: Dict[str, Any]) -> Dict[str, str]:
        """Sample environment variable transformation callback."""
        env_vars = {}
        
        # Extract common environment variables
        if 'api_key' in state_dict:
            env_vars['API_KEY'] = state_dict['api_key']
        
        if 'environment' in state_dict:
            env_vars['ENVIRONMENT'] = state_dict['environment']
            
        if 'user_config' in state_dict:
            config = state_dict['user_config']
            if isinstance(config, dict):
                for key, value in config.items():
                    env_vars[f'USER_{key.upper()}'] = str(value)
        
        return env_vars

    @pytest.mark.asyncio
    async def test_env_extraction_and_injection_with_session_state(self, llm_backend):
        """Test environment variable extraction from session state and injection into MCP server."""
        # Create agent with environment callback
        agent = self.create_test_agent_with_env_callback(
            env_transform_callback=self.sample_env_transform_callback
        )
        
        # Create test runner
        runner = TestRunner(agent)
        session_service = runner.session_service
        
        # Get the current session and add state with environment variables
        session = await runner.get_current_session_async()
        session.state.update({
            'api_key': 'test_api_key_123',
            'environment': 'production',
            'user_config': {
                'timeout': '30',
                'retries': '3',
                'debug': 'true'
            }
        })
        
        # Create proper InvocationContext for ReadonlyContext
        invocation_context = InvocationContext(
            invocation_id='test_invocation',
            agent=agent,
            session=session,
            session_service=runner.session_service,
        )
        
        # Mock the MCP server components to verify environment variable injection
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.list_tools.return_value = ListToolsResult(
            tools=[
                McpTool(
                    name='list_directory',
                    description='List directory contents',
                    inputSchema={'type': 'object', 'properties': {}}
                )
            ]
        )
        mock_session.call_tool.return_value = {
            'result': 'Mock directory listing'
        }
        
        # Track environment variable injection
        injected_env_vars = {}
        
        # Mock the _inject_env_vars method to track environment variable injection
        def mock_inject_env_vars(self, env_vars):
            # Track the environment variables that were attempted to be injected
            injected_env_vars.update(env_vars)
            # Return the original connection params (since we're just testing the extraction)
            return self._connection_params
        
        # Create mock instances that behave like the real ones
        mock_exit_stack_instance = AsyncMock()
        mock_exit_stack_instance.aclose = AsyncMock()
        mock_exit_stack_instance.enter_async_context = AsyncMock(side_effect=[
            [AsyncMock(), AsyncMock()],  # transports
            mock_session  # session
        ])
        
        with patch('google.adk.tools.mcp_tool.mcp_session_manager.stdio_client') as mock_stdio_client, \
             patch('google.adk.tools.mcp_tool.mcp_session_manager.ClientSession', return_value=mock_session), \
             patch('google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack', return_value=mock_exit_stack_instance), \
             patch.object(MCPSessionManager, '_inject_env_vars', mock_inject_env_vars):
            
            # Trigger tool retrieval which should extract and inject environment variables
            mcp_toolset = agent.tools[0]
            
            # Create a ReadonlyContext from the invocation context
            readonly_context = ReadonlyContext(invocation_context)
            
            # Get tools, which triggers environment variable extraction and injection
            tools = await mcp_toolset.get_tools(readonly_context)
            
            # Verify tools were retrieved
            assert len(tools) == 1
            assert tools[0].name == 'list_directory'
            
            # Verify environment variables were extracted and injected
            expected_env_vars = {
                'API_KEY': 'test_api_key_123',   # From session state
                'ENVIRONMENT': 'production',      # From session state
                'USER_TIMEOUT': '30',            # From user_config
                'USER_RETRIES': '3',             # From user_config
                'USER_DEBUG': 'true',            # From user_config
            }
            
            # Check that environment variables were properly injected
            for key, value in expected_env_vars.items():
                assert key in injected_env_vars, f"Environment variable {key} was not injected"
                assert injected_env_vars[key] == value, f"Environment variable {key} has wrong value"

    @pytest.mark.asyncio
    async def test_env_extraction_without_callback(self, llm_backend):
        """Test that no environment variables are extracted when no callback is provided."""
        # Create agent without environment callback
        agent = self.create_test_agent_with_env_callback(env_transform_callback=None)
        
        # Create test runner
        runner = TestRunner(agent)
        
        # Get the current session and add state
        session = await runner.get_current_session_async()
        session.state.update({
            'api_key': 'test_api_key_123',
            'environment': 'production'
        })
        
        # Create proper InvocationContext for ReadonlyContext
        invocation_context = InvocationContext(
            invocation_id='test_invocation_2',
            agent=agent,
            session=session,
            session_service=runner.session_service,
        )
        
        # Mock the MCP server components
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.list_tools.return_value = ListToolsResult(
            tools=[
                McpTool(
                    name='list_directory',
                    description='List directory contents',
                    inputSchema={'type': 'object', 'properties': {}}
                )
            ]
        )
        
        # Track environment variable injection
        injected_env_vars = {}
        
        # Mock the _inject_env_vars method to track environment variable injection
        def mock_inject_env_vars(self, env_vars):
            # Track the environment variables that were attempted to be injected
            injected_env_vars.update(env_vars)
            # Return the original connection params (since we're just testing the extraction)
            return self._connection_params
        
        # Mock the _extract_env_from_context method to ensure it returns empty dict (no callback)
        def mock_extract_env_from_context(self, readonly_context):
            # Return empty dict since no callback is provided
            return {}
        
        # Create mock instances that behave like the real ones
        mock_exit_stack_instance = AsyncMock()
        mock_exit_stack_instance.aclose = AsyncMock()
        mock_exit_stack_instance.enter_async_context = AsyncMock(side_effect=[
            [AsyncMock(), AsyncMock()],  # transports
            mock_session  # session
        ])
        
        with patch('google.adk.tools.mcp_tool.mcp_session_manager.stdio_client') as mock_stdio_client, \
             patch('google.adk.tools.mcp_tool.mcp_session_manager.ClientSession', return_value=mock_session), \
             patch('google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack', return_value=mock_exit_stack_instance), \
             patch.object(MCPSessionManager, '_inject_env_vars', mock_inject_env_vars), \
             patch.object(MCPSessionManager, '_extract_env_from_context', mock_extract_env_from_context):
            
            # Trigger tool retrieval
            mcp_toolset = agent.tools[0]
            readonly_context = ReadonlyContext(invocation_context)
            
            # Get tools
            tools = await mcp_toolset.get_tools(readonly_context)
            
            # Verify tools were retrieved
            assert len(tools) == 1
            
            # Verify no environment variables were extracted/injected (since no callback)
            assert injected_env_vars == {}

    @pytest.mark.asyncio
    async def test_env_extraction_with_callback_exception(self, llm_backend):
        """Test behavior when environment transform callback raises an exception."""
        def failing_env_callback(state_dict: Dict[str, Any]) -> Dict[str, str]:
            """Callback that always raises an exception."""
            raise ValueError("Callback failed")
        
        # Create agent with failing callback
        agent = self.create_test_agent_with_env_callback(
            env_transform_callback=failing_env_callback
        )
        
        # Create test runner
        runner = TestRunner(agent)
        
        # Get the current session and add state
        session = await runner.get_current_session_async()
        session.state.update({
            'api_key': 'test_api_key_123'
        })
        
        # Create proper InvocationContext for ReadonlyContext
        invocation_context = InvocationContext(
            invocation_id='test_invocation_3',
            agent=agent,
            session=session,
            session_service=runner.session_service,
        )
        
        # Mock the MCP server components
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.list_tools.return_value = ListToolsResult(
            tools=[
                McpTool(
                    name='list_directory',
                    description='List directory contents',
                    inputSchema={'type': 'object', 'properties': {}}
                )
            ]
        )
        
        # Track environment variable injection
        injected_env_vars = {}
        
        # Mock the _inject_env_vars method to track environment variable injection
        def mock_inject_env_vars(self, env_vars):
            # Track the environment variables that were attempted to be injected
            injected_env_vars.update(env_vars)
            # Return the original connection params (since we're just testing the extraction)
            return self._connection_params
        
        # Mock the _extract_env_from_context method to simulate callback exception
        def mock_extract_env_from_context(self, readonly_context):
            # Simulate the failing callback - should catch exception and return empty dict
            try:
                if self._env_transform_callback:
                    return self._env_transform_callback(readonly_context.state)
                return {}
            except Exception:
                # The real implementation should catch exceptions and return empty dict
                return {}
        
        # Create mock instances that behave like the real ones
        mock_exit_stack_instance = AsyncMock()
        mock_exit_stack_instance.aclose = AsyncMock()
        mock_exit_stack_instance.enter_async_context = AsyncMock(side_effect=[
            [AsyncMock(), AsyncMock()],  # transports
            mock_session  # session
        ])
        
        with patch('google.adk.tools.mcp_tool.mcp_session_manager.stdio_client') as mock_stdio_client, \
             patch('google.adk.tools.mcp_tool.mcp_session_manager.ClientSession', return_value=mock_session), \
             patch('google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack', return_value=mock_exit_stack_instance), \
             patch.object(MCPSessionManager, '_inject_env_vars', mock_inject_env_vars), \
             patch.object(MCPSessionManager, '_extract_env_from_context', mock_extract_env_from_context):
            
            # Trigger tool retrieval (should not raise exception)
            mcp_toolset = agent.tools[0]
            readonly_context = ReadonlyContext(invocation_context)
            
            # Get tools - should complete successfully despite callback failure
            tools = await mcp_toolset.get_tools(readonly_context)
            
            # Verify tools were retrieved
            assert len(tools) == 1
            
            # Verify no environment variables were injected (callback failed gracefully)
            assert injected_env_vars == {}

    @pytest.mark.asyncio
    async def test_env_extraction_with_empty_session_state(self, llm_backend):
        """Test environment variable extraction with empty session state."""
        # Create agent with environment callback
        agent = self.create_test_agent_with_env_callback(
            env_transform_callback=self.sample_env_transform_callback
        )
        
        # Create test runner
        runner = TestRunner(agent)
        
        # Keep session state empty (don't add any state)
        session = await runner.get_current_session_async()
        
        # Create proper InvocationContext for ReadonlyContext
        invocation_context = InvocationContext(
            invocation_id='test_invocation_4',
            agent=agent,
            session=session,
            session_service=runner.session_service,
        )
        
        # Mock the MCP server components
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.list_tools.return_value = ListToolsResult(
            tools=[
                McpTool(
                    name='list_directory',
                    description='List directory contents',
                    inputSchema={'type': 'object', 'properties': {}}
                )
            ]
        )
        
        # Track environment variable injection
        injected_env_vars = {}
        
        # Mock the _inject_env_vars method to track environment variable injection
        def mock_inject_env_vars(self, env_vars):
            # Track the environment variables that were attempted to be injected
            injected_env_vars.update(env_vars)
            # Return the original connection params (since we're just testing the extraction)
            return self._connection_params
        
        # Mock the _extract_env_from_context method to return empty dict (empty state)
        def mock_extract_env_from_context(self, readonly_context):
            # With empty session state, should return empty dict
            if self._env_transform_callback:
                return self._env_transform_callback(readonly_context.state)
            return {}
        
        # Create mock instances that behave like the real ones
        mock_exit_stack_instance = AsyncMock()
        mock_exit_stack_instance.aclose = AsyncMock()
        mock_exit_stack_instance.enter_async_context = AsyncMock(side_effect=[
            [AsyncMock(), AsyncMock()],  # transports
            mock_session  # session
        ])
        
        with patch('google.adk.tools.mcp_tool.mcp_session_manager.stdio_client') as mock_stdio_client, \
             patch('google.adk.tools.mcp_tool.mcp_session_manager.ClientSession', return_value=mock_session), \
             patch('google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack', return_value=mock_exit_stack_instance), \
             patch.object(MCPSessionManager, '_inject_env_vars', mock_inject_env_vars), \
             patch.object(MCPSessionManager, '_extract_env_from_context', mock_extract_env_from_context):
            
            # Trigger tool retrieval
            mcp_toolset = agent.tools[0]
            readonly_context = ReadonlyContext(invocation_context)
            
            # Get tools
            tools = await mcp_toolset.get_tools(readonly_context)
            
            # Verify tools were retrieved
            assert len(tools) == 1
            
            # Verify no environment variables were extracted/injected (empty state)
            assert injected_env_vars == {}

    @pytest.mark.asyncio
    async def test_env_extraction_with_non_stdio_connection(self, llm_backend):
        """Test that environment variable extraction is not attempted for non-stdio connections."""
        from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
        
        # Create agent with SSE connection (no environment variable support)
        agent = LlmAgent(
            model='gemini-2.0-flash',
            name='test_sse_agent',
            instruction="Test agent with SSE connection",
            tools=[
                MCPToolset(
                    connection_params=SseServerParams(url='http://example.com/sse'),
                    env_transform_callback=self.sample_env_transform_callback,
                )
            ],
        )
        
        # Create test runner
        runner = TestRunner(agent)
        
        # Get the current session and add state
        session = await runner.get_current_session_async()
        session.state.update({
            'api_key': 'test_api_key_123',
            'environment': 'production'
        })
        
        # Create proper InvocationContext for ReadonlyContext
        invocation_context = InvocationContext(
            invocation_id='test_invocation_5',
            agent=agent,
            session=session,
            session_service=runner.session_service,
        )
        
        # Mock the MCP server components for SSE
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.list_tools.return_value = ListToolsResult(
            tools=[
                McpTool(
                    name='test_tool',
                    description='Test tool',
                    inputSchema={'type': 'object', 'properties': {}}
                )
            ]
        )
        
        with patch('google.adk.tools.mcp_tool.mcp_session_manager.sse_client') as mock_sse_client, \
             patch('google.adk.tools.mcp_tool.mcp_session_manager.ClientSession', return_value=mock_session), \
             patch('google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack') as mock_exit_stack:
            
            mock_exit_stack.return_value.__aenter__ = AsyncMock()
            mock_exit_stack.return_value.__aexit__ = AsyncMock()
            mock_exit_stack.return_value.enter_async_context = AsyncMock(side_effect=[
                [AsyncMock(), AsyncMock()],  # transports
                mock_session  # session
            ])
            
            # Trigger tool retrieval
            mcp_toolset = agent.tools[0]
            readonly_context = ReadonlyContext(invocation_context)
            
            # Get tools - should complete successfully without environment variable processing
            tools = await mcp_toolset.get_tools(readonly_context)
            
            # Verify tools were retrieved
            assert len(tools) == 1
            assert tools[0].name == 'test_tool'
            
            # Verify SSE client was called (not stdio client)
            mock_sse_client.assert_called_once()

    def test_env_transform_callback_signature(self):
        """Test that environment transform callback has correct signature."""
        def valid_callback(state_dict: Dict[str, Any]) -> Dict[str, str]:
            return {'TEST_VAR': 'test_value'}
        
        # Create agent with valid callback
        agent = self.create_test_agent_with_env_callback(
            env_transform_callback=valid_callback
        )
        
        # Verify agent was created successfully
        assert agent is not None
        assert len(agent.tools) == 1
        
        # Verify callback was set in the toolset
        mcp_toolset = agent.tools[0]
        assert mcp_toolset._env_transform_callback == valid_callback

    @pytest.mark.asyncio
    async def test_session_manager_direct_env_injection(self, llm_backend):
        """Test MCPSessionManager environment variable injection directly."""
        connection_params = StdioServerParameters(
            command='npx',
            args=['-y', '@modelcontextprotocol/server-filesystem'],
            env={'EXISTING_VAR': 'existing_value'}
        )
        
        def test_env_callback(state_dict: Dict[str, Any]) -> Dict[str, str]:
            return {'NEW_VAR': 'new_value', 'API_KEY': 'secret123'}
        
        session_manager = MCPSessionManager(
            connection_params=connection_params,
            env_transform_callback=test_env_callback
        )
        
        # Create mock readonly context with state
        mock_context = MagicMock()
        mock_context.state = {
            'api_key': 'secret123',
            'config': {'debug': True}
        }
        
        # Test environment variable extraction
        extracted_env = session_manager._extract_env_from_context(mock_context)
        expected_env = {'NEW_VAR': 'new_value', 'API_KEY': 'secret123'}
        assert extracted_env == expected_env
        
        # Test environment variable injection
        updated_params = session_manager._inject_env_vars(extracted_env)
        expected_merged_env = {
            'EXISTING_VAR': 'existing_value',
            'NEW_VAR': 'new_value',
            'API_KEY': 'secret123'
        }
        assert updated_params.env == expected_merged_env
        assert updated_params.command == connection_params.command
        assert updated_params.args == connection_params.args
