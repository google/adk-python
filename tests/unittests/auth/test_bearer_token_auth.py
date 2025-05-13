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

"""Tests for Bearer token authentication in ADK."""

from unittest.mock import MagicMock, patch, call

import pytest
from fastapi.openapi.models import HTTPBearer

from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.tool_context import ToolContext


class TestBearerTokenAuth:
    """Tests for Bearer token authentication functionality."""

    @pytest.fixture
    def test_token(self):
        """Provides a test token."""
        return "test_bearer_token_123"
    
    @pytest.fixture
    def http_bearer_scheme(self):
        """Creates an HTTPBearer authentication scheme."""
        return HTTPBearer(bearerFormat="JWT")
    
    @pytest.fixture
    def bearer_auth_credential(self, test_token):
        """Creates a credential for a Bearer token."""
        return AuthCredential(
            auth_type=AuthCredentialTypes.HTTP,
            http=HttpAuth(
                scheme="bearer",
                credentials=HttpCredentials(token=test_token)
            )
        )

    def test_new_invocation_context_with_auth_token(self, test_token):
        """Tests that the authentication token is correctly configured in requested_auth_configs."""
        with patch("google.adk.runners.InvocationContext") as mock_context_class:
            from google.adk.runners import Runner
            
            mock_artifact_service = MagicMock()
            mock_session_service = MagicMock()
            mock_memory_service = MagicMock()
            mock_agent = MagicMock()
            mock_session = MagicMock()
            
            runner = Runner(
                app_name="test_app",
                agent=mock_agent,
                artifact_service=mock_artifact_service,
                session_service=mock_session_service,
                memory_service=mock_memory_service
            )
            
            runner._new_invocation_context(
                mock_session,
                auth_token=test_token
            )
            
            calls = mock_context_class.call_args_list
            assert len(calls) == 1
            
            kwargs = calls[0].kwargs
            assert "requested_auth_configs" in kwargs
            auth_configs = kwargs["requested_auth_configs"]
            
            assert auth_configs is not None
            assert "bearer" in auth_configs
            
            bearer_auth = auth_configs["bearer"]
            assert bearer_auth.auth_type == AuthCredentialTypes.HTTP
            assert bearer_auth.http.scheme == "bearer"
            assert bearer_auth.http.credentials.token == test_token

    def test_tool_context_get_auth_response_with_bearer_token(self, test_token, http_bearer_scheme, bearer_auth_credential):
        """Tests that ToolContext.get_auth_response returns the Bearer token when available."""
        mock_invocation_context = MagicMock()
        mock_invocation_context.requested_auth_configs = {
            "bearer": bearer_auth_credential
        }
        
        tool_context = ToolContext(
            invocation_context=mock_invocation_context,
            function_call_id="test_function_call",
            event_actions=MagicMock()
        )
        
        auth_config = AuthConfig(
            auth_scheme=http_bearer_scheme,
            raw_auth_credential=bearer_auth_credential
        )
        
        auth_response = tool_context.get_auth_response(auth_config)
        
        assert auth_response is not None
        assert auth_response.auth_type == AuthCredentialTypes.HTTP
        assert auth_response.http.scheme == "bearer"
        assert auth_response.http.credentials.token == test_token

    def test_tool_context_get_auth_response_without_bearer_token(self, http_bearer_scheme, bearer_auth_credential):
        """Tests fallback behavior when no Bearer token is available."""
        mock_invocation_context = MagicMock()
        mock_invocation_context.requested_auth_configs = {}
        
        tool_context = ToolContext(
            invocation_context=mock_invocation_context,
            function_call_id="test_function_call",
            event_actions=MagicMock()
        )
        
        auth_config = AuthConfig(
            auth_scheme=http_bearer_scheme,
            raw_auth_credential=bearer_auth_credential
        )
        
        mock_auth_handler = MagicMock()
        mock_auth_handler.return_value.get_auth_response.return_value = "fallback_response"
        
        with patch("google.adk.tools.tool_context.AuthHandler", mock_auth_handler):
            auth_response = tool_context.get_auth_response(auth_config)
            
            mock_auth_handler.return_value.get_auth_response.assert_called_once()
            assert auth_response == "fallback_response"

    @pytest.mark.asyncio
    async def test_run_async_passes_auth_token_to_invocation_context(self, test_token, monkeypatch):
        """Tests that run_async passes auth_token to _new_invocation_context."""
        from google.adk.runners import Runner
        from google.genai import types
        
        mock_artifact_service = MagicMock()
        mock_session_service = MagicMock()
        mock_memory_service = MagicMock()
        mock_agent = MagicMock()
        mock_session = MagicMock()
        
        mock_session_service.get_session.return_value = mock_session
        
        runner = Runner(
            app_name="test_app",
            agent=mock_agent,
            artifact_service=mock_artifact_service,
            session_service=mock_session_service,
            memory_service=mock_memory_service
        )
        
        original_method = runner._new_invocation_context
        calls = []
        
        def mock_new_invocation_context(*args, **kwargs):
            calls.append((args, kwargs))
            mock_context = MagicMock()
            mock_context.agent = mock_agent
            mock_context.invocation_id = "test_invocation_id"
            return mock_context
            
        monkeypatch.setattr(runner, '_new_invocation_context', mock_new_invocation_context)
        
        async def mock_run_async(*args, **kwargs):
            yield MagicMock()
        
        mock_agent.run_async = mock_run_async
        
        new_message = types.Content(parts=[types.Part(text="test")])
        
        async for _ in runner.run_async(
            user_id="test_user",
            session_id="test_session",
            new_message=new_message,
            auth_token=test_token
        ):
            pass  # Consume the async generator
        
        assert len(calls) > 0
        
        last_call_kwargs = calls[-1][1]
        assert "auth_token" in last_call_kwargs
        assert last_call_kwargs["auth_token"] == test_token