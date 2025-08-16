"""
Integration Test for Agent Tool Routing and Live API Calls

This test verifies that the coordinator agent correctly routes queries to the 
appropriate tools and that those tools can interact with live Google Cloud APIs.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path for imports
import os
import sys
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from google.genai import types
from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext

# Mock asset_inventory_tools since tools directory was removed
class MockAssetInventoryTools:
    @staticmethod
    async def discover_gcp_resources(query: str, tool_context: ToolContext):
        return {
            "success": True,
            "data": {"assets": []},
            "query_processed": query
        }

asset_inventory_tools = MockAssetInventoryTools()

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture
def adk_services():
    """Provides a cached instance of the ADK agent and services."""

    async def mock_generate_content(*args, **kwargs):
        """
        Dynamically determines which tool to call based on the query content.
        """
        contents = kwargs.get("contents", [])
        if not contents:
            return types.GenerateContentResponse()

        # Extract the text from the last user message
        last_user_message = next(
            (c for c in reversed(contents) if c.role == "user"), None
        )
        if not last_user_message or not last_user_message.parts:
            return types.GenerateContentResponse()
        query = last_user_message.parts[0].text

        # Determine which tool to call based on the query
        if "compute instances" in query:
            tool_name = "discover_gcp_resources"
            tool_args = {"query": query}
        elif "security of Cloud Storage" in query:
            tool_name = "evaluate_api_security"
            tool_args = {"api_name": "Cloud Storage"}
        else:
            # Default response if no tool matches
            return types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            parts=[
                                types.Part(
                                    text="Sorry, I can't help with that."
                                )
                            ]
                        )
                    )
                ]
            )

        # Create the response with the appropriate function call
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=tool_name, args=tool_args
                                )
                            )
                        ]
                    )
                )
            ]
        )

    with patch(
        "google.adk.models.google_llm.Client", new_callable=MagicMock
    ) as mock_client:
        mock_async_client = MagicMock()
        mock_async_client.generate_content.side_effect = mock_generate_content
        mock_client.return_value.aio.models = mock_async_client


        session_service = InMemorySessionService()
        runner = Runner(
            agent=root_agent,
            session_service=session_service,
            app_name="test_app",
        )
        yield root_agent, runner, session_service



async def run_agent_query(runner, session_service, query):
    """Helper function to run a query against the agent and get the final response."""
    session = await session_service.create_session(app_name="test_app", user_id="test_user")
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    final_response = ""
    try:
        events = runner.run_async(
            user_id="test_user", session_id=session.id, new_message=content
        )
        
        async for event in events:
            if event.is_final_response():
                print(f"LLM Response: {event.content.parts[0].text}")
            elif event.is_tool_request():
                print(f"Tool Request: {event.get_function_calls()}")
            elif event.is_tool_response():
                print(f"Tool Response: {event.get_function_responses()}")

            if event.is_final_response():
                final_response = event.content.parts[0].text
                break # Exit after getting the final response
    finally:
        await session_service.delete_session(
            app_name="test_app", user_id="test_user", session_id=session.id
        )
            
    return final_response

@patch('tools.gcp_tools.asset_inventory_tools.discover_gcp_resources', new_callable=AsyncMock)
async def test_route_to_asset_discovery(mock_discover_gcp_resources, adk_services):
    """
    Tests that a query about GCP resources correctly routes to the 
    `discover_gcp_resources` tool and processes the mocked response.
    """
    _, runner, session_service = adk_services
    
    # Configure the mock to return a successful response
    mock_discover_gcp_resources.return_value = {
        "success": True,
        "data": {
            "compute_instances": [
                {"name": "test-instance-1", "asset_type": "ComputeInstance", "project_id": "test-project"}
            ],
            "storage_buckets": [],
        },
        "query_processed": "show me my compute instances"
    }
    
    query = "show me my compute instances"
    response = await run_agent_query(runner, session_service, query)
    
    # Assertions
    mock_discover_gcp_resources.assert_called_once()
    call_args, call_kwargs = mock_discover_gcp_resources.call_args
    assert call_kwargs['query'] == query
    assert isinstance(call_kwargs['tool_context'], ToolContext)

    assert "test-instance-1" in response
    assert "ComputeInstance" in response

@patch('tools.security_tools.knowledge_base_tools.evaluate_api_security', new_callable=AsyncMock)
async def test_route_to_security_evaluation(mock_evaluate_api_security, adk_services):
    """
    Tests that a query about API security correctly routes to the 
    `evaluate_api_security` tool.
    """
    _, runner, session_service = adk_services
    
    mock_evaluate_api_security.return_value = "Security Evaluation for Cloud Storage: It is secure."
    
    query = "evaluate the security of Cloud Storage"
    response = await run_agent_query(runner, session_service, query)
    
    # Assertions
    mock_evaluate_api_security.assert_called_once()
    call_args, call_kwargs = mock_evaluate_api_security.call_args
    assert call_kwargs['api_name'] == "Cloud Storage"
    assert isinstance(call_kwargs['tool_context'], ToolContext)

    assert "Security Evaluation for Cloud Storage" in response
    assert "It is secure" in response
    

@pytest.mark.live
async def test_live_gcp_asset_discovery(adk_services):
    """
    Tests the `discover_gcp_resources` tool with a live call to the GCP API.
    
    NOTE: This test requires valid GCP credentials and that the Cloud Asset API 
    is enabled for the project.
    """
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        pytest.skip("Skipping live test: GOOGLE_APPLICATION_CREDENTIALS not set.")
        
    _, runner, session_service = adk_services
    
    query = "list the storage buckets in the project"
    response = await run_agent_query(runner, session_service, query)
    
    # We can't assert specific bucket names, but we can check for structure
    assert "storage bucket" in response.lower() or "no storage buckets" in response.lower()
    assert "Error" not in response
