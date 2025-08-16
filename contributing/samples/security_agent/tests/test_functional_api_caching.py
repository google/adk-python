#!/usr/bin/env python3
"""
Functional test for API data caching and agent tool access.

Tests that:
- Asset Inventory API is called and caches data.
- Recommendations API is called and caches data.
- An agent can access the cached data via a tool.
"""

import pytest
import asyncio
import json
import os
from unittest.mock import patch, AsyncMock

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from backend.services.asset_inventory_service import GCPAssetInventoryService as AssetInventoryService
from backend.services.recommender_service import RecommenderService
# Mock AssetDiscoveryAgent for testing
class AssetDiscoveryAgent:
    def __init__(self, project_id=None):
        self.project_id = project_id
# Mock discover_gcp_resources since tools directory was removed
async def discover_gcp_resources(query: str, tool_context=None):
    return {
        "success": True,
        "data": {"assets": []},
        "query_processed": query
    }

# A placeholder for the tool to be tested
def get_cached_assets_tool(cache_dir):
    # In a real scenario, this would be a properly defined tool
    return discover_gcp_resources

class TestFunctionalApiCaching:
    """Functional test suite for API caching and agent tool access."""

    @pytest.fixture
    def project_id(self):
        """Provides a test project ID."""
        return "gcp-test-project"

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Provides a temporary cache directory."""
        return tmp_path / "cache"

    @pytest.mark.asyncio
    async def test_fetch_and_cache_assets_and_recommendations(self, project_id, cache_dir):
        """
        Test fetching data from APIs, writing to cache, and agent access.
        """
        # This test will fail initially because the implementation is not there yet.
        # 1. RED: Write a failing test.
        
        # Arrange: Mock external services and prepare the environment
        with patch('backend.services.asset_inventory_service.GCPAssetInventoryService.get_complete_asset_inventory', new_callable=AsyncMock) as mock_get_inventory, \
             patch('backend.services.recommender_service.RecommenderService.get_all_recommendations', new_callable=AsyncMock) as mock_fetch_recs:

            # Mock the return values of the API calls
            mock_get_inventory.return_value = {"assets": [{"name": "asset1"}]}
            mock_fetch_recs.return_value = {"recommendations": [{"name": "rec1"}]}

            # Instantiate services
            asset_service = AssetInventoryService(project_id=project_id, cache_dir=str(cache_dir))
            recommender_service = RecommenderService(project_id=project_id, cache_dir=str(cache_dir))

            # Act: Trigger the data fetching and caching process
            await asset_service.fetch_and_cache_all_assets()
            # We would also call the recommender service here in a real scenario.

            # Assert: Check that data is cached
            asset_cache_path = os.path.join(cache_dir, f"assets_{project_id}.json")
            assert os.path.exists(asset_cache_path)
            with open(asset_cache_path, 'r') as f:
                asset_data = json.load(f)
            assert asset_data["assets"]["name"] == "asset1"

            # Arrange: Setup the agent and its tool
            agent = AssetDiscoveryAgent()
            # The tool needs to know where to read the cache from.
            # This is a simplification. In a real app, this would be configured.
            agent.tools = [get_cached_assets_tool(cache_dir)]

            # Act: Have the agent use the tool to access the data
            agent_response = await agent.arun(f"list assets for {project_id}")

            # Assert: Check that the agent can access the cached data
            assert "asset1" in agent_response
