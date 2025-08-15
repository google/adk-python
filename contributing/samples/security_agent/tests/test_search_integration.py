"""
Test suite for Google Web Search integration in ADK Security Agent.
Tests the search service, tools, and LLM agent integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

# Import components to test
from backend.api.search import SearchService
from backend.models.search_models import (
    SearchRequest, SearchResponse, SearchResult,
    SearchContextRequest, SearchContextResponse
)
from tools.api_tools.google_search_tools import (
    search_web, get_search_context, search_security_topics
)


class TestSearchService:
    """Test the SearchService class."""
    
    @pytest.fixture
    def search_service(self):
        """Create a search service instance."""
        return SearchService()
    
    @pytest.mark.asyncio
    async def test_search_with_mock_results(self, search_service):
        """Test search returns mock results when API not configured."""
        # API keys not configured, should return mock results
        results = await search_service.search(
            query="GCP security best practices",
            max_results=5,
            safe_search=True
        )
        
        assert results is not None
        assert "results" in results
        assert len(results["results"]) > 0
        assert results["query"] == "GCP security best practices"
        
        # Check mock result structure
        first_result = results["results"][0]
        assert "title" in first_result
        assert "url" in first_result
        assert "snippet" in first_result
        assert "Security" in first_result["title"]  # Security query should have security results
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, search_service):
        """Test rate limiting functionality."""
        user_id = "test_user"
        
        # Should allow initial requests
        assert search_service._check_rate_limit(user_id) is True
        
        # Add requests to tracker
        for _ in range(100):  # Default limit is 100/minute
            search_service._update_rate_limit(user_id)
        
        # Should now be rate limited
        assert search_service._check_rate_limit(user_id) is False
    
    @pytest.mark.asyncio
    async def test_search_caching(self, search_service):
        """Test search result caching."""
        query = "test query"
        
        # First search should hit API/mock
        results1 = await search_service.search(query, max_results=5)
        
        # Second search should use cache (faster)
        import time
        start_time = time.time()
        results2 = await search_service.search(query, max_results=5)
        cache_time = time.time() - start_time
        
        # Results should be identical
        assert results1 == results2
        # Cache lookup should be very fast
        assert cache_time < 0.1


class TestSearchModels:
    """Test Pydantic models for search."""
    
    def test_search_request_validation(self):
        """Test SearchRequest model validation."""
        # Valid request
        request = SearchRequest(
            query="test query",
            session_id="session123",
            user_id="user123",
            safe_search=True,
            max_results=10
        )
        assert request.query == "test query"
        assert request.max_results == 10
        
        # Test validation - empty query should fail
        with pytest.raises(ValueError):
            SearchRequest(
                query="",  # Empty query
                session_id="session123",
                user_id="user123"
            )
        
        # Test max results validation
        with pytest.raises(ValueError):
            SearchRequest(
                query="test",
                session_id="session123",
                user_id="user123",
                max_results=25  # Over limit of 20
            )
    
    def test_search_response_model(self):
        """Test SearchResponse model."""
        response = SearchResponse(
            success=True,
            query="test",
            results=[
                SearchResult(
                    title="Test Result",
                    url="https://example.com",
                    snippet="Test snippet",
                    display_url="example.com"
                )
            ],
            total_results=1,
            search_time_ms=150,
            session_id="session123"
        )
        
        assert response.success is True
        assert len(response.results) == 1
        assert response.results[0].title == "Test Result"


class TestSearchTools:
    """Test the search tool functions."""
    
    @patch('requests.post')
    def test_search_web_tool(self, mock_post):
        """Test search_web tool function."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "snippet": "Test snippet",
                    "display_url": "example.com"
                }
            ],
            "total_results": 1,
            "search_time_ms": 100
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Call search tool
        result = search_web(
            query="test query",
            max_results=5,
            session_id="session123"
        )
        
        # Verify result
        assert result is not None
        result_data = json.loads(result)
        assert result_data["query"] == "test query"
        assert len(result_data["results"]) == 1
    
    @patch('requests.post')
    def test_search_security_topics(self, mock_post):
        """Test security-focused search."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "results": [
                {
                    "title": "Security Best Practices",
                    "url": "https://security.example.com",
                    "snippet": "Important security information",
                    "display_url": "security.example.com"
                }
            ],
            "total_results": 1,
            "search_time_ms": 100
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Call security search
        result = search_security_topics(
            query="authentication vulnerabilities",
            session_id="session123"
        )
        
        # Parse result
        result_data = json.loads(result)
        
        # Should have security analysis
        assert "security_analysis" in result_data
        assert result_data["security_analysis"]["query_type"] == "security"
        assert len(result_data["security_analysis"]["focus_areas"]) > 0
        assert "authentication" in result_data["security_analysis"]["focus_areas"]


class TestAgentIntegration:
    """Test integration with LLM agent."""
    
    @pytest.mark.asyncio
    @patch('backend.api.agent_llm.SEARCH_SERVICE_AVAILABLE', True)
    async def test_search_agent_routing(self):
        """Test that search queries route to SearchAgent."""
        from backend.api.agent_llm import process_with_llm_agent
        
        # Search query should route to search agent
        response, agent_name = await process_with_llm_agent(
            query="search for GCP security best practices",
            project_id="test-project",
            request_id="test123"
        )
        
        assert agent_name == "SearchAgent"
        assert "Search" in response or "search" in response.lower()
    
    @pytest.mark.asyncio
    async def test_search_keywords_detection(self):
        """Test detection of search-related keywords."""
        from backend.api.agent_llm import process_with_llm_agent
        
        search_queries = [
            "find information about IAM policies",
            "lookup security vulnerabilities",
            "search for compliance requirements",
            "what is zero trust architecture",
            "how to implement MFA",
            "latest security threats",
            "recent CVE reports",
            "documentation for GCP security"
        ]
        
        for query in search_queries:
            _, agent_name = await process_with_llm_agent(
                query=query,
                project_id="test-project",
                request_id=f"test-{query[:10]}"
            )
            assert agent_name == "SearchAgent", f"Query '{query}' should route to SearchAgent"


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_search_flow(self):
        """Test complete search flow from query to response."""
        # This would test the full integration in a real environment
        # For now, we'll test the components are properly connected
        
        # 1. User query comes in
        query = "search for cloud security best practices"
        
        # 2. Tool function processes it
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "success": True,
                "results": [{"title": "Security Guide", "url": "https://example.com", "snippet": "Best practices"}],
                "total_results": 1,
                "search_time_ms": 100
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            result = search_web(query)
            assert result is not None
            
        # 3. Agent processes and formats response
        from backend.api.agent_llm import generate_response_with_real_data
        
        with patch('backend.api.agent_llm.SEARCH_SERVICE_AVAILABLE', True):
            with patch('backend.api.search.SearchService.search') as mock_search:
                mock_search.return_value = {
                    "results": [{"title": "Test", "url": "http://test.com", "snippet": "Test"}],
                    "total_results": 1,
                    "search_time_ms": 100,
                    "query": query
                }
                
                response = await generate_response_with_real_data(
                    query=query,
                    project_id="test-project",
                    agent_type="search",
                    request_id="test123"
                )
                
                assert "Web Search Results" in response
                assert query in response


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])