"""
Tests for Frontend Router Agent and Local Lookup Agent.

Tests query analysis, enhancement, and local cache functionality.
"""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
from frontend.agents.frontend_router import FrontendRouterAgent, LocalLookupAgent, QueryAnalysis
from frontend.services.agent_service import FrontendAgentService
from frontend.agents.prompts import PromptTemplates

class TestFrontendRouterAgent:
    """Test the FrontendRouterAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the configuration to avoid requiring actual API keys
        with patch('frontend.agents.frontend_router.FrontendConfig') as mock_config:
            mock_config.get_frontend_agent_config.return_value = {
                'router_enabled': True,
                'gemini_api_key': 'test-key',
                'router_model': 'gemini-1.5-flash',
                'log_enhancements': False
            }
            
            with patch('frontend.agents.frontend_router.genai'):
                self.router = FrontendRouterAgent()
                self.router.model = MagicMock()
                self.router.enabled = True

    def test_simple_analysis_fallback(self):
        """Test simple analysis when LLM is not available."""
        router = FrontendRouterAgent()
        router.enabled = False
        
        # Test bucket query
        analysis = router.analyze_query("Show me bucket encryption status")
        assert analysis.query_type == 'data'
        assert analysis.needs_backend == True
        assert analysis.suggested_tool == 'storage_buckets'
        assert analysis.confidence == 0.7

    def test_data_query_classification(self):
        """Test classification of different data queries."""
        router = FrontendRouterAgent()
        router.enabled = False  # Use simple analysis
        
        test_cases = [
            ("What security findings do I have?", 'security_findings'),
            ("List IAM users", 'iam_accounts'),
            ("Show me storage buckets", 'storage_buckets'),
            ("Network security analysis", 'networks'),
            ("Asset inventory", 'assets')
        ]
        
        for query, expected_tool in test_cases:
            analysis = router.analyze_query(query)
            assert analysis.query_type == 'data'
            assert analysis.suggested_tool == expected_tool

    def test_help_query_classification(self):
        """Test classification of help queries."""
        router = FrontendRouterAgent()
        router.enabled = False  # Use simple analysis
        
        help_queries = [
            "How do I encrypt my data?",
            "What are the best practices?",
            "Explain security findings",
            "Help me understand IAM"
        ]
        
        for query in help_queries:
            analysis = router.analyze_query(query)
            assert analysis.query_type == 'help'
            assert analysis.needs_backend == True

    @patch('frontend.agents.frontend_router.genai')
    def test_llm_analysis_success(self, mock_genai):
        """Test successful LLM-based analysis."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.text = '''
        {
            "query_type": "data",
            "needs_backend": true,
            "enhanced_query": "Show detailed encryption status for all Cloud Storage buckets including key management",
            "suggested_tool": "storage_buckets",
            "confidence": 0.9
        }
        '''
        
        self.router.model.generate_content.return_value = mock_response
        
        analysis = self.router.analyze_query("bucket encryption")
        
        assert analysis.query_type == 'data'
        assert analysis.needs_backend == True
        assert analysis.suggested_tool == 'storage_buckets'
        assert analysis.confidence == 0.9
        assert "detailed encryption status" in analysis.enhanced_query

    def test_llm_analysis_fallback_on_error(self):
        """Test fallback to simple analysis when LLM fails."""
        # Mock LLM to raise an exception
        self.router.model.generate_content.side_effect = Exception("API Error")
        
        analysis = self.router.analyze_query("Show me bucket encryption")
        
        # Should fall back to simple analysis
        assert analysis.query_type == 'data'
        assert analysis.suggested_tool == 'storage_buckets'
        assert analysis.confidence == 0.7  # Simple analysis confidence

    def test_enhance_for_backend(self):
        """Test query enhancement for backend."""
        analysis = QueryAnalysis(
            query_type='data',
            needs_backend=True,
            enhanced_query="Enhanced query about buckets",
            suggested_tool='storage_buckets',
            confidence=0.8
        )
        
        enhanced = self.router.enhance_for_backend(
            "Show buckets", 
            analysis
        )
        
        assert "IMPORTANT: Use query_security_data with query_type='storage_buckets'" in enhanced
        assert "Enhanced query about buckets" in enhanced


class TestLocalLookupAgent:
    """Test the LocalLookupAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.local_agent = LocalLookupAgent()

    def test_can_handle_locally(self):
        """Test local query detection."""
        test_cases = [
            ("How to encrypt data?", True),
            ("Best practices for security", True),
            ("Help me with GCP", True),
            ("What can you do?", True),
            ("Show me specific bucket data", False),
            ("List my IAM users", False)
        ]
        
        for query, expected in test_cases:
            assert self.local_agent.can_handle_locally(query) == expected

    def test_handle_query_encryption(self):
        """Test handling of encryption queries."""
        response = self.local_agent.handle_query("How do I encrypt my data?")
        
        assert response['success'] == True
        assert 'Storage Buckets' in response['response']
        assert 'Cloud KMS' in response['response']
        assert response['needs_backend'] == True
        assert response['source'] == 'local_cache'

    def test_handle_query_best_practices(self):
        """Test handling of best practices queries."""
        response = self.local_agent.handle_query("What are the best practices?")
        
        assert response['success'] == True
        assert 'MFA' in response['response']
        assert 'least privilege' in response['response']
        assert response['needs_backend'] == True

    def test_handle_query_help(self):
        """Test handling of help queries."""
        response = self.local_agent.handle_query("help me")
        
        assert response['success'] == True
        assert 'Security Analysis' in response['response']
        assert response['needs_backend'] == False  # Help doesn't need backend

    def test_handle_query_not_found(self):
        """Test handling of queries not in local knowledge."""
        response = self.local_agent.handle_query("specific project data")
        
        assert response['success'] == False
        assert response['needs_backend'] == True


class TestFrontendAgentService:
    """Test the FrontendAgentService integration."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('frontend.services.agent_service.FrontendConfig'):
            self.service = FrontendAgentService()
            self.service.router_agent = MagicMock()
            self.service.local_agent = MagicMock()

    def test_process_query_local_cache_hit(self):
        """Test processing with local cache hit."""
        # Mock local agent to handle query
        self.service.local_agent.can_handle_locally.return_value = True
        self.service.local_agent.handle_query.return_value = {
            'success': True,
            'response': 'Local response about encryption',
            'needs_backend': False,
            'source': 'local_cache'
        }
        
        response = self.service.process_query("How to encrypt data?")
        
        assert response['success'] == True
        assert response['source'] == 'local_cache'
        assert response['metadata']['cache_hit'] == True
        assert 'Local response about encryption' in response['response']

    def test_process_query_local_with_backend_followup(self):
        """Test local response that suggests backend followup."""
        # Mock local agent to provide response but suggest backend
        self.service.local_agent.can_handle_locally.return_value = True
        self.service.local_agent.handle_query.return_value = {
            'success': True,
            'response': 'General encryption info',
            'needs_backend': True,
            'source': 'local_cache'
        }
        
        # Mock backend service
        with patch('frontend.services.agent_service.send_message') as mock_send:
            mock_send.return_value = {
                'success': True,
                'response': 'Backend response with specific data'
            }
            
            response = self.service.process_query("Show bucket encryption")
            
            assert response['success'] == True
            assert 'General encryption info' in response['response']
            assert 'Backend response with specific data' in response['response']

    def test_process_query_with_enhancement(self):
        """Test query processing with enhancement."""
        # Mock local agent to not handle query
        self.service.local_agent.can_handle_locally.return_value = False
        
        # Mock router agent analysis
        mock_analysis = QueryAnalysis(
            query_type='data',
            needs_backend=True,
            enhanced_query='Enhanced bucket query',
            suggested_tool='storage_buckets',
            confidence=0.9
        )
        self.service.router_agent.analyze_query.return_value = mock_analysis
        self.service.router_agent.enhance_for_backend.return_value = 'IMPORTANT: Use query_security_data with query_type=\'storage_buckets\'. Enhanced bucket query'
        self.service.router_agent.enabled = True
        
        # Mock backend service
        with patch('frontend.services.agent_service.send_message') as mock_send:
            mock_send.return_value = {
                'success': True,
                'response': 'Backend response'
            }
            
            response = self.service.process_query("show buckets")
            
            assert response['success'] == True
            assert response['metadata']['enhanced'] == True
            assert response['metadata']['analysis']['query_type'] == 'data'
            assert response['metadata']['analysis']['suggested_tool'] == 'storage_buckets'

    def test_get_recent_context(self):
        """Test conversation context extraction."""
        conversation = [
            {'role': 'user', 'content': 'Message 1'},
            {'role': 'assistant', 'content': 'Response 1'},
            {'role': 'user', 'content': 'Message 2'},
            {'role': 'assistant', 'content': 'Response 2'},
            {'role': 'user', 'content': 'Message 3'},
            {'role': 'assistant', 'content': 'Response 3'},
        ]
        
        context = self.service._get_recent_context(conversation)
        
        # Should return last 4 messages
        assert len(context) == 4
        assert context[0]['content'] == 'Response 1'
        assert context[-1]['content'] == 'Response 3'

    def test_get_stats(self):
        """Test service statistics."""
        self.service.router_agent.enabled = True
        
        stats = self.service.get_stats()
        
        assert 'router_enabled' in stats
        assert 'local_cache_enabled' in stats
        assert 'enhancement_enabled' in stats
        assert 'max_context_messages' in stats


class TestPromptTemplates:
    """Test the PromptTemplates class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.templates = PromptTemplates()

    def test_build_analysis_prompt(self):
        """Test analysis prompt building."""
        prompt = self.templates.build_analysis_prompt(
            "Show me buckets",
            "Previous context about security"
        )
        
        assert "Show me buckets" in prompt
        assert "Previous context about security" in prompt
        assert "JSON only" in prompt
        assert "query_type" in prompt

    def test_build_context_string(self):
        """Test context string building."""
        history = [
            {'role': 'user', 'content': 'First message'},
            {'role': 'assistant', 'content': 'First response'},
            {'role': 'user', 'content': 'Second message'}
        ]
        
        context = self.templates.build_context_string(history)
        
        assert "1. User: First message" in context
        assert "2. Assistant: First response" in context
        assert "3. User: Second message" in context

    def test_get_enhancement_prompt(self):
        """Test enhancement prompt generation."""
        analysis = {
            'suggested_tool': 'storage_buckets',
            'query_type': 'data'
        }
        
        enhanced = self.templates.get_enhancement_prompt(
            "show buckets",
            analysis
        )
        
        assert "Cloud Storage" in enhanced
        assert "show buckets" in enhanced

    def test_get_local_response(self):
        """Test local response retrieval."""
        help_response = self.templates.get_local_response('help')
        capabilities_response = self.templates.get_local_response('capabilities')
        
        assert "Security Analysis" in help_response
        assert "GCP Security Assistant" in capabilities_response

    def test_get_error_response(self):
        """Test error response retrieval."""
        error_response = self.templates.get_error_response('api_failure')
        unknown_error = self.templates.get_error_response('unknown_error')
        
        assert "trouble analyzing" in error_response
        assert "unexpected error" in unknown_error


if __name__ == '__main__':
    pytest.main([__file__])
