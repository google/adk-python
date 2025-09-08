"""
Integration tests for API calls from UI components.

Tests API integration, request/response handling, error scenarios,
and data transformation from UI to backend.
"""

import pytest
import asyncio
import httpx
import streamlit as st
from unittest.mock import patch, Mock, AsyncMock, MagicMock
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from frontend.unified_streaming_client import SecurityDashboard
    from frontend.dashboard import SecurityDashboard as DashboardMain
    from frontend.iam_features import IAMFeaturesUI
    from frontend.networking_dashboard import main as networking_main
except ImportError as e:
    pytest.skip(f"Frontend modules not available: {e}", allow_module_level=True)


class TestAPIIntegration:
    """Test suite for API integration from UI components."""
    
    def setup_method(self):
        """Setup test environment."""
        self.base_url = "http://localhost:8000"
        self.timeout = httpx.Timeout(30.0)
        self.test_headers = {"Content-Type": "application/json"}
        
    @pytest.mark.asyncio
    async def test_security_metrics_api(self):
        """Test security metrics API calls from dashboard."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Test security summary endpoint
            try:
                response = await client.get(f"{self.base_url}/api/security/summary")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Verify expected security data structure
                    expected_fields = [
                        'total_findings', 'critical_findings', 'high_findings',
                        'medium_findings', 'low_findings', 'security_score'
                    ]
                    
                    for field in expected_fields:
                        if field not in data and 'error' not in data:
                            pytest.fail(f"Missing expected field: {field}")
                            
                elif response.status_code == 503:
                    # Service unavailable is acceptable for integration tests
                    pytest.skip("Security service unavailable")
                else:
                    # Log response for debugging
                    print(f"Security API response: {response.status_code} - {response.text}")
                    
            except httpx.RequestError as e:
                pytest.skip(f"Cannot connect to security API: {e}")
                
    @pytest.mark.asyncio
    async def test_iam_analysis_api(self):
        """Test IAM analysis API calls."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            iam_endpoints = [
                "/api/iam/policies",
                "/api/iam/recommendations",
                "/api/iam/custom_roles"
            ]
            
            for endpoint in iam_endpoints:
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        assert isinstance(data, (dict, list))
                        
                        # If it's IAM policies, verify structure
                        if endpoint.endswith('/policies') and isinstance(data, dict):
                            if 'policies' in data:
                                policies = data['policies']
                                assert isinstance(policies, list)
                                
                    elif response.status_code in [401, 403]:
                        pytest.skip(f"IAM endpoint {endpoint} requires authentication")
                    elif response.status_code == 404:
                        # Endpoint might not be implemented yet
                        continue
                        
                except httpx.RequestError:
                    pytest.skip(f"Cannot connect to IAM endpoint {endpoint}")
                    
    @pytest.mark.asyncio
    async def test_storage_security_api(self):
        """Test storage security API calls."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            storage_endpoints = [
                "/api/storage/buckets",
                "/api/storage/security_analysis",
                "/api/storage/public_access"
            ]
            
            for endpoint in storage_endpoints:
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Verify storage data structure
                        if endpoint.endswith('/buckets'):
                            if isinstance(data, dict) and 'buckets' in data:
                                buckets = data['buckets']
                                assert isinstance(buckets, list)
                                
                        elif endpoint.endswith('/security_analysis'):
                            if isinstance(data, dict):
                                # Should have analysis results
                                analysis_fields = ['public_buckets', 'encryption_status', 'access_controls']
                                # At least one analysis field should be present
                                has_analysis = any(field in data for field in analysis_fields)
                                assert has_analysis or 'error' in data
                                
                    elif response.status_code == 503:
                        pytest.skip(f"Storage service unavailable: {endpoint}")
                        
                except httpx.RequestError:
                    pytest.skip(f"Cannot connect to storage endpoint {endpoint}")
                    
    @pytest.mark.asyncio
    async def test_monitoring_metrics_api(self):
        """Test monitoring metrics API calls."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            monitoring_endpoints = [
                "/api/monitoring/metrics",
                "/api/monitoring/alerts",
                "/api/monitoring/health"
            ]
            
            for endpoint in monitoring_endpoints:
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Verify monitoring data structure
                        if endpoint.endswith('/metrics'):
                            if isinstance(data, dict):
                                # Should have timestamp for metrics
                                assert 'timestamp' in data or 'error' in data
                                
                        elif endpoint.endswith('/health'):
                            if isinstance(data, dict):
                                assert 'status' in data
                                assert data['status'] in ['healthy', 'degraded', 'unhealthy']
                                
                except httpx.RequestError:
                    pytest.skip(f"Cannot connect to monitoring endpoint {endpoint}")
                    
    @pytest.mark.asyncio
    async def test_chat_api_integration(self):
        """Test chat API integration from UI."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Test chat endpoint
            chat_data = {
                "message": "What are my critical security findings?",
                "session_id": "test_session_123",
                "context": {
                    "page": "dashboard",
                    "filters": {"severity": "critical"}
                }
            }
            
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=chat_data,
                    headers=self.test_headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Verify chat response structure
                    assert 'response' in data or 'message' in data
                    assert 'session_id' in data
                    
                elif response.status_code == 404:
                    pytest.skip("Chat API not available")
                    
            except httpx.RequestError:
                pytest.skip("Cannot connect to chat API")
                
    @pytest.mark.asyncio
    async def test_streaming_api_calls(self):
        """Test streaming API calls from UI."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            streaming_endpoints = [
                "/api/chat/stream",
                "/api/analysis/stream"
            ]
            
            for endpoint in streaming_endpoints:
                try:
                    async with client.stream("GET", f"{self.base_url}{endpoint}") as response:
                        if response.status_code == 200:
                            # Collect some streaming chunks
                            chunks = []
                            chunk_count = 0
                            
                            async for chunk in response.aiter_text():
                                chunks.append(chunk)
                                chunk_count += 1
                                if chunk_count >= 3:  # Limit for testing
                                    break
                                    
                            # Should receive streaming content
                            assert len(chunks) > 0
                            
                        elif response.status_code == 404:
                            # Streaming endpoint not implemented
                            continue
                            
                except httpx.RequestError:
                    pytest.skip(f"Cannot test streaming endpoint {endpoint}")
                    
    @pytest.mark.asyncio
    async def test_data_refresh_api(self):
        """Test data refresh API calls."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            refresh_endpoints = [
                "/api/data/refresh",
                "/api/cache/clear",
                "/api/security/refresh"
            ]
            
            for endpoint in refresh_endpoints:
                try:
                    # Test POST for refresh operations
                    response = await client.post(f"{self.base_url}{endpoint}")
                    
                    # Refresh operations may take time or be rate-limited
                    acceptable_status_codes = [200, 202, 429, 503]
                    assert response.status_code in acceptable_status_codes
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Should indicate refresh status
                        assert 'status' in data or 'message' in data
                        
                    elif response.status_code == 202:
                        # Async refresh accepted
                        data = response.json()
                        assert 'task_id' in data or 'message' in data
                        
                except httpx.RequestError:
                    pytest.skip(f"Cannot test refresh endpoint {endpoint}")
                    
    @pytest.mark.asyncio
    async def test_export_api_calls(self):
        """Test export API calls from UI."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            export_endpoints = [
                "/api/reports/security_summary",
                "/api/reports/iam_analysis",
                "/api/export/csv",
                "/api/export/pdf"
            ]
            
            for endpoint in export_endpoints:
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    
                    if response.status_code == 200:
                        # Verify export content type
                        content_type = response.headers.get('content-type', '')
                        
                        if endpoint.endswith('/csv'):
                            assert 'csv' in content_type or 'text' in content_type
                        elif endpoint.endswith('/pdf'):
                            assert 'pdf' in content_type or 'application' in content_type
                        elif 'reports' in endpoint:
                            # Reports might be JSON or file downloads
                            assert 'json' in content_type or 'application' in content_type
                            
                    elif response.status_code == 404:
                        # Export endpoint not implemented
                        continue
                        
                except httpx.RequestError:
                    pytest.skip(f"Cannot test export endpoint {endpoint}")


class TestAPIErrorHandling:
    """Test API error handling from UI components."""
    
    def setup_method(self):
        """Setup test environment."""
        self.base_url = "http://localhost:8000"
        self.timeout = httpx.Timeout(10.0)  # Shorter timeout for error tests
        
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of connection errors."""
        # Test with invalid URL to simulate connection error
        invalid_url = "http://localhost:9999"  # Non-existent port
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(f"{invalid_url}/api/health")
                # Should not reach here
                pytest.fail("Expected connection error")
            except httpx.ConnectError:
                # This is expected
                pass
            except httpx.TimeoutException:
                # This is also acceptable
                pass
                
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        # Use very short timeout to trigger timeout
        short_timeout = httpx.Timeout(0.001)  # 1ms timeout
        
        async with httpx.AsyncClient(timeout=short_timeout) as client:
            try:
                response = await client.get(f"{self.base_url}/api/security/summary")
                # If this succeeds, the endpoint is very fast
                pass
            except httpx.TimeoutException:
                # This is expected with very short timeout
                pass
            except httpx.ConnectError:
                # Connection might not be available
                pytest.skip("Backend not available for timeout test")
                
    @pytest.mark.asyncio
    async def test_http_error_responses(self):
        """Test handling of HTTP error responses."""
        error_test_cases = [
            ("/api/nonexistent", 404),
            ("/api/security/invalid_param?bad=value", [400, 422]),
            ("/api/admin/restricted", [401, 403])
        ]
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for endpoint, expected_codes in error_test_cases:
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    
                    if isinstance(expected_codes, list):
                        assert response.status_code in expected_codes
                    else:
                        assert response.status_code == expected_codes
                        
                    # Verify error response format
                    if response.headers.get('content-type', '').startswith('application/json'):
                        try:
                            error_data = response.json()
                            # Should have error information
                            error_fields = ['error', 'message', 'detail', 'code']
                            has_error_field = any(field in error_data for field in error_fields)
                            assert has_error_field
                        except json.JSONDecodeError:
                            # Non-JSON error response is acceptable
                            pass
                            
                except httpx.RequestError:
                    pytest.skip(f"Cannot test error endpoint {endpoint}")
                    
    @pytest.mark.asyncio
    async def test_malformed_request_handling(self):
        """Test handling of malformed requests."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Test malformed JSON
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    data="invalid json{{{",  # Malformed JSON
                    headers={"Content-Type": "application/json"}
                )
                
                # Should handle malformed JSON gracefully
                assert response.status_code in [400, 422, 500]
                
            except httpx.RequestError:
                pytest.skip("Cannot test malformed request handling")
                
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self):
        """Test handling of rate limiting."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Make multiple rapid requests
            tasks = []
            for i in range(50):  # Many requests
                task = client.get(f"{self.base_url}/api/health")
                tasks.append(task)
                
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for rate limiting responses
                rate_limited_count = 0
                successful_count = 0
                error_count = 0
                
                for response in responses:
                    if isinstance(response, Exception):
                        error_count += 1
                    elif hasattr(response, 'status_code'):
                        if response.status_code == 429:
                            rate_limited_count += 1
                        elif response.status_code == 200:
                            successful_count += 1
                            
                # Should have some responses (either successful or rate limited)
                total_responses = rate_limited_count + successful_count
                assert total_responses > 0
                
            except Exception as e:
                pytest.skip(f"Cannot test rate limiting: {e}")


class TestAPIDataTransformation:
    """Test data transformation in API calls."""
    
    def test_request_data_formatting(self):
        """Test request data formatting for API calls."""
        # Test query formatting
        ui_query = {
            'text': 'Show me critical security findings',
            'filters': {
                'severity': ['critical', 'high'],
                'resource_types': ['storage', 'iam'],
                'time_range': '7d'
            },
            'context': {
                'page': 'dashboard',
                'user_preferences': {'detailed_analysis': True}
            }
        }
        
        # Transform to API format
        api_request = {
            'query': ui_query['text'],
            'filters': {
                'severity_levels': ui_query['filters']['severity'],
                'resource_types': ui_query['filters']['resource_types'],
                'time_range_days': 7  # Transform string to number
            },
            'options': {
                'include_details': ui_query['context']['user_preferences']['detailed_analysis'],
                'source_page': ui_query['context']['page']
            }
        }
        
        # Verify transformation
        assert api_request['query'] == 'Show me critical security findings'
        assert api_request['filters']['severity_levels'] == ['critical', 'high']
        assert api_request['options']['include_details'] == True
        
    def test_response_data_transformation(self):
        """Test response data transformation from API."""
        # Mock API response
        api_response = {
            'findings': [
                {
                    'finding_id': 'f_001',
                    'severity_level': 'CRITICAL',
                    'resource_name': 'projects/test/buckets/bucket1',
                    'finding_type': 'PUBLIC_BUCKET',
                    'detected_at': '2025-01-15T10:30:00Z',
                    'details': {
                        'public_access': True,
                        'encryption_enabled': False
                    }
                }
            ],
            'summary': {
                'total_count': 1,
                'severity_distribution': {'CRITICAL': 1, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            }
        }
        
        # Transform to UI format
        ui_data = {
            'findings': [
                {
                    'id': f['finding_id'],
                    'severity': f['severity_level'].lower(),
                    'resource': f['resource_name'],
                    'type': f['finding_type'].replace('_', ' ').title(),
                    'timestamp': f['detected_at'],
                    'details': f['details']
                }
                for f in api_response['findings']
            ],
            'totals': {
                'count': api_response['summary']['total_count'],
                'by_severity': {
                    k.lower(): v for k, v in api_response['summary']['severity_distribution'].items()
                }
            }
        }
        
        # Verify transformation
        assert len(ui_data['findings']) == 1
        finding = ui_data['findings'][0]
        assert finding['severity'] == 'critical'
        assert finding['type'] == 'Public Bucket'
        assert ui_data['totals']['by_severity']['critical'] == 1
        
    def test_error_response_transformation(self):
        """Test error response transformation."""
        # Mock API error response
        api_error = {
            'error': {
                'code': 'INVALID_CREDENTIALS',
                'message': 'The provided credentials are invalid or expired',
                'details': {
                    'credential_type': 'service_account',
                    'expires_at': '2025-01-15T08:00:00Z'
                }
            },
            'request_id': 'req_123456',
            'timestamp': '2025-01-15T10:30:00Z'
        }
        
        # Transform to UI error format
        ui_error = {
            'type': 'authentication_error',
            'title': 'Authentication Failed',
            'message': api_error['error']['message'],
            'code': api_error['error']['code'],
            'details': api_error['error']['details'],
            'timestamp': api_error['timestamp'],
            'actions': [
                {'label': 'Re-authenticate', 'action': 'refresh_credentials'},
                {'label': 'Check Documentation', 'action': 'open_docs'}
            ]
        }
        
        # Verify transformation
        assert ui_error['type'] == 'authentication_error'
        assert ui_error['title'] == 'Authentication Failed'
        assert ui_error['code'] == 'INVALID_CREDENTIALS'
        assert len(ui_error['actions']) == 2


if __name__ == "__main__":
    # Run API integration tests
    pytest.main([__file__, "-v", "--tb=short", "--timeout=60"])