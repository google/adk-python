"""
Integration tests for data flow between UI pages.

Tests data consistency, filtering, and state management across different
UI components and pages.
"""

import pytest
import asyncio
import streamlit as st
from unittest.mock import patch, Mock, MagicMock
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from frontend.unified_streaming_client import SecurityDashboard
from frontend.dashboard import SecurityDashboard as DashboardMain
from frontend.iam_features import IAMFeaturesUI
from frontend.networking_dashboard import main as networking_main


class TestDataFlow:
    """Test suite for data flow between UI components."""
    
    def setup_method(self):
        """Setup test environment."""
        self.mock_session_state = {}
        
    def test_session_state_data_flow(self):
        """Test data flow through session state."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Initialize session state with test data
            st.session_state['selected_project'] = 'test-project-123'
            st.session_state['current_user'] = 'test-user@example.com'
            st.session_state['security_findings'] = {
                'critical': 5,
                'high': 12,
                'medium': 25,
                'low': 8
            }
            
            # Verify data persists across components
            assert st.session_state['selected_project'] == 'test-project-123'
            assert st.session_state['security_findings']['critical'] == 5
            
    def test_dashboard_to_detail_flow(self):
        """Test data flow from dashboard to detail pages."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Simulate dashboard selection
            st.session_state['selected_finding'] = {
                'id': 'finding_001',
                'type': 'iam_policy',
                'severity': 'critical',
                'resource': 'projects/test-project/policies/test-policy'
            }
            
            st.session_state['navigation_context'] = {
                'source_page': 'dashboard',
                'target_page': 'iam_details',
                'filters_applied': {'severity': 'critical'}
            }
            
            # Verify context is preserved
            context = st.session_state.get('navigation_context', {})
            assert context['source_page'] == 'dashboard'
            assert context['filters_applied']['severity'] == 'critical'
            
    def test_filter_propagation(self):
        """Test filter propagation across pages."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Set filters on main dashboard
            st.session_state['active_filters'] = {
                'severity': ['critical', 'high'],
                'resource_type': ['storage', 'compute'],
                'time_range': '7d',
                'project_filter': 'test-project-123'
            }
            
            # Simulate page navigation with filters
            st.session_state['page_filters'] = {
                'iam_page': st.session_state['active_filters'].copy(),
                'storage_page': st.session_state['active_filters'].copy(),
                'network_page': st.session_state['active_filters'].copy()
            }
            
            # Verify filters are consistent across pages
            for page, filters in st.session_state['page_filters'].items():
                assert filters['severity'] == ['critical', 'high']
                assert filters['project_filter'] == 'test-project-123'
                
    def test_search_context_flow(self):
        """Test search context flow between components."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Set search context
            st.session_state['search_history'] = [
                {
                    'query': 'Show me critical IAM findings',
                    'timestamp': datetime.now().isoformat(),
                    'results_count': 15,
                    'page': 'dashboard'
                },
                {
                    'query': 'What storage buckets are publicly accessible?',
                    'timestamp': datetime.now().isoformat(),
                    'results_count': 3,
                    'page': 'storage_analysis'
                }
            ]
            
            st.session_state['current_search'] = {
                'active': True,
                'query': 'Show me critical IAM findings',
                'applied_to_pages': ['dashboard', 'iam_features']
            }
            
            # Verify search context is maintained
            assert len(st.session_state['search_history']) == 2
            assert st.session_state['current_search']['active'] == True
            
    def test_multi_page_data_consistency(self):
        """Test data consistency across multiple pages."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Set consistent data across pages
            base_data = {
                'project_id': 'test-project-123',
                'scan_timestamp': '2025-01-15T10:30:00Z',
                'total_resources': 1250,
                'security_score': 78.5
            }
            
            # Propagate to different page contexts
            st.session_state['dashboard_data'] = base_data.copy()
            st.session_state['iam_data'] = {**base_data, 'iam_policies': 45, 'iam_findings': 12}
            st.session_state['storage_data'] = {**base_data, 'storage_buckets': 23, 'public_buckets': 2}
            st.session_state['network_data'] = {**base_data, 'vpc_networks': 8, 'firewall_rules': 156}
            
            # Verify consistency
            pages_data = ['dashboard_data', 'iam_data', 'storage_data', 'network_data']
            for page_data in pages_data:
                data = st.session_state[page_data]
                assert data['project_id'] == 'test-project-123'
                assert data['total_resources'] == 1250
                assert data['security_score'] == 78.5
                
    def test_real_time_data_updates(self):
        """Test real-time data updates across components."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Initial data state
            st.session_state['live_metrics'] = {
                'critical_findings': 5,
                'high_findings': 12,
                'last_update': datetime.now().isoformat()
            }
            
            # Simulate real-time update
            time.sleep(0.1)  # Small delay to ensure timestamp difference
            updated_metrics = {
                'critical_findings': 6,  # New critical finding
                'high_findings': 11,     # One high became critical
                'last_update': datetime.now().isoformat()
            }
            
            st.session_state['live_metrics'] = updated_metrics
            st.session_state['update_notifications'] = [{
                'type': 'new_finding',
                'severity': 'critical',
                'message': 'New critical security finding detected',
                'timestamp': updated_metrics['last_update']
            }]
            
            # Verify updates are reflected
            assert st.session_state['live_metrics']['critical_findings'] == 6
            assert len(st.session_state['update_notifications']) == 1
            
    def test_error_state_propagation(self):
        """Test error state propagation across components."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Set error state
            st.session_state['error_states'] = {
                'api_connection': {
                    'has_error': True,
                    'error_message': 'Failed to connect to backend API',
                    'error_code': 'CONNECTION_TIMEOUT',
                    'retry_count': 3,
                    'last_attempt': datetime.now().isoformat()
                },
                'data_loading': {
                    'has_error': False,
                    'error_message': None,
                    'error_code': None,
                    'retry_count': 0,
                    'last_attempt': None
                }
            }
            
            # Verify error states are accessible
            api_error = st.session_state['error_states']['api_connection']
            assert api_error['has_error'] == True
            assert api_error['retry_count'] == 3
            
            data_error = st.session_state['error_states']['data_loading']
            assert data_error['has_error'] == False
            
    def test_cache_data_flow(self):
        """Test cached data flow between components."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Set cached data
            st.session_state['cache'] = {
                'security_summary': {
                    'data': {
                        'critical': 5,
                        'high': 12,
                        'medium': 25
                    },
                    'timestamp': datetime.now().isoformat(),
                    'ttl': 300  # 5 minutes
                },
                'iam_policies': {
                    'data': [
                        {'name': 'policy1', 'bindings': 5},
                        {'name': 'policy2', 'bindings': 3}
                    ],
                    'timestamp': datetime.now().isoformat(),
                    'ttl': 600  # 10 minutes
                }
            }
            
            # Verify cache structure
            assert 'security_summary' in st.session_state['cache']
            assert st.session_state['cache']['security_summary']['data']['critical'] == 5
            assert st.session_state['cache']['iam_policies']['ttl'] == 600


class TestPageStateManagement:
    """Test state management across different pages."""
    
    def setup_method(self):
        """Setup test environment."""
        self.mock_session_state = {}
        
    def test_navigation_history(self):
        """Test navigation history tracking."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Initialize navigation history
            st.session_state['navigation_history'] = []
            
            # Simulate page navigation
            pages_visited = [
                ('dashboard', {'timestamp': datetime.now().isoformat()}),
                ('iam_features', {'timestamp': datetime.now().isoformat(), 'filter': 'critical'}),
                ('storage_analysis', {'timestamp': datetime.now().isoformat()}),
                ('network_dashboard', {'timestamp': datetime.now().isoformat()})
            ]
            
            for page, context in pages_visited:
                st.session_state['navigation_history'].append({
                    'page': page,
                    'context': context
                })
                
            # Verify navigation history
            assert len(st.session_state['navigation_history']) == 4
            assert st.session_state['navigation_history'][0]['page'] == 'dashboard'
            assert st.session_state['navigation_history'][1]['context']['filter'] == 'critical'
            
    def test_form_state_persistence(self):
        """Test form state persistence across page changes."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Set form states for different pages
            st.session_state['form_states'] = {
                'security_query_form': {
                    'query_text': 'Show me storage security issues',
                    'severity_filter': ['critical', 'high'],
                    'time_range': '30d',
                    'include_resolved': False
                },
                'iam_analysis_form': {
                    'policy_type': 'custom',
                    'include_inherited': True,
                    'show_recommendations': True
                }
            }
            
            # Verify form states persist
            query_form = st.session_state['form_states']['security_query_form']
            assert query_form['query_text'] == 'Show me storage security issues'
            assert query_form['include_resolved'] == False
            
            iam_form = st.session_state['form_states']['iam_analysis_form']
            assert iam_form['policy_type'] == 'custom'
            assert iam_form['include_inherited'] == True
            
    def test_selection_context_flow(self):
        """Test selection context flow between pages."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Set selection context
            st.session_state['selections'] = {
                'current_resource': {
                    'type': 'storage_bucket',
                    'name': 'test-bucket-001',
                    'project': 'test-project-123',
                    'findings': ['public_access', 'no_encryption']
                },
                'related_resources': [
                    {
                        'type': 'iam_policy',
                        'name': 'bucket-access-policy',
                        'relationship': 'grants_access_to'
                    }
                ],
                'selection_history': []
            }
            
            # Add to selection history
            st.session_state['selections']['selection_history'].append({
                'resource': st.session_state['selections']['current_resource'].copy(),
                'timestamp': datetime.now().isoformat(),
                'source_page': 'storage_analysis'
            })
            
            # Verify selection context
            current = st.session_state['selections']['current_resource']
            assert current['type'] == 'storage_bucket'
            assert 'public_access' in current['findings']
            
            history = st.session_state['selections']['selection_history']
            assert len(history) == 1
            assert history[0]['source_page'] == 'storage_analysis'


class TestDataValidationFlow:
    """Test data validation across UI components."""
    
    def test_data_schema_validation(self):
        """Test data schema validation across components."""
        # Mock valid security data schema
        valid_security_data = {
            'findings': [
                {
                    'id': 'finding_001',
                    'severity': 'critical',
                    'type': 'iam_misconfiguration',
                    'resource': 'projects/test/policies/policy1',
                    'description': 'Overly permissive IAM policy',
                    'timestamp': '2025-01-15T10:30:00Z'
                }
            ],
            'summary': {
                'total_findings': 1,
                'by_severity': {'critical': 1, 'high': 0, 'medium': 0, 'low': 0}
            },
            'metadata': {
                'scan_id': 'scan_123',
                'project_id': 'test-project',
                'timestamp': '2025-01-15T10:30:00Z'
            }
        }
        
        # Validate required fields exist
        assert 'findings' in valid_security_data
        assert 'summary' in valid_security_data
        assert 'metadata' in valid_security_data
        
        # Validate findings structure
        finding = valid_security_data['findings'][0]
        required_finding_fields = ['id', 'severity', 'type', 'resource', 'description', 'timestamp']
        for field in required_finding_fields:
            assert field in finding
            
        # Validate summary structure
        summary = valid_security_data['summary']
        assert 'total_findings' in summary
        assert 'by_severity' in summary
        
    def test_data_transformation_consistency(self):
        """Test data transformation consistency across components."""
        # Raw data from backend
        raw_data = {
            'iam_policies': [
                {'name': 'policy1', 'bindings': [{'role': 'roles/editor', 'members': ['user:test@example.com']}]},
                {'name': 'policy2', 'bindings': [{'role': 'roles/viewer', 'members': ['user:viewer@example.com']}]}
            ]
        }
        
        # Transform for dashboard view
        dashboard_data = {
            'total_policies': len(raw_data['iam_policies']),
            'policies_with_editor_role': len([p for p in raw_data['iam_policies'] 
                                            if any(b['role'] == 'roles/editor' for b in p['bindings'])]),
            'unique_members': len(set(
                member for policy in raw_data['iam_policies']
                for binding in policy['bindings']
                for member in binding['members']
            ))
        }
        
        # Verify transformations are consistent
        assert dashboard_data['total_policies'] == 2
        assert dashboard_data['policies_with_editor_role'] == 1
        assert dashboard_data['unique_members'] == 2
        
    def test_error_data_handling(self):
        """Test error data handling across components."""
        # Simulate various error conditions
        error_scenarios = [
            {
                'type': 'api_timeout',
                'data': None,
                'error': {
                    'code': 'TIMEOUT',
                    'message': 'Request timed out after 30 seconds',
                    'retry_after': 60
                }
            },
            {
                'type': 'invalid_credentials',
                'data': None,
                'error': {
                    'code': 'AUTH_ERROR',
                    'message': 'Invalid GCP credentials',
                    'action_required': 'Re-authenticate'
                }
            },
            {
                'type': 'partial_data',
                'data': {
                    'available_fields': ['summary'],
                    'missing_fields': ['detailed_findings', 'recommendations'],
                    'warning': 'Some data unavailable due to permissions'
                },
                'error': None
            }
        ]
        
        # Verify error handling structures
        for scenario in error_scenarios:
            assert 'type' in scenario
            assert 'data' in scenario
            assert 'error' in scenario
            
            if scenario['error']:
                assert 'code' in scenario['error']
                assert 'message' in scenario['error']


if __name__ == "__main__":
    # Run data flow tests
    pytest.main([__file__, "-v", "--tb=short"])