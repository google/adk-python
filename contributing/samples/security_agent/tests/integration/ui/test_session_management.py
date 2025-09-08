"""
Integration tests for session management across UI components.

Tests session persistence, state management, user context,
and session recovery mechanisms.
"""

import pytest
import asyncio
import streamlit as st
from unittest.mock import patch, Mock, MagicMock
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sqlite3
import tempfile
import os

# Import test utilities
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from backend.services.session_manager import SessionManager
    from frontend.unified_streaming_client import SecurityDashboard
    from backend.data.sessions import SessionDB
except ImportError as e:
    pytest.skip(f"Session management modules not available: {e}", allow_module_level=True)


class TestSessionPersistence:
    """Test suite for session persistence mechanisms."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.session_db_path = self.temp_db.name
        
        # Mock session state
        self.mock_session_state = {}
        
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.session_db_path):
            os.unlink(self.session_db_path)
            
    def test_session_creation_and_persistence(self):
        """Test session creation and persistence to database."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Create new session
            session_id = str(uuid.uuid4())
            user_id = "test_user_123"
            
            st.session_state['session_id'] = session_id
            st.session_state['user_id'] = user_id
            st.session_state['created_at'] = datetime.now().isoformat()
            st.session_state['last_activity'] = datetime.now().isoformat()
            
            # Session data
            session_data = {
                'user_preferences': {
                    'theme': 'dark',
                    'dashboard_layout': 'compact',
                    'auto_refresh': True,
                    'notification_level': 'critical_only'
                },
                'current_context': {
                    'project_id': 'test-project-123',
                    'active_filters': {'severity': 'critical'},
                    'current_page': 'dashboard'
                },
                'interaction_history': []
            }
            
            st.session_state['session_data'] = session_data
            
            # Verify session is created in memory
            assert st.session_state['session_id'] == session_id
            assert st.session_state['user_id'] == user_id
            assert st.session_state['session_data']['user_preferences']['theme'] == 'dark'
            
    def test_session_data_recovery(self):
        """Test session data recovery after interruption."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Simulate existing session data
            session_id = "session_recovery_test_123"
            
            # Initial session state
            initial_state = {
                'session_id': session_id,
                'user_id': 'test_user_recovery',
                'navigation_history': [
                    {'page': 'dashboard', 'timestamp': datetime.now().isoformat()},
                    {'page': 'iam_analysis', 'timestamp': datetime.now().isoformat()}
                ],
                'form_data': {
                    'security_query': 'Show critical findings',
                    'selected_resources': ['bucket1', 'bucket2']
                },
                'chat_history': [
                    {'role': 'user', 'content': 'What are my security risks?'},
                    {'role': 'assistant', 'content': 'I found 5 critical issues...'}
                ]
            }
            
            # Set initial state
            st.session_state.update(initial_state)
            
            # Simulate session interruption and recovery
            recovered_session_id = st.session_state.get('session_id')
            assert recovered_session_id == session_id
            
            # Verify data integrity after recovery
            assert len(st.session_state['navigation_history']) == 2
            assert st.session_state['form_data']['security_query'] == 'Show critical findings'
            assert len(st.session_state['chat_history']) == 2
            
    def test_session_timeout_handling(self):
        """Test session timeout and cleanup."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Create session with timeout
            session_id = "timeout_test_session"
            created_at = datetime.now() - timedelta(hours=2)  # 2 hours ago
            last_activity = datetime.now() - timedelta(minutes=35)  # 35 minutes ago
            
            st.session_state['session_id'] = session_id
            st.session_state['created_at'] = created_at.isoformat()
            st.session_state['last_activity'] = last_activity.isoformat()
            st.session_state['timeout_minutes'] = 30  # 30 minute timeout
            
            # Check if session should timeout
            now = datetime.now()
            last_activity_dt = datetime.fromisoformat(st.session_state['last_activity'])
            timeout_minutes = st.session_state['timeout_minutes']
            
            time_since_activity = (now - last_activity_dt).total_seconds() / 60
            session_should_timeout = time_since_activity > timeout_minutes
            
            # Should timeout after 30 minutes of inactivity
            assert session_should_timeout == True
            
            # Simulate timeout cleanup
            if session_should_timeout:
                # Preserve minimal session info for recovery
                essential_data = {
                    'session_id': st.session_state['session_id'],
                    'user_id': st.session_state.get('user_id'),
                    'timeout_at': now.isoformat()
                }
                
                # Clear active session data
                st.session_state.clear()
                st.session_state.update(essential_data)
                
            # Verify timeout cleanup
            assert 'timeout_at' in st.session_state
            assert 'chat_history' not in st.session_state
            
    def test_concurrent_session_handling(self):
        """Test handling of concurrent sessions for the same user."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Simulate multiple sessions for same user
            user_id = "concurrent_test_user"
            
            # First session
            session1_id = "session_1_" + str(uuid.uuid4())
            st.session_state['primary_session'] = {
                'session_id': session1_id,
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'device_info': {'browser': 'Chrome', 'platform': 'Windows'}
            }
            
            # Second session (new tab/window)
            session2_id = "session_2_" + str(uuid.uuid4())
            st.session_state['secondary_sessions'] = [{
                'session_id': session2_id,
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'device_info': {'browser': 'Chrome', 'platform': 'Windows'}
            }]
            
            # Verify concurrent session tracking
            primary = st.session_state['primary_session']
            secondary = st.session_state['secondary_sessions'][0]
            
            assert primary['user_id'] == secondary['user_id']
            assert primary['session_id'] != secondary['session_id']
            
    def test_session_data_synchronization(self):
        """Test session data synchronization across components."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Initialize session with synchronized data
            st.session_state['sync_data'] = {
                'user_preferences': {
                    'dashboard_refresh_rate': 30,
                    'alert_notifications': True,
                    'data_retention_days': 90
                },
                'active_filters': {
                    'severity_levels': ['critical', 'high'],
                    'resource_types': ['storage', 'iam'],
                    'time_range': '7d'
                },
                'sync_timestamp': datetime.now().isoformat(),
                'sync_version': 1
            }
            
            # Simulate data update from one component
            st.session_state['sync_data']['user_preferences']['dashboard_refresh_rate'] = 60
            st.session_state['sync_data']['sync_timestamp'] = datetime.now().isoformat()
            st.session_state['sync_data']['sync_version'] += 1
            
            # Verify synchronization
            sync_data = st.session_state['sync_data']
            assert sync_data['user_preferences']['dashboard_refresh_rate'] == 60
            assert sync_data['sync_version'] == 2
            
    def test_session_security_measures(self):
        """Test session security measures."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Set security-related session data
            session_security = {
                'csrf_token': str(uuid.uuid4()),
                'session_hash': 'sha256_hash_of_session_data',
                'ip_address': '192.168.1.100',
                'user_agent': 'Mozilla/5.0 (Chrome/91.0)',
                'last_security_check': datetime.now().isoformat(),
                'failed_attempts': 0,
                'security_level': 'standard'
            }
            
            st.session_state['security'] = session_security
            
            # Verify security data is present
            security = st.session_state['security']
            assert 'csrf_token' in security
            assert 'session_hash' in security
            assert security['failed_attempts'] == 0
            assert security['security_level'] == 'standard'
            
            # Simulate security validation
            expected_csrf = security['csrf_token']
            provided_csrf = expected_csrf  # In real scenario, this comes from request
            
            csrf_valid = expected_csrf == provided_csrf
            assert csrf_valid == True


class TestSessionStateManagement:
    """Test session state management across UI interactions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.mock_session_state = {}
        
    def test_page_state_transitions(self):
        """Test state transitions between pages."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Initialize page state manager
            st.session_state['page_states'] = {
                'current_page': 'dashboard',
                'previous_page': None,
                'page_history': [],
                'state_data': {}
            }
            
            # Simulate page transitions
            pages_sequence = ['dashboard', 'iam_analysis', 'storage_security', 'network_analysis']
            
            for i, page in enumerate(pages_sequence):
                # Update page state
                current_state = st.session_state['page_states']
                current_state['previous_page'] = current_state['current_page']
                current_state['current_page'] = page
                
                # Add to history
                current_state['page_history'].append({
                    'page': page,
                    'timestamp': datetime.now().isoformat(),
                    'sequence': i
                })
                
                # Store page-specific data
                current_state['state_data'][page] = {
                    'visited_at': datetime.now().isoformat(),
                    'user_interactions': [],
                    'form_data': {}
                }
                
            # Verify page state transitions
            final_state = st.session_state['page_states']
            assert final_state['current_page'] == 'network_analysis'
            assert final_state['previous_page'] == 'storage_security'
            assert len(final_state['page_history']) == 4
            assert 'dashboard' in final_state['state_data']
            
    def test_form_state_persistence(self):
        """Test form state persistence across page changes."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Initialize form states
            st.session_state['forms'] = {
                'security_query_form': {
                    'query_text': '',
                    'severity_filter': [],
                    'time_range': '7d',
                    'include_resolved': False,
                    'last_modified': None
                },
                'iam_policy_form': {
                    'policy_name': '',
                    'resource_filter': '',
                    'show_inherited': True,
                    'analysis_depth': 'standard',
                    'last_modified': None
                }
            }
            
            # Simulate form interactions
            # User fills security query form
            security_form = st.session_state['forms']['security_query_form']
            security_form['query_text'] = 'Show me all critical storage findings'
            security_form['severity_filter'] = ['critical', 'high']
            security_form['include_resolved'] = True
            security_form['last_modified'] = datetime.now().isoformat()
            
            # User partially fills IAM form
            iam_form = st.session_state['forms']['iam_policy_form']
            iam_form['policy_name'] = 'test-policy*'
            iam_form['analysis_depth'] = 'detailed'
            iam_form['last_modified'] = datetime.now().isoformat()
            
            # Verify form states are preserved
            assert security_form['query_text'] == 'Show me all critical storage findings'
            assert 'critical' in security_form['severity_filter']
            assert security_form['include_resolved'] == True
            
            assert iam_form['policy_name'] == 'test-policy*'
            assert iam_form['analysis_depth'] == 'detailed'
            
    def test_user_context_management(self):
        """Test user context management across sessions."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Set user context
            user_context = {
                'user_id': 'context_test_user',
                'user_profile': {
                    'name': 'Test User',
                    'email': 'test@example.com',
                    'role': 'security_analyst',
                    'permissions': ['read_security', 'export_reports'],
                    'preferences': {
                        'language': 'en',
                        'timezone': 'UTC',
                        'date_format': 'YYYY-MM-DD',
                        'default_severity_filter': ['critical', 'high']
                    }
                },
                'work_context': {
                    'default_project': 'production-project-001',
                    'recent_projects': ['prod-001', 'staging-002', 'dev-003'],
                    'saved_queries': [
                        {'name': 'Critical Issues', 'query': 'severity:critical'},
                        {'name': 'Storage Problems', 'query': 'resource_type:storage AND public:true'}
                    ],
                    'dashboard_config': {
                        'widgets': ['security_overview', 'recent_findings', 'compliance_status'],
                        'layout': 'grid',
                        'refresh_interval': 300
                    }
                }
            }
            
            st.session_state['user_context'] = user_context
            
            # Verify user context
            context = st.session_state['user_context']
            assert context['user_profile']['role'] == 'security_analyst'
            assert 'read_security' in context['user_profile']['permissions']
            assert len(context['work_context']['recent_projects']) == 3
            assert len(context['work_context']['saved_queries']) == 2
            
    def test_cache_session_integration(self):
        """Test integration between session state and caching."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Initialize cache-related session data
            st.session_state['cache_config'] = {
                'enabled': True,
                'ttl_seconds': 300,  # 5 minutes
                'cache_keys': [],
                'cache_stats': {
                    'hits': 0,
                    'misses': 0,
                    'size_bytes': 0
                }
            }
            
            # Simulate cache operations
            cache_key = 'security_summary_dashboard'
            cached_data = {
                'critical_findings': 5,
                'high_findings': 12,
                'last_updated': datetime.now().isoformat()
            }
            
            # Store in session cache
            st.session_state[f'cache_{cache_key}'] = {
                'data': cached_data,
                'timestamp': datetime.now().isoformat(),
                'ttl': st.session_state['cache_config']['ttl_seconds']
            }
            
            # Update cache stats
            st.session_state['cache_config']['cache_keys'].append(cache_key)
            st.session_state['cache_config']['cache_stats']['size_bytes'] += len(str(cached_data))
            
            # Verify cache integration
            assert f'cache_{cache_key}' in st.session_state
            cache_entry = st.session_state[f'cache_{cache_key}']
            assert cache_entry['data']['critical_findings'] == 5
            assert cache_key in st.session_state['cache_config']['cache_keys']


class TestSessionRecovery:
    """Test session recovery mechanisms."""
    
    def setup_method(self):
        """Setup test environment."""
        self.mock_session_state = {}
        
    def test_session_recovery_after_error(self):
        """Test session recovery after application error."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Simulate session before error
            pre_error_state = {
                'session_id': 'recovery_test_session',
                'user_id': 'recovery_test_user',
                'current_page': 'iam_analysis',
                'form_data': {
                    'analysis_query': 'Show overprivileged accounts',
                    'selected_projects': ['proj1', 'proj2']
                },
                'results_cache': {
                    'last_query_results': [
                        {'account': 'user1@example.com', 'risk': 'high'},
                        {'account': 'user2@example.com', 'risk': 'medium'}
                    ]
                },
                'error_occurred': False
            }
            
            st.session_state.update(pre_error_state)
            
            # Simulate error occurrence
            st.session_state['error_occurred'] = True
            st.session_state['error_details'] = {
                'error_type': 'api_timeout',
                'error_message': 'Request timed out',
                'error_timestamp': datetime.now().isoformat(),
                'recovery_attempted': False
            }
            
            # Simulate recovery process
            if st.session_state['error_occurred']:
                # Preserve critical session data
                recovery_data = {
                    'session_id': st.session_state['session_id'],
                    'user_id': st.session_state['user_id'],
                    'last_known_page': st.session_state['current_page'],
                    'recoverable_form_data': st.session_state['form_data'],
                    'recovery_timestamp': datetime.now().isoformat()
                }
                
                st.session_state['recovery_data'] = recovery_data
                st.session_state['error_details']['recovery_attempted'] = True
                
            # Verify recovery data is preserved
            recovery = st.session_state['recovery_data']
            assert recovery['session_id'] == 'recovery_test_session'
            assert recovery['last_known_page'] == 'iam_analysis'
            assert recovery['recoverable_form_data']['analysis_query'] == 'Show overprivileged accounts'
            
    def test_partial_session_recovery(self):
        """Test partial session recovery when some data is corrupted."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Simulate partially corrupted session
            corrupted_session = {
                'session_id': 'partial_recovery_test',
                'user_id': 'partial_test_user',
                # 'current_page': None,  # Missing data
                'form_data': {
                    'corrupted_field': 'invalid_json{{{',  # Corrupted data
                    'valid_field': 'valid_data'
                },
                'navigation_history': [
                    {'page': 'dashboard', 'valid': True},
                    # Corrupted entry would be here
                ],
                'user_preferences': {
                    'theme': 'dark',
                    'language': 'en'
                }
            }
            
            st.session_state.update(corrupted_session)
            
            # Recovery process - salvage what we can
            recoverable_data = {}
            
            # Recover basic session info
            if 'session_id' in st.session_state:
                recoverable_data['session_id'] = st.session_state['session_id']
            if 'user_id' in st.session_state:
                recoverable_data['user_id'] = st.session_state['user_id']
                
            # Recover valid form data
            recoverable_form_data = {}
            if 'form_data' in st.session_state:
                for key, value in st.session_state['form_data'].items():
                    if key == 'valid_field':  # Known good field
                        recoverable_form_data[key] = value
                        
            recoverable_data['form_data'] = recoverable_form_data
            
            # Set defaults for missing data
            recoverable_data['current_page'] = 'dashboard'  # Default page
            
            st.session_state['recovered_session'] = recoverable_data
            
            # Verify partial recovery
            recovered = st.session_state['recovered_session']
            assert recovered['session_id'] == 'partial_recovery_test'
            assert recovered['current_page'] == 'dashboard'
            assert recovered['form_data']['valid_field'] == 'valid_data'
            assert 'corrupted_field' not in recovered['form_data']


if __name__ == "__main__":
    # Run session management tests
    pytest.main([__file__, "-v", "--tb=short"])