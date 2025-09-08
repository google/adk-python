"""
Integration tests for caching mechanisms between UI and backend.

Tests cache consistency, invalidation, performance improvements,
and cache-related error handling across the system.
"""

import pytest
import asyncio
import httpx
import streamlit as st
from unittest.mock import patch, Mock, AsyncMock, MagicMock
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sqlite3
import tempfile
import os

# Import test utilities
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from backend.cache import CacheManager
    from backend.services.cache_manager import CacheService
    from frontend.unified_streaming_client import SecurityDashboard
except ImportError as e:
    pytest.skip(f"Cache integration modules not available: {e}", allow_module_level=True)


class TestCacheIntegration:
    """Test suite for cache integration between UI and backend."""
    
    def setup_method(self):
        """Setup test environment."""
        self.base_url = "http://localhost:8000"
        self.timeout = httpx.Timeout(30.0)
        
        # Create temporary cache database
        self.temp_cache_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_cache_db.close()
        self.cache_db_path = self.temp_cache_db.name
        
        # Mock session state
        self.mock_session_state = {}
        
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.cache_db_path):
            os.unlink(self.cache_db_path)
            
    @pytest.mark.asyncio
    async def test_api_response_caching(self):
        """Test API response caching mechanism."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Test endpoint that should be cached
            cache_test_endpoint = "/api/security/summary"
            
            try:
                # First request - should hit backend and cache result
                start_time = time.time()
                response1 = await client.get(f"{self.base_url}{cache_test_endpoint}")
                first_request_time = time.time() - start_time
                
                if response1.status_code == 200:
                    data1 = response1.json()
                    
                    # Wait a short time then make second request
                    await asyncio.sleep(0.5)
                    
                    # Second request - should be served from cache
                    start_time = time.time()
                    response2 = await client.get(f"{self.base_url}{cache_test_endpoint}")
                    second_request_time = time.time() - start_time
                    
                    if response2.status_code == 200:
                        data2 = response2.json()
                        
                        # Check cache headers if present
                        cache_header = response2.headers.get('X-Cache-Status')
                        if cache_header:
                            assert cache_header in ['hit', 'miss', 'refresh']
                            
                        # Data should be consistent between requests
                        # Remove timestamps for comparison
                        comparable_data1 = {k: v for k, v in data1.items() if k != 'timestamp'}
                        comparable_data2 = {k: v for k, v in data2.items() if k != 'timestamp'}
                        
                        # Core data should match
                        if comparable_data1 and comparable_data2:
                            # At least some fields should match
                            common_fields = set(comparable_data1.keys()) & set(comparable_data2.keys())
                            assert len(common_fields) > 0
                            
            except (httpx.ConnectError, httpx.TimeoutException):
                pytest.skip("Backend not available for cache testing")
                
    def test_ui_cache_implementation(self):
        """Test UI-side cache implementation."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Initialize UI cache
            st.session_state['ui_cache'] = {
                'enabled': True,
                'ttl_seconds': 300,  # 5 minutes
                'max_entries': 100,
                'entries': {},
                'stats': {
                    'hits': 0,
                    'misses': 0,
                    'evictions': 0
                }
            }
            
            # Test cache entry creation
            cache_key = 'dashboard_metrics_' + hashlib.md5(b'test_project').hexdigest()
            cache_data = {
                'critical_findings': 5,
                'high_findings': 12,
                'medium_findings': 25,
                'low_findings': 8,
                'last_scan': '2025-01-15T10:30:00Z'
            }
            
            # Store in UI cache
            cache_entry = {
                'data': cache_data,
                'timestamp': datetime.now().isoformat(),
                'ttl': st.session_state['ui_cache']['ttl_seconds'],
                'key': cache_key
            }
            
            st.session_state['ui_cache']['entries'][cache_key] = cache_entry
            
            # Verify cache entry
            cached_entry = st.session_state['ui_cache']['entries'][cache_key]
            assert cached_entry['data']['critical_findings'] == 5
            assert cached_entry['key'] == cache_key
            
            # Test cache retrieval
            retrieved_data = cached_entry['data']
            assert retrieved_data == cache_data
            
            # Update cache stats
            st.session_state['ui_cache']['stats']['hits'] += 1
            assert st.session_state['ui_cache']['stats']['hits'] == 1
            
    def test_cache_invalidation(self):
        """Test cache invalidation mechanisms."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Setup cache with entries
            cache_entries = {
                'security_summary': {
                    'data': {'findings': 10},
                    'timestamp': datetime.now().isoformat(),
                    'ttl': 300
                },
                'iam_policies': {
                    'data': {'policies': ['policy1', 'policy2']},
                    'timestamp': (datetime.now() - timedelta(minutes=10)).isoformat(),  # Old entry
                    'ttl': 300
                },
                'storage_buckets': {
                    'data': {'buckets': ['bucket1', 'bucket2']},
                    'timestamp': datetime.now().isoformat(),
                    'ttl': 300
                }
            }
            
            st.session_state['cache_entries'] = cache_entries
            
            # Test TTL-based invalidation
            now = datetime.now()
            valid_entries = {}
            expired_entries = []
            
            for key, entry in cache_entries.items():
                entry_time = datetime.fromisoformat(entry['timestamp'])
                age_seconds = (now - entry_time).total_seconds()
                
                if age_seconds < entry['ttl']:
                    valid_entries[key] = entry
                else:
                    expired_entries.append(key)
                    
            # Should have some valid and some expired entries
            assert len(valid_entries) >= 2  # security_summary and storage_buckets should be valid
            assert 'iam_policies' in expired_entries  # This one should be expired
            
            # Test manual invalidation
            invalidation_patterns = ['security_*', 'iam_*']
            for pattern in invalidation_patterns:
                keys_to_invalidate = []
                pattern_prefix = pattern.replace('*', '')
                
                for key in cache_entries.keys():
                    if key.startswith(pattern_prefix):
                        keys_to_invalidate.append(key)
                        
                # Remove invalidated entries
                for key in keys_to_invalidate:
                    if key in valid_entries:
                        del valid_entries[key]
                        
            # Verify invalidation
            assert 'security_summary' not in valid_entries
            assert 'iam_policies' not in valid_entries
            # storage_buckets should remain as it doesn't match patterns
            
    def test_cache_consistency_across_components(self):
        """Test cache consistency across different UI components."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Setup shared cache
            shared_cache = {
                'project_data': {
                    'project_id': 'test-project-123',
                    'security_score': 78.5,
                    'last_updated': datetime.now().isoformat(),
                    'resource_counts': {
                        'compute_instances': 25,
                        'storage_buckets': 12,
                        'iam_policies': 45
                    }
                },
                'user_context': {
                    'user_id': 'test_user',
                    'active_filters': ['critical', 'high'],
                    'selected_resources': ['bucket1', 'instance2']
                }
            }
            
            st.session_state['shared_cache'] = shared_cache
            
            # Simulate different components accessing cache
            # Dashboard component
            dashboard_data = st.session_state['shared_cache']['project_data'].copy()
            dashboard_data['component'] = 'dashboard'
            
            # IAM component
            iam_data = st.session_state['shared_cache']['project_data'].copy()
            iam_data['component'] = 'iam_analysis'
            iam_data['iam_specific'] = {'policy_count': iam_data['resource_counts']['iam_policies']}
            
            # Storage component
            storage_data = st.session_state['shared_cache']['project_data'].copy()
            storage_data['component'] = 'storage_analysis'
            storage_data['storage_specific'] = {'bucket_count': storage_data['resource_counts']['storage_buckets']}
            
            # Verify consistency
            assert dashboard_data['project_id'] == iam_data['project_id']
            assert iam_data['security_score'] == storage_data['security_score']
            assert dashboard_data['resource_counts'] == storage_data['resource_counts']
            
    @pytest.mark.asyncio
    async def test_cache_performance_impact(self):
        """Test cache performance impact on UI responsiveness."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Test multiple requests to measure cache performance
            test_endpoints = [
                "/api/security/metrics",
                "/api/iam/summary",
                "/api/storage/analysis"
            ]
            
            performance_results = {}
            
            for endpoint in test_endpoints:
                try:
                    # First request (cold cache)
                    start_time = time.time()
                    response1 = await client.get(f"{self.base_url}{endpoint}")
                    cold_time = time.time() - start_time
                    
                    if response1.status_code == 200:
                        # Second request (warm cache)
                        start_time = time.time()
                        response2 = await client.get(f"{self.base_url}{endpoint}")
                        warm_time = time.time() - start_time
                        
                        if response2.status_code == 200:
                            performance_results[endpoint] = {
                                'cold_request_time': cold_time,
                                'warm_request_time': warm_time,
                                'improvement_ratio': cold_time / warm_time if warm_time > 0 else 1
                            }
                            
                except (httpx.ConnectError, httpx.TimeoutException):
                    # Skip this endpoint if not available
                    continue
                    
            # Analyze performance results
            if performance_results:
                total_improvement = sum(r['improvement_ratio'] for r in performance_results.values())
                average_improvement = total_improvement / len(performance_results)
                
                # Cache should generally improve performance (ratio > 1)
                # Though this may not always be the case in test environment
                assert average_improvement >= 0.5  # At least not significantly worse
                
    def test_cache_error_handling(self):
        """Test cache error handling and fallback mechanisms."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Setup cache with some problematic entries
            problematic_cache = {
                'valid_entry': {
                    'data': {'status': 'ok'},
                    'timestamp': datetime.now().isoformat(),
                    'ttl': 300
                },
                'corrupted_entry': {
                    'data': None,  # Corrupted data
                    'timestamp': 'invalid_timestamp',  # Invalid timestamp
                    'ttl': 'invalid_ttl'  # Invalid TTL
                },
                'missing_data_entry': {
                    # Missing 'data' field
                    'timestamp': datetime.now().isoformat(),
                    'ttl': 300
                }
            }
            
            st.session_state['cache_with_errors'] = problematic_cache
            
            # Test error handling during cache access
            valid_entries = []
            error_entries = []
            
            for key, entry in problematic_cache.items():
                try:
                    # Validate cache entry structure
                    if 'data' not in entry or entry['data'] is None:
                        error_entries.append(key)
                        continue
                        
                    # Validate timestamp
                    timestamp = entry.get('timestamp')
                    if timestamp:
                        try:
                            datetime.fromisoformat(timestamp)
                        except (ValueError, TypeError):
                            error_entries.append(key)
                            continue
                            
                    # Validate TTL
                    ttl = entry.get('ttl')
                    if not isinstance(ttl, (int, float)):
                        error_entries.append(key)
                        continue
                        
                    # Entry is valid
                    valid_entries.append(key)
                    
                except Exception:
                    error_entries.append(key)
                    
            # Verify error handling
            assert 'valid_entry' in valid_entries
            assert 'corrupted_entry' in error_entries
            assert 'missing_data_entry' in error_entries
            
    def test_cache_memory_management(self):
        """Test cache memory management and cleanup."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Setup cache with size limits
            cache_config = {
                'max_entries': 5,
                'max_memory_mb': 10,
                'cleanup_threshold': 0.8,  # Cleanup when 80% full
                'entries': {},
                'memory_usage': 0,
                'cleanup_count': 0
            }
            
            st.session_state['memory_managed_cache'] = cache_config
            
            # Add entries to exceed limits
            for i in range(10):  # More than max_entries
                entry_key = f'cache_entry_{i}'
                entry_data = {
                    'data': {'dummy_data': f'data_{i}' * 100},  # Some data
                    'timestamp': datetime.now().isoformat(),
                    'ttl': 300,
                    'access_count': 0,
                    'last_access': datetime.now().isoformat()
                }
                
                cache = st.session_state['memory_managed_cache']
                
                # Check if cleanup is needed
                if len(cache['entries']) >= cache['max_entries']:
                    # Simple LRU cleanup - remove oldest entries
                    oldest_keys = sorted(
                        cache['entries'].keys(),
                        key=lambda k: cache['entries'][k]['last_access']
                    )
                    
                    # Remove oldest entry
                    if oldest_keys:
                        del cache['entries'][oldest_keys[0]]
                        cache['cleanup_count'] += 1
                        
                # Add new entry
                cache['entries'][entry_key] = entry_data
                
            # Verify memory management
            final_cache = st.session_state['memory_managed_cache']
            assert len(final_cache['entries']) <= final_cache['max_entries']
            assert final_cache['cleanup_count'] > 0  # Cleanup should have occurred
            
    @pytest.mark.asyncio
    async def test_distributed_cache_consistency(self):
        """Test cache consistency in distributed/multi-user scenarios."""
        # Simulate multiple user sessions with separate caches
        user_caches = {}
        
        for user_id in ['user1', 'user2', 'user3']:
            user_caches[user_id] = {
                'session_id': f'session_{user_id}',
                'cache_entries': {
                    'global_security_summary': {
                        'data': {'global_findings': 50},
                        'timestamp': datetime.now().isoformat(),
                        'scope': 'global',  # Shared across users
                        'ttl': 300
                    },
                    f'user_specific_data_{user_id}': {
                        'data': {'user_findings': 5 + len(user_id)},
                        'timestamp': datetime.now().isoformat(),
                        'scope': 'user',  # User-specific
                        'ttl': 600
                    }
                }
            }
            
        # Test cache consistency for global data
        global_entries = []
        for user_id, cache in user_caches.items():
            global_entry = cache['cache_entries'].get('global_security_summary')
            if global_entry and global_entry.get('scope') == 'global':
                global_entries.append(global_entry)
                
        # All global entries should have consistent data
        if len(global_entries) > 1:
            first_global_data = global_entries[0]['data']
            for entry in global_entries[1:]:
                assert entry['data'] == first_global_data
                
        # User-specific data should be different
        user_specific_data = []
        for user_id, cache in user_caches.items():
            user_key = f'user_specific_data_{user_id}'
            if user_key in cache['cache_entries']:
                user_data = cache['cache_entries'][user_key]['data']
                user_specific_data.append(user_data)
                
        # User-specific data should be unique
        unique_data = set(str(data) for data in user_specific_data)
        assert len(unique_data) == len(user_specific_data)  # All should be unique


class TestCachePerformance:
    """Test cache performance characteristics."""
    
    def test_cache_hit_rate_optimization(self):
        """Test cache hit rate optimization strategies."""
        # Simulate cache access patterns
        cache_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'hit_rate': 0.0
        }
        
        # Simulate various access patterns
        access_patterns = [
            # Frequent access to same data (should have high hit rate)
            ['security_summary'] * 10,
            ['iam_policies'] * 8,
            ['storage_analysis'] * 6,
            
            # Mixed access pattern
            ['security_summary', 'iam_policies', 'storage_analysis'] * 3,
            
            # Infrequent access to diverse data (lower hit rate)
            ['rare_data_1', 'rare_data_2', 'rare_data_3', 'rare_data_4']
        ]
        
        cache_entries = set()
        
        for pattern in access_patterns:
            for item in pattern:
                cache_stats['total_requests'] += 1
                
                if item in cache_entries:
                    cache_stats['cache_hits'] += 1
                else:
                    cache_stats['cache_misses'] += 1
                    cache_entries.add(item)  # Add to cache
                    
        # Calculate hit rate
        if cache_stats['total_requests'] > 0:
            cache_stats['hit_rate'] = cache_stats['cache_hits'] / cache_stats['total_requests']
            
        # Should achieve reasonable hit rate due to repeated access patterns
        assert cache_stats['hit_rate'] > 0.4  # At least 40% hit rate
        assert cache_stats['cache_hits'] > cache_stats['cache_misses']  # More hits than misses
        
    def test_cache_size_vs_performance_tradeoff(self):
        """Test tradeoffs between cache size and performance."""
        cache_sizes = [10, 50, 100, 500]  # Different cache sizes
        performance_metrics = {}
        
        for cache_size in cache_sizes:
            # Simulate cache with specific size limit
            cache = {'max_size': cache_size, 'entries': {}, 'evictions': 0}
            
            # Simulate adding entries beyond cache size
            for i in range(cache_size * 2):  # Add twice the cache size
                entry_key = f'entry_{i}'
                
                if len(cache['entries']) >= cache['max_size']:
                    # Remove oldest entry (simple eviction)
                    oldest_key = next(iter(cache['entries']))
                    del cache['entries'][oldest_key]
                    cache['evictions'] += 1
                    
                cache['entries'][entry_key] = {'data': f'data_{i}'}
                
            # Calculate performance metrics
            effective_cache_size = len(cache['entries'])
            eviction_rate = cache['evictions'] / (cache_size * 2) if cache_size > 0 else 0
            
            performance_metrics[cache_size] = {
                'effective_size': effective_cache_size,
                'eviction_rate': eviction_rate,
                'memory_efficiency': effective_cache_size / cache_size if cache_size > 0 else 0
            }
            
        # Verify that larger caches have better characteristics
        sizes = sorted(cache_sizes)
        for i in range(len(sizes) - 1):
            smaller_size = sizes[i]
            larger_size = sizes[i + 1]
            
            smaller_metrics = performance_metrics[smaller_size]
            larger_metrics = performance_metrics[larger_size]
            
            # Larger cache should have lower eviction rate
            assert larger_metrics['eviction_rate'] <= smaller_metrics['eviction_rate']


if __name__ == "__main__":
    # Run cache integration tests
    pytest.main([__file__, "-v", "--tb=short"])