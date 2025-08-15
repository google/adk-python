"""
Asset Inventory API Client

This module provides a dedicated client for interacting with the GCP Asset Inventory API
endpoints, with caching, error handling, and performance optimization.
"""

import requests
import streamlit as st
import time
import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class AssetInventoryClient:
    """Client for GCP Asset Inventory API with caching and error handling."""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.base_url = f"{backend_url}/api/v1/asset-inventory"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Cache settings
        self.cache_ttl = 300  # 5 minutes
        self._cache = {}
    
    def _get_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate cache key for request."""
        key = f"{endpoint}"
        if params:
            key += f"_{hash(str(sorted(params.items())))}"
        return key
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        
        cached_time = self._cache[cache_key]["timestamp"]
        return (datetime.now() - cached_time).seconds < self.cache_ttl
    
    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]
        return None
    
    def _cache_data(self, cache_key: str, data: Dict[str, Any]):
        """Cache response data."""
        self._cache[cache_key] = {
            "data": data,
            "timestamp": datetime.now()
        }
    
    def _make_request(self, endpoint: str, params: Dict = None, use_cache: bool = True) -> Dict[str, Any]:
        """Make API request with caching and error handling."""
        cache_key = self._get_cache_key(endpoint, params) if use_cache else None
        
        # Check cache first
        if cache_key:
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                logger.info(f"Returning cached data for {endpoint}")
                return cached_data
        
        # Add project context
        if not params:
            params = {}
        
        if hasattr(st.session_state, 'selected_project') and st.session_state.selected_project:
            params['project_id'] = st.session_state.selected_project
        
        try:
            start_time = time.time()
            url = f"{self.base_url}{endpoint}"
            
            logger.info(f"Making request to {url} with params: {params}")
            
            response = self.session.get(url, params=params, timeout=15)
            execution_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Add performance metadata
                if isinstance(data, dict):
                    data['_performance'] = {
                        'response_time_ms': execution_time,
                        'from_cache': False,
                        'endpoint': endpoint
                    }
                
                # Cache successful responses
                if cache_key:
                    self._cache_data(cache_key, data)
                
                logger.info(f"API request completed in {execution_time:.1f}ms")
                return data
            else:
                error_msg = f"API request failed: HTTP {response.status_code}"
                logger.error(f"{error_msg}: {response.text}")
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code
                }
                
        except requests.exceptions.Timeout:
            error_msg = "Request timed out - Asset Inventory API may be slow"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "timeout": True
            }
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Asset Inventory API"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "connection_error": True
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    # Asset Inventory API Methods
    
    def get_asset_summary(self, project_id: str = None) -> Dict[str, Any]:
        """Get comprehensive asset inventory summary."""
        params = {}
        if project_id:
            params['project_id'] = project_id
            
        return self._make_request("/summary", params)
    
    def discover_assets(self, query: str, project_id: str = None) -> Dict[str, Any]:
        """Discover assets using natural language query."""
        params = {"query": query}
        if project_id:
            params['project_id'] = project_id
            
        return self._make_request("/discover", params, use_cache=False)
    
    def get_compute_instances(self, project_id: str = None) -> Dict[str, Any]:
        """Get all compute instances."""
        params = {}
        if project_id:
            params['project_id'] = project_id
            
        return self._make_request("/compute/instances", params)
    
    def get_storage_buckets(self, project_id: str = None) -> Dict[str, Any]:
        """Get all storage buckets."""
        params = {}
        if project_id:
            params['project_id'] = project_id
            
        return self._make_request("/storage/buckets", params)
    
    def get_cloud_functions(self, project_id: str = None) -> Dict[str, Any]:
        """Get all cloud functions."""
        params = {}
        if project_id:
            params['project_id'] = project_id
            
        return self._make_request("/serverless/functions", params)
    
    def get_databases(self, project_id: str = None) -> Dict[str, Any]:
        """Get all databases."""
        params = {}
        if project_id:
            params['project_id'] = project_id
            
        return self._make_request("/data/databases", params)
    
    def get_kubernetes_clusters(self, project_id: str = None) -> Dict[str, Any]:
        """Get all Kubernetes clusters."""
        params = {}
        if project_id:
            params['project_id'] = project_id
            
        return self._make_request("/container/clusters", params)
    
    def analyze_security_assets(self, project_id: str = None) -> Dict[str, Any]:
        """Analyze security-related assets."""
        params = {}
        if project_id:
            params['project_id'] = project_id
            
        return self._make_request("/security/analyze", params)
    
    def search_assets(self, name_pattern: str, project_id: str = None) -> Dict[str, Any]:
        """Search assets by name pattern."""
        params = {"name_pattern": name_pattern}
        if project_id:
            params['project_id'] = project_id
            
        return self._make_request("/search", params, use_cache=False)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Check Asset Inventory service health."""
        return self._make_request("/health", use_cache=False)
    
    # Utility methods
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Asset Inventory cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        valid_entries = sum(1 for key in self._cache.keys() if self._is_cache_valid(key))
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": total_entries - valid_entries,
            "cache_ttl_seconds": self.cache_ttl
        }
    
    def get_performance_metrics(self) -> List[Dict[str, Any]]:
        """Get performance metrics from session state."""
        if hasattr(st.session_state, 'asset_api_performance'):
            return st.session_state.asset_api_performance
        return []

# Global client instance
_asset_client = None

def get_asset_inventory_client() -> AssetInventoryClient:
    """Get singleton asset inventory client instance."""
    global _asset_client
    if _asset_client is None:
        _asset_client = AssetInventoryClient()
    return _asset_client

# Convenience functions for easy import
asset_client = get_asset_inventory_client()

def get_asset_summary(project_id: str = None) -> Dict[str, Any]:
    """Get asset inventory summary - convenience function."""
    return asset_client.get_asset_summary(project_id)

def discover_assets(query: str, project_id: str = None) -> Dict[str, Any]:
    """Discover assets with natural language - convenience function."""
    return asset_client.discover_assets(query, project_id)

def search_assets(name_pattern: str, project_id: str = None) -> Dict[str, Any]:
    """Search assets by name - convenience function."""
    return asset_client.search_assets(name_pattern, project_id)