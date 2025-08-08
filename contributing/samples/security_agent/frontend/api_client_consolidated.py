"""
IMMEDIATE FIX: Drop-in replacement for existing API client
Fixes bottlenecks without requiring full migration.
"""

import requests
import streamlit as st
import time
import logging
from typing import Dict, Any, Optional
from functools import wraps
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class PerformantAPIClient:
    """High-performance API client - drop-in replacement."""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.session = self._create_session()
        self._cache = {}
        self._cache_lock = threading.Lock()
        
    def _create_session(self):
        """Create optimized session with connection pooling."""
        session = requests.Session()
        
        # Connection pooling for performance
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Default headers
        session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
        
        return session
    
    def _cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate cache key."""
        key = endpoint
        if params:
            key += str(sorted(params.items()))
        return key
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached response."""
        with self._cache_lock:
            item = self._cache.get(key)
            if item and time.time() - item['timestamp'] < 300:  # 5 min cache
                return item['data']
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Set cached response."""
        with self._cache_lock:
            self._cache[key] = {
                'data': data,
                'timestamp': time.time()
            }
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, 
                     use_cache: bool = True) -> Dict[str, Any]:
        """Optimized request method."""
        url = f"{self.backend_url}{endpoint}"
        cache_key = self._cache_key(endpoint, data) if use_cache else None
        
        # Check cache first
        if cache_key and method == "GET":
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        
        # Add project context
        if hasattr(st.session_state, 'selected_project') and st.session_state.selected_project:
            if data is None:
                data = {}
            data['project_id'] = st.session_state.selected_project
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                response = self.session.get(url, params=data, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=15)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            execution_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                
                # Cache successful GET requests
                if cache_key and method == "GET":
                    self._set_cache(cache_key, result)
                
                # Add performance metadata
                if isinstance(result, dict):
                    result['_performance'] = {
                        'response_time_ms': execution_time,
                        'from_cache': False
                    }
                
                return result
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "status_code": response.status_code
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out. Backend may be slow or unavailable."
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Cannot connect to backend. Ensure server is running."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
    
    # Drop-in replacement methods for existing API client
    def get_projects(self) -> Dict[str, Any]:
        """Get GCP projects - PERFORMANCE OPTIMIZED."""
        return self._make_request("GET", "/api/v1/gcp/projects")
    
    def get_security_score(self) -> Dict[str, Any]:
        """Get security score - PERFORMANCE OPTIMIZED."""
        return self._make_request("GET", "/api/v1/security/score")
    
    def get_recommendations(self, priority: str = "high") -> Dict[str, Any]:
        """Get recommendations - PERFORMANCE OPTIMIZED."""
        return self._make_request("POST", "/api/v1/recommendations/dashboard", 
                                {"priority": priority})
    
    def discover_apis(self, service_filter: str = None, preferred_only: bool = True, 
                     include_deprecated: bool = False) -> Dict[str, Any]:
        """Discover APIs - PERFORMANCE OPTIMIZED."""
        data = {
            "service_filter": service_filter,
            "preferred_only": preferred_only,
            "include_deprecated": include_deprecated
        }
        return self._make_request("POST", "/api/v1/gcp-api-explorer/discover", data)
    
    def test_endpoint(self, test_request: Dict[str, Any]) -> Dict[str, Any]:
        """Test endpoint - PERFORMANCE OPTIMIZED."""
        return self._make_request("POST", "/api/v1/gcp-api-explorer/test", 
                                test_request, use_cache=False)

# Create optimized global instance
_optimized_client = None

def get_optimized_api_client() -> PerformantAPIClient:
    """Get the optimized API client instance."""
    global _optimized_client
    if _optimized_client is None:
        _optimized_client = PerformantAPIClient()
    return _optimized_client

# Backward compatibility - replace your existing api_client
api_client = get_optimized_api_client()

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor API call performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            
            # Store performance metrics in session state
            if 'api_performance' not in st.session_state:
                st.session_state.api_performance = []
            
            st.session_state.api_performance.append({
                'function': func.__name__,
                'execution_time': execution_time,
                'success': result.get('success', False) if isinstance(result, dict) else True,
                'timestamp': time.time()
            })
            
            # Keep only last 100 records
            if len(st.session_state.api_performance) > 100:
                st.session_state.api_performance = st.session_state.api_performance[-100:]
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Performance monitoring error in {func.__name__}: {e}")
            raise
    
    return wrapper

# Apply monitoring to API client methods
api_client.get_projects = monitor_performance(api_client.get_projects)
api_client.get_security_score = monitor_performance(api_client.get_security_score)
api_client.get_recommendations = monitor_performance(api_client.get_recommendations)