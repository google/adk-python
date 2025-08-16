"""
Unified API Client - Single Source of Truth for All Backend Communication

This module consolidates all backend API communication into a single, unified client.
It combines the best features from both AssetDataService and PerformantAPIClient,
eliminating duplication and providing a consistent interface for all frontend components.

Key Features:
- Singleton pattern (one instance for entire app)
- Unified caching with TTL
- Retry logic with exponential backoff
- Connection pooling for performance
- Comprehensive error handling
- Type-safe with Pydantic models
- Automatic project context injection

Design Principles:
- DRY (Don't Repeat Yourself): No duplicate API calls or caching logic
- Single Source of Truth: All backend communication goes through this client
- Fail gracefully: Always return sensible defaults on error
- Performance first: Connection pooling, caching, and async where possible
"""

import os
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import wraps

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class UnifiedAPIClient:
    """
    Unified API client for all frontend-backend communication.
    Implements singleton pattern to ensure single source of truth.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the unified API client (only once due to singleton)."""
        if self._initialized:
            return
            
        # Backend configuration
        self.backend_host = os.getenv("BACKEND_HOST", "localhost")
        self.backend_port = os.getenv("BACKEND_PORT", "8000")
        self.backend_url = f"http://{self.backend_host}:{self.backend_port}"
        
        # Cache configuration
        self.cache_ttl = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes default
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Create optimized session with retry and pooling
        self.session = self._create_session()
        
        self._initialized = True
        logger.info(f"✅ UnifiedAPIClient initialized with backend: {self.backend_url}")
    
    def _create_session(self) -> requests.Session:
        """Create an optimized requests session with retry and connection pooling."""
        session = requests.Session()
        
        # Retry strategy with exponential backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        
        # Connection pooling adapter
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
    
    # ==================== CACHING METHODS ====================
    
    def _cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate a cache key for the given endpoint and parameters."""
        key = endpoint
        if params:
            # Sort params for consistent cache keys
            key += "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return key
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if not expired."""
        with self._cache_lock:
            if key in self._cache:
                cached_item = self._cache[key]
                if time.time() - cached_item['timestamp'] < self.cache_ttl:
                    logger.debug(f"🎯 Cache hit for: {key}")
                    return cached_item['data']
                else:
                    # Remove expired cache
                    del self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Store data in cache with timestamp."""
        with self._cache_lock:
            self._cache[key] = {
                'data': data,
                'timestamp': time.time()
            }
            logger.debug(f"💾 Cached: {key}")
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache entries matching pattern or all if no pattern."""
        with self._cache_lock:
            if pattern:
                keys_to_delete = [k for k in self._cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self._cache[key]
                logger.info(f"🗑️ Cleared {len(keys_to_delete)} cache entries matching '{pattern}'")
            else:
                self._cache.clear()
                logger.info("🗑️ Cleared all cache")
    
    # ==================== REQUEST METHODS ====================
    
    def _get_project_id(self) -> Optional[str]:
        """Get current project ID from session state."""
        if hasattr(st.session_state, 'selected_project'):
            return st.session_state.selected_project
        return os.getenv('GOOGLE_CLOUD_PROJECT')
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Make HTTP request to backend with caching and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data (for POST/PUT)
            params: Query parameters
            use_cache: Whether to use caching for GET requests
            timeout: Request timeout in seconds
            
        Returns:
            Response data or error dict
        """
        url = f"{self.backend_url}{endpoint}"
        
        # Auto-inject project ID if available
        project_id = self._get_project_id()
        if project_id:
            if params is None:
                params = {}
            if 'project_id' not in params:
                params['project_id'] = project_id
        
        # Check cache for GET requests
        if method.upper() == "GET" and use_cache:
            cache_key = self._cache_key(endpoint, params)
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        
        # Make the request
        try:
            logger.debug(f"🌐 {method} {url}")
            
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Cache successful GET responses
            if method.upper() == "GET" and use_cache and response.status_code == 200:
                cache_key = self._cache_key(endpoint, params)
                self._set_cache(cache_key, result)
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Timeout on {endpoint}")
            return {"success": False, "error": "Request timeout"}
            
        except requests.exceptions.ConnectionError:
            logger.error(f"🔌 Connection error to {endpoint}")
            return {"success": False, "error": "Backend unavailable"}
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ HTTP error on {endpoint}: {e}")
            return {"success": False, "error": str(e)}
            
        except Exception as e:
            logger.error(f"🚨 Unexpected error on {endpoint}: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== ASSET INVENTORY METHODS ====================
    
    def get_asset_summary(self, project_id: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
        """Get asset inventory summary for a project."""
        if not project_id:
            project_id = self._get_project_id()
        
        params = {"force_refresh": force_refresh} if force_refresh else None
        
        # Try snapshot endpoint first (most comprehensive)
        result = self._make_request(
            "GET", 
            f"/api/v1/assets/snapshot/{project_id}",
            params=params,
            use_cache=not force_refresh,
            timeout=45
        )
        
        if result.get("success"):
            return self._normalize_asset_data(result.get("data", {}))
        
        # Fallback to summary endpoint
        result = self._make_request(
            "GET",
            "/api/v1/assets/summary",
            params={"project_id": project_id},
            use_cache=not force_refresh
        )
        
        if result.get("success"):
            return self._normalize_asset_data(result.get("data", {}))
        
        # Fallback to summary endpoint
        result = self._make_request(
            "GET",
            "/api/v1/assets/summary",
            params={"project_id": project_id},
            use_cache=not force_refresh
        )
        
        if result.get("success"):
            return result.get("data", {})
        
        # Return empty structure on failure
        return self._get_empty_asset_summary()
    
    def discover_assets(self, query: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Discover assets using natural language query."""
        if not project_id:
            project_id = self._get_project_id()
            
        return self._make_request(
            "POST",
            "/api/v1/assets/search",
            data={"query": query, "project_id": project_id},
            use_cache=False
        )
    
    # ==================== SECURITY METHODS ====================
    
    def get_security_score(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get security score for a project."""
        return self._make_request("GET", "/api/v1/security/score")
    
    def get_security_findings(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get security findings for a project."""
        return self._make_request("GET", "/api/v1/security/findings")
    
    # ==================== RECOMMENDATIONS METHODS ====================
    
    def get_recommendations(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get security recommendations for a project."""
        data = {"count": 10}
        if project_id:
            data["project_id"] = project_id
            
        return self._make_request(
            "POST",
            "/api/v1/recommendations/dashboard",
            data=data
        )
    
    # ==================== IAM METHODS ====================
    
    def analyze_iam(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze IAM permissions for a project."""
        if not project_id:
            project_id = self._get_project_id()
            
        return self._make_request(
            "GET",
            f"/api/v1/iam/project/{project_id}/analyze-all-users"
        )
    
    def get_iam_policy(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get IAM policy for a project."""
        if not project_id:
            project_id = self._get_project_id()
            
        return self._make_request(
            "GET",
            f"/api/v1/iam/project/{project_id}/policy"
        )
    
    def analyze_user_permissions(self, user_email: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze permissions for a specific user."""
        if not project_id:
            project_id = self._get_project_id()
            
        return self._make_request(
            "GET",
            f"/api/v1/iam/project/{project_id}/user/{user_email}"
        )
    
    def analyze_all_users(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze permissions for all users in a project."""
        if not project_id:
            project_id = self._get_project_id()
            
        return self._make_request(
            "GET",
            f"/api/v1/iam/project/{project_id}/analyze-all-users"
        )
    
    # ==================== COMPLIANCE METHODS ====================
    
    def evaluate_compliance(self, framework: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate compliance for a specific framework."""
        if not project_id:
            project_id = self._get_project_id()
            
        return self._make_request(
            "POST",
            "/api/v1/compliance/evaluate",
            data={"framework": framework, "project_id": project_id}
        )
    
    def get_enabled_apis(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get list of enabled APIs for a project."""
        if not project_id:
            project_id = self._get_project_id()
            
        return self._make_request(
            "GET",
            "/api/v1/security/enabled-apis",
            params={"project_id": project_id}
        )
    
    # ==================== CHAT/AGENT METHODS ====================
    
    def chat_with_agent(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Send message to chat agent."""
        data = {
            "message": message,
            "project_id": self._get_project_id()
        }
        
        if context:
            data["context"] = context
            
        return self._make_request(
            "POST",
            "/api/v1/agent/chat",
            data=data,
            use_cache=False
        )
    
    # ==================== RADAR METHODS ====================
    
    def radar_chat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send RADAR phase-specific chat query to backend.
        
        Args:
            data: RADAR chat request data including:
                - query: User query
                - phase: RADAR phase (recognition, assessment, etc.)
                - project_id: GCP project ID
                - context: Phase-specific context
                
        Returns:
            RADAR chat response
        """
        return self._make_request(
            "POST",
            "/api/v1/radar/chat",
            data=data,
            use_cache=False
        )
    
    def discover_resources(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover resources for RADAR Recognition phase.
        
        Args:
            data: Discovery request data
            
        Returns:
            Discovery response
        """
        return self._make_request(
            "POST",
            "/api/v1/radar/discover",
            data=data,
            use_cache=False
        )
    
    # ==================== PROJECT METHODS ====================
    
    def get_projects(self) -> Dict[str, Any]:
        """Get list of available projects."""
        return self._make_request("GET", "/api/v1/gcp/projects")
    
    def get_project_info(self, project_id: str) -> Dict[str, Any]:
        """Get detailed project information."""
        return self._make_request("GET", f"/api/v1/gcp/projects/{project_id}")
    
    # ==================== MONITORING METHODS ====================
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        return self._make_request("GET", "/api/v1/monitoring/summary")
    
    # ==================== HELPER METHODS ====================
    
    def _normalize_asset_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize asset data to consistent format."""
        return {
            "total_assets": data.get("summary", {}).get("total_assets", 0),
            "asset_types": data.get("summary", {}).get("by_type", {}),
            "security_findings": data.get("security", {}).get("total_findings", 0),
            "high_risk_assets": data.get("security", {}).get("high_severity_findings", 0),
            "recommendations": data.get("recommendations", {}).get("total", 0),
            "last_updated": data.get("cache_info", {}).get("cached_at", datetime.now().isoformat()),
            "data_source": data.get("api_metadata", {}).get("source", "backend"),
            "raw_data": data
        }
    
    def _get_empty_asset_summary(self) -> Dict[str, Any]:
        """Return empty asset summary structure."""
        return {
            "total_assets": 0,
            "asset_types": {},
            "security_findings": 0,
            "high_risk_assets": 0,
            "recommendations": 0,
            "last_updated": datetime.now().isoformat(),
            "data_source": "fallback",
            "error": "Unable to fetch asset data"
        }
    
    def get_metrics_for_dashboard(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics formatted for dashboard display."""
        summary = self.get_asset_summary(project_id)
        
        return {
            "total_assets": summary.get("total_assets", 0),
            "security_findings": summary.get("security_findings", 0),
            "high_risk_assets": summary.get("high_risk_assets", 0),
            "active_recommendations": summary.get("recommendations", 0)
        }
    
    # ==================== AGENT INTERACTION METHODS ====================
    
    def chat_with_radar(self, query: str, session_id: str = None, user_id: str = "default_user", 
                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a chat request to the RADAR coordinator agent.
        
        Args:
            query: The user's question or request
            session_id: Optional session ID for conversation continuity
            user_id: User identifier
            context: Additional context for the agent
            
        Returns:
            Agent response with recommendations and analysis
        """
        payload = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": self._get_project_id(),
            "context": context or {}
        }
        
        return self._make_request(
            method="POST",
            endpoint="/api/v1/radar/chat",
            data=payload,
            use_cache=False,  # Don't cache chat responses
            timeout=60  # Longer timeout for agent processing
        )
    
    def get_agent_status(self, agent_type: str = "radar") -> Dict[str, Any]:
        """
        Get the status of a specific agent or all agents.
        
        Args:
            agent_type: Type of agent to check (radar, recognition, assessment, etc.)
            
        Returns:
            Agent status information
        """
        return self._make_request(
            method="GET",
            endpoint=f"/api/v1/agents/{agent_type}/status",
            use_cache=True,
            timeout=10
        )
    
    def execute_radar_phase(self, phase: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a specific RADAR phase.
        
        Args:
            phase: The RADAR phase to execute (recognition, assessment, decision, action, review)
            input_data: Input data for the phase
            
        Returns:
            Phase execution results
        """
        payload = {
            "phase": phase,
            "input_data": input_data or {},
            "project_id": self._get_project_id()
        }
        
        return self._make_request(
            method="POST",
            endpoint=f"/api/v1/radar/execute/{phase}",
            data=payload,
            use_cache=False,
            timeout=60
        )
    
    def stream_agent_response(self, query: str, callback=None) -> None:
        """
        Stream agent responses using WebSocket connection.
        
        Args:
            query: The user's question
            callback: Function to call with each response chunk
        """
        # This would require WebSocket implementation
        # For now, returning a placeholder
        logger.warning("WebSocket streaming not yet implemented in unified client")
        return {"success": False, "message": "Streaming not implemented"}


# ==================== SINGLETON INSTANCE ====================

# Create the single instance that all components will use
api_client = UnifiedAPIClient()

# Export both the class and instance for flexibility
__all__ = ['UnifiedAPIClient', 'api_client']