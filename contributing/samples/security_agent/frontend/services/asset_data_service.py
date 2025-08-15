"""
Unified Asset Data Service - Single Source of Truth for Asset Inventory Data

This service follows DRY and SOLID principles by providing a centralized interface
for all asset inventory data operations. It consolidates API calls and provides
caching to improve performance and reduce redundant backend calls.

Following SOLID principles:
- Single Responsibility: Manages only asset data operations
- Open/Closed: Extensible for new asset types without modification
- Liskov Substitution: Provides consistent interface for all asset operations
- Interface Segregation: Clean, focused interface for asset data
- Dependency Inversion: Depends on abstract backend interface
"""

import streamlit as st
import requests
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
try:
    from requests.packages.urllib3.util.retry import Retry
except ImportError:
    try:
        from urllib3.util.retry import Retry
    except ImportError:
        # Fallback for systems without urllib3 retry
        Retry = None

logger = logging.getLogger(__name__)

class AssetDataService:
    """Centralized service for all asset inventory data operations."""
    
    def __init__(self, backend_url: str = None):
        # Auto-detect backend URL from environment or use default
        if backend_url is None:
            backend_port = os.getenv("BACKEND_PORT", "8000")
            backend_host = os.getenv("BACKEND_HOST", "localhost")
            self.backend_url = f"http://{backend_host}:{backend_port}"
        else:
            self.backend_url = backend_url
        self.cache_duration = 300  # 5 minutes cache
        
        # Setup requests session with retry strategy
        self.session = self._create_requests_session()
        
        logger.info(f"🔗 AssetDataService initialized with backend URL: {self.backend_url}")
    
    def _create_requests_session(self) -> requests.Session:
        """Create a requests session with retry strategy and timeouts."""
        session = requests.Session()
        
        # Only configure retry strategy if Retry is available
        if Retry is not None:
            # Define retry strategy
            retry_strategy = Retry(
                total=3,  # Total number of retries
                backoff_factor=1,  # Wait time between retries (1, 2, 4 seconds)
                status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
                allowed_methods=["HEAD", "GET", "OPTIONS"]  # Only retry safe methods
            )
            
            # Mount adapter with retry strategy
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        else:
            logger.warning("urllib3 Retry not available - using basic session without retry logic")
        
        return session
        
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if f"{cache_key}_timestamp" not in st.session_state:
            return False
        
        cached_time = st.session_state[f"{cache_key}_timestamp"]
        return (time.time() - cached_time) < self.cache_duration
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from session cache if valid."""
        if self._is_cache_valid(cache_key) and cache_key in st.session_state:
            logger.info(f"Using cached data for {cache_key}")
            return st.session_state[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store data in session cache with timestamp."""
        st.session_state[cache_key] = data
        st.session_state[f"{cache_key}_timestamp"] = time.time()
        logger.info(f"Cached data for {cache_key}")
    
    def get_asset_summary(self, project_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive asset inventory summary for a project.
        
        This is the single source of truth for asset data across the application.
        Now uses the snapshot endpoint for real-time GCP data with JSON caching.
        
        Args:
            project_id: GCP project ID
            force_refresh: Force refresh cache if True
            
        Returns:
            Dictionary containing asset summary data
        """
        cache_key = f"asset_summary_{project_id}"
        
        # Check cache first (DRY principle - avoid redundant API calls)
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                # Add cache indicator
                cached_data["from_frontend_cache"] = True
                return cached_data
        
        # Try multiple endpoints with proper fallback handling
        endpoints_to_try = [
            {
                "name": "snapshot",
                "url": f"{self.backend_url}/api/v1/assets/snapshot/{project_id}",
                "params": {"force_refresh": force_refresh},
                "timeout": 45,
                "processor": self._normalize_snapshot_data
            },
            {
                "name": "summary",
                "url": f"{self.backend_url}/api/v1/assets/summary",
                "params": {"project_id": project_id},
                "timeout": 20,
                "processor": self._normalize_asset_data
            }
        ]
        
        for endpoint_info in endpoints_to_try:
            try:
                logger.info(f"🔍 Trying {endpoint_info['name']} endpoint for project: {project_id}")
                
                response = self.session.get(
                    endpoint_info["url"],
                    params=endpoint_info["params"],
                    timeout=endpoint_info["timeout"]
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("success") and data.get("data"):
                        # Process the data using the appropriate processor
                        if endpoint_info["name"] == "snapshot":
                            asset_data = endpoint_info["processor"](data["data"])
                            # Add snapshot metadata
                            asset_data["data_source"] = data.get("data", {}).get("api_metadata", {}).get("source", "unknown")
                            asset_data["cache_info"] = data.get("data", {}).get("cache_info")
                        else:
                            asset_data = endpoint_info["processor"](data)
                        
                        asset_data["from_frontend_cache"] = False
                        asset_data["endpoint_used"] = endpoint_info["name"]
                        
                        # Cache the successful response
                        self._store_in_cache(cache_key, asset_data)
                        
                        logger.info(f"✅ Successfully fetched from {endpoint_info['name']}: {asset_data.get('total_assets', 0)} assets")
                        return asset_data
                    else:
                        logger.warning(f"{endpoint_info['name']} endpoint returned invalid data structure")
                        continue
                else:
                    logger.warning(f"{endpoint_info['name']} endpoint error: {response.status_code}")
                    continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"{endpoint_info['name']} endpoint timeout after {endpoint_info['timeout']}s")
                continue
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"Cannot connect to {endpoint_info['name']} endpoint")
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error with {endpoint_info['name']} endpoint: {e}")
                continue
        
        # If all endpoints failed, try to return cached data or fallback
        logger.error("All asset endpoints failed, using cached or fallback data")
        cached_data = self._get_from_cache(cache_key)
        return cached_data if cached_data else self._get_fallback_data()
    
    def _normalize_snapshot_data(self, snapshot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize snapshot data from the new API endpoint.
        Extracts asset counts and categories from the real-time GCP data.
        """
        normalized = {
            "total_assets": 0,
            "asset_categories": {},
            "security_findings": [],
            "locations": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Extract from snapshot structure
        if "summary" in snapshot_data:
            summary = snapshot_data["summary"]
            normalized["total_assets"] = summary.get("total_assets", 0)
            normalized["asset_categories"] = summary.get("categories", {})
            normalized["security_findings_count"] = summary.get("security_findings_count", 0)
        
        # Extract from categorized assets
        if "assets_by_category" in snapshot_data:
            for category, data in snapshot_data["assets_by_category"].items():
                if category not in normalized["asset_categories"]:
                    normalized["asset_categories"][category] = 0
                normalized["asset_categories"][category] = data.get("count", 0)
                
                # Extract location data
                for asset in data.get("assets", []):
                    location = asset.get("location", "unknown")
                    if location not in normalized["locations"]:
                        normalized["locations"][location] = 0
                    normalized["locations"][location] += 1
        
        # Extract security findings
        if "security_findings" in snapshot_data:
            normalized["security_findings"] = snapshot_data["security_findings"][:5]  # Top 5
            normalized["high_risk_count"] = len([f for f in snapshot_data["security_findings"] 
                                                 if f.get("severity") == "high"])
        
        # Add API metadata if available
        if "api_metadata" in snapshot_data:
            normalized["api_call_duration"] = snapshot_data["api_metadata"].get("call_duration")
            normalized["data_timestamp"] = snapshot_data["api_metadata"].get("timestamp")
        
        return normalized
    
    def _fetch_from_summary_endpoint(self, project_id: str) -> Dict[str, Any]:
        """Fallback to summary endpoint if snapshot fails."""
        try:
            response = self.session.get(
                f"{self.backend_url}/api/v1/assets/summary",
                params={"project_id": project_id},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._normalize_asset_data(data)
            else:
                return self._get_fallback_data()
        except Exception as e:
            logger.error(f"Summary endpoint failed: {e}")
            return self._get_fallback_data()
    
    def _normalize_asset_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize asset data structure for consistent use across components."""
        data = raw_data.get("data", {})
        
        return {
            "success": True,
            "timestamp": raw_data.get("timestamp", datetime.now().isoformat()),
            "total_assets": data.get("total_assets", 0),
            "security_findings": data.get("security_findings", 0),
            "high_risk_assets": data.get("high_risk_assets", 0),
            "active_recommendations": data.get("active_recommendations", 0),
            "asset_types": data.get("asset_types", {}),
            
            # Computed metrics (following Single Responsibility)
            "security_score": self._calculate_security_score(data),
            "risk_ratio": self._calculate_risk_ratio(data),
            "asset_diversity": self._calculate_asset_diversity(data.get("asset_types", {})),
            
            # Categorized data for different use cases
            "metrics": self._extract_metrics(data),
            "charts_data": self._prepare_charts_data(data),
            "chat_summary": self._prepare_chat_summary(data)
        }
    
    def _calculate_security_score(self, data: Dict[str, Any]) -> int:
        """Calculate overall security score (0-100)."""
        total_assets = data.get("total_assets", 0)
        if total_assets == 0:
            return 100
        
        high_risk = data.get("high_risk_assets", 0)
        security_findings = data.get("security_findings", 0)
        
        risk_ratio = (high_risk + security_findings) / total_assets
        return max(0, 100 - int(risk_ratio * 100))
    
    def _calculate_risk_ratio(self, data: Dict[str, Any]) -> float:
        """Calculate risk ratio (0.0-1.0)."""
        total_assets = data.get("total_assets", 0)
        if total_assets == 0:
            return 0.0
        
        high_risk = data.get("high_risk_assets", 0)
        security_findings = data.get("security_findings", 0)
        
        return (high_risk + security_findings) / total_assets
    
    def _calculate_asset_diversity(self, asset_types: Dict[str, int]) -> float:
        """Calculate asset type diversity score (0.0-100.0)."""
        if not asset_types:
            return 0.0
        
        active_types = len([count for count in asset_types.values() if count > 0])
        total_types = len(asset_types)
        
        return (active_types / total_types * 100) if total_types > 0 else 0.0
    
    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics optimized for dashboard display."""
        return {
            "total_assets": data.get("total_assets", 0),
            "security_findings": data.get("security_findings", 0),
            "high_risk_assets": data.get("high_risk_assets", 0),
            "active_recommendations": data.get("active_recommendations", 0),
            "security_score": self._calculate_security_score(data),
            "iam_accounts": data.get("asset_types", {}).get("IAM Accounts", 0)
        }
    
    def _prepare_charts_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data optimized for chart rendering."""
        asset_types = data.get("asset_types", {})
        
        return {
            "asset_breakdown": {
                "labels": list(asset_types.keys()),
                "values": list(asset_types.values()),
                "total": sum(asset_types.values())
            },
            "security_status": {
                "secure_assets": data.get("total_assets", 0) - data.get("high_risk_assets", 0),
                "high_risk_assets": data.get("high_risk_assets", 0)
            },
            "findings_categories": self._simulate_findings_breakdown(data.get("security_findings", 0))
        }
    
    def _simulate_findings_breakdown(self, total_findings: int) -> Dict[str, int]:
        """Simulate findings breakdown by category (placeholder for real data)."""
        if total_findings == 0:
            return {}
        
        return {
            "IAM & Access": max(1, total_findings // 3),
            "Network Security": max(1, total_findings // 4),
            "Data Protection": max(1, total_findings // 4),
            "Configuration": max(1, total_findings - (total_findings // 3) - (total_findings // 4) - (total_findings // 4))
        }
    
    def _prepare_chat_summary(self, data: Dict[str, Any]) -> str:
        """Prepare human-readable summary for chat integration."""
        total_assets = data.get("total_assets", 0)
        security_findings = data.get("security_findings", 0)
        high_risk = data.get("high_risk_assets", 0)
        asset_types = data.get("asset_types", {})
        
        if total_assets == 0:
            return "No assets discovered in this project yet. Run an asset inventory scan to get started."
        
        # Build natural language summary
        summary_parts = [
            f"I found {total_assets} assets across {len(asset_types)} different categories."
        ]
        
        if high_risk > 0:
            summary_parts.append(f"⚠️ {high_risk} assets require immediate attention due to high security risk.")
        
        if security_findings > 0:
            summary_parts.append(f"🔍 {security_findings} security findings need to be reviewed.")
        else:
            summary_parts.append("✅ No critical security findings detected.")
        
        # Top asset types
        if asset_types:
            top_assets = sorted(asset_types.items(), key=lambda x: x[1], reverse=True)[:3]
            asset_list = ", ".join([f"{count} {asset_type.lower()}" for asset_type, count in top_assets])
            summary_parts.append(f"Top assets: {asset_list}.")
        
        return " ".join(summary_parts)
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Provide fallback data when API is unavailable."""
        return {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "total_assets": 0,
            "security_findings": 0,
            "high_risk_assets": 0,
            "active_recommendations": 0,
            "asset_types": {},
            "security_score": 0,
            "risk_ratio": 0.0,
            "asset_diversity": 0.0,
            "metrics": {},
            "charts_data": {},
            "chat_summary": "Asset inventory data is currently unavailable. Please check your connection to the backend service."
        }
    
    def clear_cache(self, project_id: str = None) -> None:
        """Clear cached asset data."""
        if project_id:
            cache_key = f"asset_summary_{project_id}"
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            if f"{cache_key}_timestamp" in st.session_state:
                del st.session_state[f"{cache_key}_timestamp"]
            logger.info(f"Cleared cache for project: {project_id}")
        else:
            # Clear all asset caches
            keys_to_clear = [key for key in st.session_state.keys() if key.startswith("asset_summary_")]
            for key in keys_to_clear:
                del st.session_state[key]
            logger.info("Cleared all asset data caches")
    
    def get_metrics_for_dashboard(self, project_id: str) -> Dict[str, Any]:
        """Get metrics specifically formatted for dashboard display."""
        data = self.get_asset_summary(project_id)
        return data.get("metrics", {})
    
    def get_charts_data(self, project_id: str) -> Dict[str, Any]:
        """Get data specifically formatted for chart rendering."""
        data = self.get_asset_summary(project_id)
        return data.get("charts_data", {})
    
    def get_chat_summary(self, project_id: str) -> str:
        """Get human-readable summary for chat integration."""
        data = self.get_asset_summary(project_id)
        return data.get("chat_summary", "Asset data unavailable.")
    
    def is_data_available(self, project_id: str) -> bool:
        """Check if asset data is available for the project."""
        data = self.get_asset_summary(project_id)
        return data.get("success", False) and data.get("total_assets", 0) > 0
    
    def check_backend_health(self) -> Dict[str, Any]:
        """Check backend service health and connectivity."""
        health_status = {
            "backend_url": self.backend_url,
            "connected": False,
            "response_time_ms": None,
            "status_code": None,
            "error": None,
            "endpoints_available": []
        }
        
        try:
            start_time = time.time()
            response = self.session.get(
                f"{self.backend_url}/health",
                timeout=10
            )
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            health_status.update({
                "connected": True,
                "response_time_ms": round(response_time, 2),
                "status_code": response.status_code
            })
            
            if response.status_code == 200:
                backend_health = response.json()
                health_status["endpoints_available"] = list(backend_health.get("endpoints", {}).keys())
                health_status["backend_features"] = backend_health.get("features", {})
            
        except requests.exceptions.ConnectionError:
            health_status["error"] = "Connection refused - backend may not be running"
        except requests.exceptions.Timeout:
            health_status["error"] = "Backend health check timeout"
        except Exception as e:
            health_status["error"] = str(e)
        
        return health_status
    
    def get_debug_info(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive debug information for troubleshooting."""
        return {
            "service_config": {
                "backend_url": self.backend_url,
                "cache_duration": self.cache_duration,
                "session_configured": self.session is not None
            },
            "backend_health": self.check_backend_health(),
            "cache_status": {
                "cache_key": f"asset_summary_{project_id}",
                "has_cached_data": self._is_cache_valid(f"asset_summary_{project_id}"),
                "cache_timestamp": st.session_state.get(f"asset_summary_{project_id}_timestamp")
            },
            "environment": {
                "backend_port": os.getenv("BACKEND_PORT", "8000"),
                "backend_host": os.getenv("BACKEND_HOST", "localhost")
            }
        }


# Global service instance (Singleton pattern for consistency)
asset_data_service = AssetDataService()