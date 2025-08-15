"""
GCP Asset Inventory Tools for ADK Integration

These tools provide unified access to GCP resources through natural language queries,
integrating with the Asset Inventory API for comprehensive resource discovery.
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
import json

# Import the enhanced service
import sys
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

try:
    from services.enhanced_asset_inventory_service import EnhancedGCPAssetInventoryService
    SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced Asset Inventory Service not available: {e}")
    SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_asset_service() -> Optional[EnhancedGCPAssetInventoryService]:
    """Get asset inventory service instance."""
    if not SERVICE_AVAILABLE:
        return None
    
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
    return EnhancedGCPAssetInventoryService(project_id)

def discover_gcp_resources(query: str) -> Dict[str, Any]:
    """
    Discover GCP resources using natural language queries.
    
    This tool processes natural language queries like:
    - "show me my compute instances"
    - "what databases do I have"
    - "list my cloud functions"
    - "analyze my security assets"
    
    Args:
        query: Natural language query describing what resources to find
        
    Returns:
        Dict containing discovered resources and analysis
    """
    logger.info(f"[Asset Discovery] Processing query: '{query}'")
    
    service = get_asset_service()
    if not service:
        return {
            "success": False,
            "error": "Asset Inventory service not available",
            "suggestion": "Please ensure Google Cloud Asset Inventory API is enabled"
        }
    
    try:
        # Use asyncio to run the async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                service.process_natural_language_query(query)
            )
            
            logger.info(f"[Asset Discovery] Successfully processed query, found assets")
            return {
                "success": True,
                "data": result,
                "query_processed": query
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"[Asset Discovery] Error processing query: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "suggestion": "Check Google Cloud credentials and Asset Inventory API permissions"
        }

def get_compute_instances() -> Dict[str, Any]:
    """
    Get all compute instances in the project.
    
    Returns:
        Dict containing compute instance details and analysis
    """
    logger.info("[Asset Discovery] Getting compute instances")
    
    service = get_asset_service()
    if not service:
        return {
            "success": False,
            "error": "Asset Inventory service not available"
        }
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(service.get_compute_instances())
            return {
                "success": True,
                "data": result
            }
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"[Asset Discovery] Error getting compute instances: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_storage_buckets() -> Dict[str, Any]:
    """
    Get all storage buckets in the project.
    
    Returns:
        Dict containing storage bucket details and analysis
    """
    logger.info("[Asset Discovery] Getting storage buckets")
    
    service = get_asset_service()
    if not service:
        return {
            "success": False,
            "error": "Asset Inventory service not available"
        }
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(service.get_storage_buckets())
            return {
                "success": True,
                "data": result
            }
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"[Asset Discovery] Error getting storage buckets: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_cloud_functions() -> Dict[str, Any]:
    """
    Get all cloud functions in the project.
    
    Returns:
        Dict containing cloud function details and analysis
    """
    logger.info("[Asset Discovery] Getting cloud functions")
    
    service = get_asset_service()
    if not service:
        return {
            "success": False,
            "error": "Asset Inventory service not available"
        }
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(service.get_cloud_functions())
            return {
                "success": True,
                "data": result
            }
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"[Asset Discovery] Error getting cloud functions: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_databases() -> Dict[str, Any]:
    """
    Get all databases in the project (Cloud SQL, Spanner, BigQuery, etc.).
    
    Returns:
        Dict containing database details and analysis
    """
    logger.info("[Asset Discovery] Getting databases")
    
    service = get_asset_service()
    if not service:
        return {
            "success": False,
            "error": "Asset Inventory service not available"
        }
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(service.get_databases())
            return {
                "success": True,
                "data": result
            }
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"[Asset Discovery] Error getting databases: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_kubernetes_clusters() -> Dict[str, Any]:
    """
    Get all Kubernetes clusters in the project.
    
    Returns:
        Dict containing Kubernetes cluster details and analysis
    """
    logger.info("[Asset Discovery] Getting Kubernetes clusters")
    
    service = get_asset_service()
    if not service:
        return {
            "success": False,
            "error": "Asset Inventory service not available"
        }
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(service.get_kubernetes_clusters())
            return {
                "success": True,
                "data": result
            }
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"[Asset Discovery] Error getting Kubernetes clusters: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def analyze_security_assets() -> Dict[str, Any]:
    """
    Analyze security-related assets and provide security recommendations.
    
    Returns:
        Dict containing security analysis and recommendations
    """
    logger.info("[Asset Discovery] Analyzing security assets")
    
    service = get_asset_service()
    if not service:
        return {
            "success": False,
            "error": "Asset Inventory service not available"
        }
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(service.get_security_assets())
            return {
                "success": True,
                "data": result
            }
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"[Asset Discovery] Error analyzing security assets: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def search_assets_by_name(name_pattern: str) -> Dict[str, Any]:
    """
    Search for assets by name pattern.
    
    Args:
        name_pattern: Pattern to search for in asset names
        
    Returns:
        Dict containing matching assets
    """
    logger.info(f"[Asset Discovery] Searching assets by name pattern: '{name_pattern}'")
    
    service = get_asset_service()
    if not service:
        return {
            "success": False,
            "error": "Asset Inventory service not available"
        }
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(service.search_assets_by_name(name_pattern))
            return {
                "success": True,
                "data": result
            }
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"[Asset Discovery] Error searching assets by name: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_asset_inventory_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of all assets in the project.
    
    Returns:
        Dict containing complete asset inventory summary
    """
    logger.info("[Asset Discovery] Getting complete asset inventory summary")
    
    service = get_asset_service()
    if not service:
        return {
            "success": False,
            "error": "Asset Inventory service not available"
        }
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                service.process_natural_language_query("provide a comprehensive overview of all my GCP resources")
            )
            return {
                "success": True,
                "data": result
            }
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"[Asset Discovery] Error getting asset inventory summary: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Export available tools
__all__ = [
    'discover_gcp_resources',
    'get_compute_instances', 
    'get_storage_buckets',
    'get_cloud_functions',
    'get_databases',
    'get_kubernetes_clusters',
    'analyze_security_assets',
    'search_assets_by_name',
    'get_asset_inventory_summary'
]