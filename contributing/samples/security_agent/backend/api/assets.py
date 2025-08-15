"""
Unified Google Cloud Asset Inventory API for comprehensive GCP service coverage
Provides natural language access to ALL GCP resources via Asset Inventory API
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
import time
import os
import re
from google.cloud import asset_v1
from google.oauth2 import service_account
import google.auth

logger = logging.getLogger(__name__)
router = APIRouter()

def _get_credentials():
    """Initialize Google Cloud credentials for real API calls"""
    try:
        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_path and os.path.exists(creds_path):
            logger.info(f"🔐 Using service account credentials from {creds_path}")
            return service_account.Credentials.from_service_account_file(creds_path)
        else:
            logger.info("🔐 Using default Google Cloud credentials")
            credentials, project = google.auth.default()
            return credentials
    except Exception as e:
        logger.warning(f"⚠️ Authentication failed, will use mock data: {e}")
        return None

# Resource type mappings for natural language queries
RESOURCE_TYPE_MAPPINGS = {
    'compute': ['compute.googleapis.com/Instance', 'compute.googleapis.com/Disk', 'compute.googleapis.com/Snapshot'],
    'storage': ['storage.googleapis.com/Bucket'],
    'database': ['sqladmin.googleapis.com/Instance', 'spanner.googleapis.com/Instance', 'bigquery.googleapis.com/Dataset'],
    'network': ['compute.googleapis.com/Network', 'compute.googleapis.com/Firewall', 'compute.googleapis.com/Route'],
    'kubernetes': ['container.googleapis.com/Cluster'],
    'functions': ['cloudfunctions.googleapis.com/CloudFunction'],
    'iam': ['iam.googleapis.com/ServiceAccount'],
    'monitoring': ['monitoring.googleapis.com/AlertPolicy'],
    'dns': ['dns.googleapis.com/ManagedZone']
}

def _parse_natural_language_query(query: str) -> List[str]:
    """Parse natural language query to determine relevant asset types"""
    query_lower = query.lower()
    asset_types = []
    
    # Map common terms to asset types
    if any(term in query_lower for term in ['compute', 'instance', 'vm', 'server']):
        asset_types.extend(RESOURCE_TYPE_MAPPINGS['compute'])
    if any(term in query_lower for term in ['storage', 'bucket', 'blob']):
        asset_types.extend(RESOURCE_TYPE_MAPPINGS['storage'])
    if any(term in query_lower for term in ['database', 'sql', 'spanner', 'bigquery']):
        asset_types.extend(RESOURCE_TYPE_MAPPINGS['database'])
    if any(term in query_lower for term in ['network', 'vpc', 'firewall', 'subnet']):
        asset_types.extend(RESOURCE_TYPE_MAPPINGS['network'])
    if any(term in query_lower for term in ['kubernetes', 'gke', 'cluster', 'k8s']):
        asset_types.extend(RESOURCE_TYPE_MAPPINGS['kubernetes'])
    if any(term in query_lower for term in ['function', 'lambda', 'serverless']):
        asset_types.extend(RESOURCE_TYPE_MAPPINGS['functions'])
    if any(term in query_lower for term in ['iam', 'user', 'role', 'permission']):
        asset_types.extend(RESOURCE_TYPE_MAPPINGS['iam'])
    if any(term in query_lower for term in ['monitor', 'alert', 'metric']):
        asset_types.extend(RESOURCE_TYPE_MAPPINGS['monitoring'])
    if any(term in query_lower for term in ['dns', 'domain']):
        asset_types.extend(RESOURCE_TYPE_MAPPINGS['dns'])
    
    # If no specific types found, return common ones for general queries
    if not asset_types:
        asset_types = (RESOURCE_TYPE_MAPPINGS['compute'] + 
                      RESOURCE_TYPE_MAPPINGS['storage'] + 
                      RESOURCE_TYPE_MAPPINGS['database'])
    
    return asset_types

async def _search_assets(project_id: str, asset_types: List[str] = None, 
                        query: str = None) -> Dict[str, Any]:
    """Search assets using Google Cloud Asset Inventory API"""
    
    parent = f"projects/{project_id}"
    logger.info(f"📡 Making HTTP POST to https://cloudasset.googleapis.com/v1/{parent}:searchAllResources")
    
    start_time = time.time()
    try:
        credentials = _get_credentials()
        if not credentials:
            raise Exception("No valid credentials available")
        
        # Initialize Asset Inventory client
        client = asset_v1.AssetServiceClient(credentials=credentials)
        
        # Build search request
        request = asset_v1.SearchAllResourcesRequest(
            scope=parent,
            asset_types=asset_types,
            page_size=100
        )
        
        # Add query filter if provided
        if query:
            request.query = query
        
        logger.info(f"📞 API Call: cloudasset.searchAllResources")
        logger.info(f"   🎯 Scope: {parent}")
        logger.info(f"   🔍 Asset Types: {len(asset_types) if asset_types else 'all'}")
        logger.info(f"   📝 Query: {query or 'none'}")
        
        # Execute search
        response = client.search_all_resources(request=request)
        resources = list(response)
        
        api_duration = time.time() - start_time
        logger.info(f"✅ Response received: 200 OK, {api_duration:.1f}s")
        logger.info(f"📊 Found {len(resources)} resources in project {project_id}")
        
        # Process and categorize resources
        categorized_resources = {}
        for resource in resources:
            asset_type = resource.asset_type
            category = _get_resource_category(asset_type)
            
            if category not in categorized_resources:
                categorized_resources[category] = []
            
            resource_data = {
                'name': resource.name,
                'display_name': resource.display_name,
                'asset_type': asset_type,
                'location': resource.location,
                'labels': dict(resource.labels) if resource.labels else {},
                'state': resource.state.name if resource.state else 'UNKNOWN',
                'create_time': resource.create_time.isoformat() if resource.create_time else None
            }
            categorized_resources[category].append(resource_data)
        
        return {
            "success": True,
            "resources": categorized_resources,
            "total_count": len(resources),
            "source": "real_api",
            "api_duration": api_duration
        }
        
    except Exception as e:
        api_duration = time.time() - start_time
        logger.error(f"❌ Asset Inventory API failed after {api_duration:.1f}s: {e}")
        return {
            "success": False,
            "error": str(e),
            "source": "api_failed",
            "api_duration": api_duration
        }

def _get_resource_category(asset_type: str) -> str:
    """Categorize asset types for better organization"""
    if 'compute' in asset_type.lower():
        return 'compute'
    elif 'storage' in asset_type.lower():
        return 'storage'
    elif any(db in asset_type.lower() for db in ['sql', 'spanner', 'bigquery']):
        return 'database'
    elif any(net in asset_type.lower() for net in ['network', 'firewall', 'route']):
        return 'network'
    elif 'container' in asset_type.lower():
        return 'kubernetes'
    elif 'function' in asset_type.lower():
        return 'serverless'
    elif 'iam' in asset_type.lower():
        return 'security'
    elif any(mon in asset_type.lower() for mon in ['monitoring', 'logging']):
        return 'monitoring'
    else:
        return 'other'

@router.post("/discover")
async def discover_resources(request: Dict[str, Any]):
    """Natural language resource discovery using Asset Inventory API"""
    query = request.get('query', '')
    project_id = request.get('project_id', 'mgm-digitalconcierge')
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    # Parse natural language query to determine asset types
    asset_types = _parse_natural_language_query(query)
    
    # Search for assets
    result = await _search_assets(project_id, asset_types, None)
    
    if result["success"]:
        logger.info(f"🎯 Successfully discovered resources for query: '{query}'")
        return {
            "query": query,
            "project_id": project_id,
            "data_source": result["source"],
            "api_duration": result["api_duration"],
            "resources": result["resources"],
            "total_count": result["total_count"],
            "asset_types_searched": asset_types
        }
    else:
        # Fallback to mock data for demonstration
        logger.warning(f"🔄 Using mock data for query: '{query}'")
        return {
            "query": query,
            "project_id": project_id,
            "data_source": "api_failed",
            "api_duration": result["api_duration"],
            "resources": {
                "compute": [
                    {"name": "instance-1", "asset_type": "compute.googleapis.com/Instance", "location": "us-central1-a"},
                    {"name": "disk-1", "asset_type": "compute.googleapis.com/Disk", "location": "us-central1-a"}
                ],
                "storage": [
                    {"name": "my-bucket", "asset_type": "storage.googleapis.com/Bucket", "location": "US"}
                ]
            },
            "total_count": 3,
            "asset_types_searched": asset_types,
            "note": "Using mock data - authentication failed"
        }

@router.get("/compute/instances")
async def get_compute_instances(project_id: str = Query('mgm-digitalconcierge')):
    """Get all compute instances using Asset Inventory API"""
    asset_types = ['compute.googleapis.com/Instance']
    result = await _search_assets(project_id, asset_types)
    
    if result["success"]:
        instances = result["resources"].get("compute", [])
        return {
            "project_id": project_id,
            "data_source": result["source"],
            "api_duration": result["api_duration"],
            "instances": instances,
            "count": len(instances)
        }
    else:
        return {
            "project_id": project_id,
            "data_source": "api_failed",
            "api_duration": result["api_duration"],
            "instances": [
                {"name": "web-server-1", "location": "us-central1-a", "status": "RUNNING"},
                {"name": "db-server-1", "location": "us-east1-b", "status": "RUNNING"}
            ],
            "count": 2,
            "note": "Using mock data - authentication failed"
        }

@router.get("/storage/all")
async def get_all_storage_resources(project_id: str = Query('mgm-digitalconcierge')):
    """Get all storage resources (buckets, disks, etc.) using Asset Inventory API"""
    asset_types = ['storage.googleapis.com/Bucket', 'compute.googleapis.com/Disk']
    result = await _search_assets(project_id, asset_types)
    
    if result["success"]:
        storage_resources = result["resources"].get("storage", [])
        compute_storage = result["resources"].get("compute", [])
        all_storage = storage_resources + compute_storage
        
        return {
            "project_id": project_id,
            "data_source": result["source"],
            "api_duration": result["api_duration"],
            "storage_resources": all_storage,
            "count": len(all_storage)
        }
    else:
        return {
            "project_id": project_id,
            "data_source": "api_failed",
            "api_duration": result["api_duration"],
            "storage_resources": [
                {"name": "my-app-bucket", "asset_type": "storage.googleapis.com/Bucket", "location": "US"},
                {"name": "backup-disk", "asset_type": "compute.googleapis.com/Disk", "location": "us-central1-a"}
            ],
            "count": 2,
            "note": "Using mock data - authentication failed"
        }

@router.get("/databases")
async def get_databases(project_id: str = Query('mgm-digitalconcierge')):
    """Get all database resources using Asset Inventory API"""
    asset_types = ['sqladmin.googleapis.com/Instance', 'spanner.googleapis.com/Instance', 'bigquery.googleapis.com/Dataset']
    result = await _search_assets(project_id, asset_types)
    
    if result["success"]:
        databases = result["resources"].get("database", [])
        return {
            "project_id": project_id,
            "data_source": result["source"],
            "api_duration": result["api_duration"],
            "databases": databases,
            "count": len(databases)
        }
    else:
        return {
            "project_id": project_id,
            "data_source": "api_failed",
            "api_duration": result["api_duration"],
            "databases": [
                {"name": "prod-mysql", "asset_type": "sqladmin.googleapis.com/Instance", "location": "us-central1"},
                {"name": "analytics-bq", "asset_type": "bigquery.googleapis.com/Dataset", "location": "US"}
            ],
            "count": 2,
            "note": "Using mock data - authentication failed"
        }

@router.get("/summary")
async def get_project_summary(project_id: str = Query('mgm-digitalconcierge')):
    """Get comprehensive project resource summary using Asset Inventory API"""
    
    # Search for major resource types
    major_asset_types = [
        'compute.googleapis.com/Instance',
        'storage.googleapis.com/Bucket', 
        'sqladmin.googleapis.com/Instance',
        'container.googleapis.com/Cluster',
        'cloudfunctions.googleapis.com/CloudFunction'
    ]
    
    result = await _search_assets(project_id, major_asset_types)
    
    if result["success"]:
        resources = result["resources"]
        summary = {
            "compute_instances": len(resources.get("compute", [])),
            "storage_buckets": len(resources.get("storage", [])),
            "databases": len(resources.get("database", [])),
            "kubernetes_clusters": len(resources.get("kubernetes", [])),
            "cloud_functions": len(resources.get("serverless", []))
        }
        
        return {
            "project_id": project_id,
            "data_source": result["source"],
            "api_duration": result["api_duration"],
            "summary": summary,
            "total_resources": result["total_count"],
            "categories": list(resources.keys())
        }
    else:
        return {
            "project_id": project_id,
            "data_source": "api_failed",
            "api_duration": result["api_duration"],
            "summary": {
                "compute_instances": 3,
                "storage_buckets": 5,
                "databases": 2,
                "kubernetes_clusters": 1,
                "cloud_functions": 4
            },
            "total_resources": 15,
            "categories": ["compute", "storage", "database", "kubernetes", "serverless"],
            "note": "Using mock data - authentication failed"
        }