"""
Data refresh API endpoints for triggering comprehensive data fetches.

This module provides endpoints to:
- Trigger full data refresh
- Check refresh status
- Query cached data directly
- Get data statistics
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

# Import data fetcher
try:
    from ..services.data_fetcher import DataFetcher, fetch_all_project_data
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    try:
        # Fallback for when running from backend directory directly
        from ..services.data_fetcher import DataFetcher, fetch_all_project_data
        DATA_FETCHER_AVAILABLE = True
    except ImportError:
        DATA_FETCHER_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()

class RefreshRequest(BaseModel):
    """Request to refresh data for a project."""
    project_id: str
    force_refresh: bool = False
    fetch_types: Optional[list] = None  # Specific types to fetch


class QueryRequest(BaseModel):
    """Request to query cached data."""
    project_id: str
    resource_type: str  # compute, storage, networks, etc.
    filters: Optional[Dict[str, Any]] = None


# Global store for tracking refresh jobs
_refresh_jobs = {}


async def run_data_refresh(project_id: str, job_id: str):
    """Background task to run data refresh."""
    try:
        _refresh_jobs[job_id] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "project_id": project_id
        }
        
        logger.info(f"Starting data refresh for project {project_id}")
        
        # Run the comprehensive fetch
        result = await fetch_all_project_data(project_id)
        
        _refresh_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result": result
        })
        
        logger.info(f"Data refresh completed for project {project_id}")
        
    except Exception as e:
        logger.error(f"Data refresh failed for project {project_id}: {e}")
        _refresh_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })


@router.post("/refresh")
async def trigger_data_refresh(
    request: RefreshRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger a comprehensive data refresh for a project.
    
    This will fetch all GCP resources and store them locally for fast querying.
    """
    if not DATA_FETCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data fetcher not available")
    
    # Generate job ID
    job_id = f"{request.project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start background refresh
    background_tasks.add_task(run_data_refresh, request.project_id, job_id)
    
    return {
        "success": True,
        "job_id": job_id,
        "project_id": request.project_id,
        "message": "Data refresh started in background",
        "status_url": f"/api/v1/data/refresh/status/{job_id}"
    }


@router.get("/refresh/status/{job_id}")
async def get_refresh_status(job_id: str):
    """Get status of a data refresh job."""
    
    if job_id not in _refresh_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return _refresh_jobs[job_id]


@router.post("/query")
async def query_cached_data(request: QueryRequest):
    """
    Query cached data directly from local database.
    
    This provides fast access to previously fetched data.
    """
    if not DATA_FETCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data fetcher not available")
    
    try:
        fetcher = DataFetcher(request.project_id)
        
        # Route to appropriate query method based on resource type
        if request.resource_type == "compute":
            status_filter = request.filters.get("status") if request.filters else None
            results = fetcher.query_compute_instances(status=status_filter)
        
        elif request.resource_type == "storage":
            public_only = request.filters.get("public_only", False) if request.filters else False
            results = fetcher.query_storage_buckets(public_only=public_only)
        
        elif request.resource_type == "security":
            severity = request.filters.get("severity") if request.filters else None
            results = fetcher.query_security_findings(severity=severity)
        
        elif request.resource_type == "summary":
            results = fetcher.get_summary_stats()
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported resource type: {request.resource_type}")
        
        return {
            "success": True,
            "project_id": request.project_id,
            "resource_type": request.resource_type,
            "count": len(results) if isinstance(results, list) else 1,
            "data": results,
            "from_cache": True
        }
        
    except Exception as e:
        logger.error(f"Error querying cached data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{project_id}")
async def get_data_stats(project_id: str):
    """Get statistics about cached data for a project."""
    if not DATA_FETCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data fetcher not available")
    
    try:
        fetcher = DataFetcher(project_id)
        stats = fetcher.get_summary_stats()
        
        return {
            "success": True,
            "project_id": project_id,
            "stats": stats,
            "cached": True
        }
        
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets/{project_id}")
async def get_cached_assets(
    project_id: str,
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
    limit: int = Query(100, description="Maximum results to return")
):
    """
    Get assets from cache - fast local query.
    
    This replaces the slow /api/v1/assets/list endpoint for cached data.
    """
    if not DATA_FETCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data fetcher not available")
    
    try:
        fetcher = DataFetcher(project_id)
        
        # Query compute instances as primary assets
        compute_instances = fetcher.query_compute_instances()
        storage_buckets = fetcher.query_storage_buckets()
        
        # Combine into unified asset format
        assets = []
        
        # Add compute instances
        for instance in compute_instances[:limit//2]:
            assets.append({
                "name": instance["name"],
                "asset_type": "compute.googleapis.com/Instance",
                "display_name": instance["name"],
                "location": instance["zone"],
                "state": instance["status"],
                "labels": instance["labels"] if isinstance(instance["labels"], dict) else {},
                "resource_data": {
                    "machine_type": instance["machine_type"],
                    "internal_ip": instance["internal_ip"],
                    "external_ip": instance["external_ip"]
                }
            })
        
        # Add storage buckets
        for bucket in storage_buckets[:limit//2]:
            assets.append({
                "name": bucket["name"],
                "asset_type": "storage.googleapis.com/Bucket",
                "display_name": bucket["name"],
                "location": bucket["location"],
                "state": "ACTIVE",
                "labels": bucket["labels"] if isinstance(bucket["labels"], dict) else {},
                "resource_data": {
                    "storage_class": bucket["storage_class"],
                    "public_access": bucket["public_access"],
                    "encryption": bucket["encryption"]
                }
            })
        
        # Filter by asset type if specified
        if asset_type:
            assets = [a for a in assets if a["asset_type"] == asset_type]
        
        return {
            "success": True,
            "project_id": project_id,
            "assets": assets[:limit],
            "total_count": len(assets),
            "from_cache": True,
            "fast_query": True
        }
        
    except Exception as e:
        logger.error(f"Error getting cached assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/findings/{project_id}")
async def get_cached_findings(
    project_id: str,
    severity: Optional[str] = Query(None, description="Filter by severity"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, description="Maximum results to return")
):
    """
    Get security findings from cache - fast local query.
    
    This replaces slow Security Command Center calls.
    """
    if not DATA_FETCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data fetcher not available")
    
    try:
        fetcher = DataFetcher(project_id)
        findings = fetcher.query_security_findings(severity=severity)
        
        # Filter by category if specified
        if category:
            findings = [f for f in findings if f.get("category") == category]
        
        # Convert to API format
        formatted_findings = []
        for finding in findings[:limit]:
            formatted_findings.append({
                "name": finding["id"],
                "category": finding["category"],
                "severity": finding["severity"],
                "state": finding["state"],
                "resource_name": finding["resource_name"],
                "finding_class": finding["finding_class"],
                "description": finding["description"],
                "recommendation": finding["recommendation"],
                "event_time": finding["event_time"]
            })
        
        return {
            "success": True,
            "project_id": project_id,
            "findings": formatted_findings,
            "total_count": len(formatted_findings),
            "from_cache": True,
            "fast_query": True
        }
        
    except Exception as e:
        logger.error(f"Error getting cached findings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/{project_id}")
async def clear_cache(project_id: str):
    """Clear cached data for a project."""
    if not DATA_FETCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data fetcher not available")
    
    try:
        fetcher = DataFetcher(project_id)
        
        # Clear all cached data
        with fetcher._get_connection() as conn:
            tables = [
                "assets", "compute_instances", "storage_buckets",
                "networks", "firewall_rules", "iam_accounts", 
                "databases", "security_findings", "fetch_status"
            ]
            
            for table in tables:
                conn.execute(f"DELETE FROM {table} WHERE project_id = ?", [project_id])
            
            conn.commit()
        
        return {
            "success": True,
            "message": f"Cache cleared for project {project_id}",
            "project_id": project_id
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warmup/{project_id}")
async def warmup_cache(project_id: str, background_tasks: BackgroundTasks):
    """
    Warm up the cache by fetching common data.
    
    This is a lighter version of full refresh that gets essential data quickly.
    """
    if not DATA_FETCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data fetcher not available")
    
    # Generate job ID
    job_id = f"warmup_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def warmup_job():
        try:
            _refresh_jobs[job_id] = {
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "project_id": project_id,
                "type": "warmup"
            }
            
            fetcher = DataFetcher(project_id)
            
            # Fetch just the most important data
            results = await asyncio.gather(
                fetcher._fetch_compute_instances(),
                fetcher._fetch_storage_buckets(),
                fetcher._fetch_security_findings(),
                return_exceptions=True
            )
            
            _refresh_jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "results": results
            })
            
        except Exception as e:
            _refresh_jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": str(e)
            })
    
    # Start warmup in background
    background_tasks.add_task(warmup_job)
    
    return {
        "success": True,
        "job_id": job_id,
        "project_id": project_id,
        "type": "warmup",
        "message": "Cache warmup started",
        "status_url": f"/api/v1/data/refresh/status/{job_id}"
    }


# Add router info
router.tags = ["Data Management"]
router.responses = {
    503: {"description": "Data fetcher service unavailable"},
    500: {"description": "Internal server error"},
    404: {"description": "Resource not found"}
}