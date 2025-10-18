"""
Unified Cloud Functions ASGI Application

This module provides a FastAPI application that consolidates all fetch functions
into a single deployable surface using Vellox for Cloud Functions compatibility.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import traceback
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fetchers import FETCHERS_REGISTRY
from shared import Config, create_response, create_error_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Unified Security Data Fetchers",
    description="Consolidated Cloud Functions for fetching GCP security data",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


class FetchRequest(BaseModel):
    """Request model for fetch operations"""
    fetcher: Optional[str] = None
    async_mode: bool = False
    force_refresh: bool = False


class FetchResponse(BaseModel):
    """Response model for fetch operations"""
    status: str
    message: str
    fetcher: Optional[str] = None
    records_processed: Optional[int] = None
    table: Optional[str] = None
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        Config.validate()
        logger.info(f"Application started - Project: {Config.PROJECT_ID}")
    except Exception as e:
        logger.error(f"Startup validation failed: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint - health check and info"""
    return {
        "service": "Unified Security Data Fetchers",
        "version": "2.0.0",
        "status": "healthy",
        "project": Config.PROJECT_ID,
        "endpoints": {
            "fetch": "/fetch/{fetcher_name}",
            "fetch_all": "/fetch/all",
            "list_fetchers": "/fetchers",
            "health": "/health",
            "docs": "/docs"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health checks
        Config.validate()
        return JSONResponse(
            content={
                "status": "healthy",
                "checks": {
                    "config": "valid",
                    "project_id": Config.PROJECT_ID,
                    "dataset": Config.BQ_DATASET_ID
                },
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=503
        )


@app.get("/fetchers")
async def list_fetchers():
    """List all available fetchers"""
    fetchers_info = []

    for name, fetcher_class in FETCHERS_REGISTRY.items():
        fetcher = fetcher_class()
        fetchers_info.append({
            "name": name,
            "table": fetcher.table_name,
            "dataset": fetcher.dataset_id,
            "endpoint": f"/fetch/{name}"
        })

    return {
        "fetchers": fetchers_info,
        "total": len(fetchers_info),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/fetch/{fetcher_name}")
async def fetch_data(
    fetcher_name: str,
    request: Optional[FetchRequest] = None
):
    """
    Execute a specific fetcher

    This endpoint triggers data fetching for a specific source.
    Cloud Scheduler can call this endpoint to trigger individual fetchers.
    """
    if fetcher_name not in FETCHERS_REGISTRY:
        if fetcher_name == "all":
            return await fetch_all_data()
        raise HTTPException(
            status_code=404,
            detail=f"Fetcher '{fetcher_name}' not found. Available: {list(FETCHERS_REGISTRY.keys())}"
        )

    request = request or FetchRequest(fetcher=fetcher_name)

    try:
        fetcher_class = FETCHERS_REGISTRY[fetcher_name]
        fetcher = fetcher_class()

        # Execute synchronously for Cloud Functions
        # (Cloud Functions handle concurrency at the instance level)
        logger.info(f"Executing fetcher: {fetcher_name}")
        result = fetcher.run()

        # Add fetcher name to result
        result['fetcher'] = fetcher_name

        return FetchResponse(
            status=result.get('status', 'unknown'),
            message=result.get('message', ''),
            fetcher=fetcher_name,
            records_processed=result.get('records_processed', 0),
            table=result.get('table'),
            timestamp=datetime.utcnow().isoformat(),
            metadata=result
        )

    except Exception as e:
        logger.error(f"Fetcher {fetcher_name} failed: {e}")
        logger.error(traceback.format_exc())

        return FetchResponse(
            status="error",
            message=str(e),
            fetcher=fetcher_name,
            records_processed=0,
            timestamp=datetime.utcnow().isoformat()
        )


@app.post("/fetch/all")
async def fetch_all_data():
    """
    Execute all fetchers

    This endpoint triggers all available fetchers sequentially.
    Use with caution as it may take significant time.
    """
    results = {}
    total_records = 0
    failed = []
    succeeded = []

    for fetcher_name, fetcher_class in FETCHERS_REGISTRY.items():
        try:
            logger.info(f"Executing fetcher: {fetcher_name}")
            fetcher = fetcher_class()
            result = fetcher.run()

            results[fetcher_name] = result
            total_records += result.get('records_processed', 0)

            if result.get('status') == 'error':
                failed.append(fetcher_name)
            else:
                succeeded.append(fetcher_name)

        except Exception as e:
            logger.error(f"Fetcher {fetcher_name} failed: {e}")
            results[fetcher_name] = {
                'status': 'error',
                'message': str(e),
                'records_processed': 0
            }
            failed.append(fetcher_name)

    return {
        "status": "completed",
        "summary": {
            "total_fetchers": len(results),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "total_records": total_records
        },
        "succeeded": succeeded,
        "failed": failed,
        "details": results,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc) if Config.ENABLE_SAMPLE_DATA else "An error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Cloud Scheduler compatibility endpoints
# These allow Cloud Scheduler to trigger individual fetchers using GET requests
def _register_trigger_endpoint(fetcher_name: str) -> None:
    """Register a GET endpoint for Cloud Scheduler compatibility."""

    @app.get(f"/trigger/{fetcher_name}", name=f"trigger_{fetcher_name}")
    async def trigger_fetcher() -> FetchResponse:
        return await fetch_data(fetcher_name)


for _fetcher_name in FETCHERS_REGISTRY.keys():
    _register_trigger_endpoint(_fetcher_name)
