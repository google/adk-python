#!/usr/bin/env python3
"""
Unified FastAPI application for GCP data fetching and BigQuery operations
Consolidates 13 Cloud Functions into a single modular API

Endpoints:
  - /health - Health check
  - /api/v1/iam/* - IAM data endpoints
  - /api/v1/compute/* - Compute data endpoints
  - /api/v1/network/* - Network data endpoints
  - /api/v1/storage/* - Storage data endpoints
  - /api/v1/security/* - Security findings endpoints
  - /api/v1/feeds/* - External feeds endpoints
  - /api/v1/admin/* - Table management endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
from datetime import datetime
import os

from .models import (
    IAMAccount, CustomRole, ServiceAccountRole,
    ComputeInstance, FirewallRule, Network,
    StorageBucket, SecurityFinding, SecurityFeed,
    ReleaseNote, ConfluencePage,
    DataFetchResponse, BulkInsertResponse, HealthCheckResponse
)
from .bigquery_ops import BigQueryOperations
from .fetchers import (
    IAMFetcher, ComputeFetcher, NetworkFetcher,
    StorageFetcher, SecurityFetcher, FeedFetcher
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GCP Data Fetching API",
    description="Unified API for fetching GCP resources and syncing to BigQuery",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize BigQuery operations
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "mgm-digitalconcierge")
DATASET_ID = os.getenv("BIGQUERY_DATASET", "security_insights")
bq_ops = BigQueryOperations(project_id=PROJECT_ID, dataset_id=DATASET_ID)

# Initialize fetchers
iam_fetcher = IAMFetcher(PROJECT_ID)
compute_fetcher = ComputeFetcher(PROJECT_ID)
network_fetcher = NetworkFetcher(PROJECT_ID)
storage_fetcher = StorageFetcher(PROJECT_ID)
security_fetcher = SecurityFetcher(PROJECT_ID)
feed_fetcher = FeedFetcher()


# ============================================================================
# Health & Admin Endpoints
# ============================================================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test BigQuery connection
        bq_connected = bq_ops.ensure_dataset_exists()

        return HealthCheckResponse(
            status="healthy" if bq_connected else "degraded",
            bigquery_connected=bq_connected,
            services_available={
                "iam": True,
                "compute": True,
                "storage": True,
                "network": True,
                "security": True
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            bigquery_connected=False,
            services_available={}
        )


@app.post("/api/v1/admin/create-tables")
async def create_all_tables(overwrite: bool = Query(False, description="Overwrite existing tables")):
    """Create all BigQuery tables from Pydantic models"""
    try:
        results = bq_ops.create_all_tables(overwrite=overwrite)

        total_tables = len(results)
        successful = sum(1 for success in results.values() if success)

        return {
            "success": successful == total_tables,
            "message": f"Created {successful}/{total_tables} tables",
            "tables": results
        }
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/admin/tables/{table_name}/info")
async def get_table_info(table_name: str):
    """Get metadata about a specific table"""
    try:
        info = bq_ops.get_table_info(table_name)
        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])
        return info
    except Exception as e:
        logger.error(f"Failed to get table info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# IAM Endpoints
# ============================================================================

@app.post("/api/v1/iam/accounts/fetch", response_model=DataFetchResponse)
async def fetch_iam_accounts(
    background_tasks: BackgroundTasks,
    sync_to_bq: bool = Query(True, description="Sync results to BigQuery")
):
    """Fetch all IAM accounts and optionally sync to BigQuery"""
    start_time = datetime.utcnow()

    try:
        # Fetch IAM accounts
        accounts = iam_fetcher.fetch_iam_accounts()

        records_fetched = len(accounts)
        records_inserted = 0

        # Sync to BigQuery if requested
        if sync_to_bq and accounts:
            result = bq_ops.insert_records("iam_accounts", accounts)
            records_inserted = result["inserted"]

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Fetched {records_fetched} IAM accounts",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="iam_accounts" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to fetch IAM accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/iam/custom-roles/fetch", response_model=DataFetchResponse)
async def fetch_custom_roles(sync_to_bq: bool = True):
    """Fetch custom IAM roles"""
    start_time = datetime.utcnow()

    try:
        roles = iam_fetcher.fetch_custom_roles()
        records_fetched = len(roles)
        records_inserted = 0

        if sync_to_bq and roles:
            result = bq_ops.insert_records("custom_roles", roles)
            records_inserted = result["inserted"]

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Fetched {records_fetched} custom roles",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="custom_roles" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to fetch custom roles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/iam/service-accounts/fetch", response_model=DataFetchResponse)
async def fetch_service_account_roles(sync_to_bq: bool = True):
    """Fetch service account roles"""
    start_time = datetime.utcnow()

    try:
        sa_roles = iam_fetcher.fetch_service_account_roles()
        records_fetched = len(sa_roles)
        records_inserted = 0

        if sync_to_bq and sa_roles:
            result = bq_ops.insert_records("service_account_roles", sa_roles)
            records_inserted = result["inserted"]

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Fetched {records_fetched} service account roles",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="service_account_roles" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to fetch service account roles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Compute Endpoints
# ============================================================================

@app.post("/api/v1/compute/instances/fetch", response_model=DataFetchResponse)
async def fetch_compute_instances(
    sync_to_bq: bool = True,
    zones: Optional[List[str]] = Query(None, description="Specific zones to fetch from")
):
    """Fetch compute instances from all zones or specific zones"""
    start_time = datetime.utcnow()

    try:
        instances = compute_fetcher.fetch_compute_instances(zones=zones)
        records_fetched = len(instances)
        records_inserted = 0

        if sync_to_bq and instances:
            result = bq_ops.insert_records("compute_instances", instances)
            records_inserted = result["inserted"]

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Fetched {records_fetched} compute instances",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="compute_instances" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to fetch compute instances: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Network Endpoints
# ============================================================================

@app.post("/api/v1/network/firewall-rules/fetch", response_model=DataFetchResponse)
async def fetch_firewall_rules(sync_to_bq: bool = True):
    """Fetch VPC firewall rules"""
    start_time = datetime.utcnow()

    try:
        rules = network_fetcher.fetch_firewall_rules()
        records_fetched = len(rules)
        records_inserted = 0

        if sync_to_bq and rules:
            result = bq_ops.insert_records("firewall_rules", rules)
            records_inserted = result["inserted"]

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Fetched {records_fetched} firewall rules",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="firewall_rules" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to fetch firewall rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/network/networks/fetch", response_model=DataFetchResponse)
async def fetch_networks(sync_to_bq: bool = True):
    """Fetch VPC networks"""
    start_time = datetime.utcnow()

    try:
        networks = network_fetcher.fetch_networks()
        records_fetched = len(networks)
        records_inserted = 0

        if sync_to_bq and networks:
            result = bq_ops.insert_records("networks", networks)
            records_inserted = result["inserted"]

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Fetched {records_fetched} networks",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="networks" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to fetch networks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Storage Endpoints
# ============================================================================

@app.post("/api/v1/storage/buckets/fetch", response_model=DataFetchResponse)
async def fetch_storage_buckets(sync_to_bq: bool = True):
    """Fetch Cloud Storage buckets"""
    start_time = datetime.utcnow()

    try:
        buckets = storage_fetcher.fetch_storage_buckets()
        records_fetched = len(buckets)
        records_inserted = 0

        if sync_to_bq and buckets:
            result = bq_ops.insert_records("storage_buckets", buckets)
            records_inserted = result["inserted"]

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Fetched {records_fetched} storage buckets",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="storage_buckets" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to fetch storage buckets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Security Endpoints
# ============================================================================

@app.post("/api/v1/security/findings/fetch", response_model=DataFetchResponse)
async def fetch_security_findings(
    sync_to_bq: bool = True,
    min_severity: Optional[str] = Query(None, description="Minimum severity (CRITICAL, HIGH, MEDIUM, LOW)")
):
    """Fetch Security Command Center findings"""
    start_time = datetime.utcnow()

    try:
        findings = security_fetcher.fetch_security_findings(min_severity=min_severity)
        records_fetched = len(findings)
        records_inserted = 0

        if sync_to_bq and findings:
            result = bq_ops.insert_records("security_findings", findings)
            records_inserted = result["inserted"]

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Fetched {records_fetched} security findings",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="security_findings" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to fetch security findings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Feeds & Documentation Endpoints
# ============================================================================

@app.post("/api/v1/feeds/security/fetch", response_model=DataFetchResponse)
async def fetch_security_feeds(sync_to_bq: bool = True):
    """Fetch external security feeds (NVD, CISA, etc.)"""
    start_time = datetime.utcnow()

    try:
        feeds = feed_fetcher.fetch_security_feeds()
        records_fetched = len(feeds)
        records_inserted = 0

        if sync_to_bq and feeds:
            result = bq_ops.insert_records("security_feeds", feeds)
            records_inserted = result["inserted"]

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Fetched {records_fetched} security feeds",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="security_feeds" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to fetch security feeds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/feeds/release-notes/fetch", response_model=DataFetchResponse)
async def fetch_release_notes(sync_to_bq: bool = True):
    """Fetch GCP release notes"""
    start_time = datetime.utcnow()

    try:
        notes = feed_fetcher.fetch_release_notes()
        records_fetched = len(notes)
        records_inserted = 0

        if sync_to_bq and notes:
            result = bq_ops.insert_records("release_notes", notes)
            records_inserted = result["inserted"]

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Fetched {records_fetched} release notes",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="release_notes" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to fetch release notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/feeds/confluence/sync", response_model=DataFetchResponse)
async def sync_confluence_pages(
    sync_to_bq: bool = True,
    space_key: Optional[str] = Query(None, description="Specific Confluence space to sync")
):
    """Sync Confluence documentation pages"""
    start_time = datetime.utcnow()

    try:
        pages = feed_fetcher.fetch_confluence_pages(space_key=space_key)
        records_fetched = len(pages)
        records_inserted = 0

        if sync_to_bq and pages:
            result = bq_ops.upsert_records("confluence_pages", pages, key_fields=["page_id"])
            records_inserted = result.get("upserted", 0)

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DataFetchResponse(
            success=True,
            message=f"Synced {records_fetched} Confluence pages",
            records_fetched=records_fetched,
            records_inserted=records_inserted,
            table_name="confluence_pages" if sync_to_bq else None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Failed to sync Confluence pages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Batch Operations
# ============================================================================

@app.post("/api/v1/batch/fetch-all")
async def fetch_all_data(background_tasks: BackgroundTasks):
    """Trigger fetch for all data sources (runs in background)"""

    async def run_all_fetchers():
        """Background task to fetch all data"""
        results = {}

        # IAM
        try:
            accounts = iam_fetcher.fetch_iam_accounts()
            bq_ops.insert_records("iam_accounts", accounts)
            results["iam_accounts"] = len(accounts)
        except Exception as e:
            logger.error(f"IAM accounts failed: {e}")
            results["iam_accounts"] = 0

        # Compute
        try:
            instances = compute_fetcher.fetch_compute_instances()
            bq_ops.insert_records("compute_instances", instances)
            results["compute_instances"] = len(instances)
        except Exception as e:
            logger.error(f"Compute instances failed: {e}")
            results["compute_instances"] = 0

        # Storage
        try:
            buckets = storage_fetcher.fetch_storage_buckets()
            bq_ops.insert_records("storage_buckets", buckets)
            results["storage_buckets"] = len(buckets)
        except Exception as e:
            logger.error(f"Storage buckets failed: {e}")
            results["storage_buckets"] = 0

        # Network
        try:
            rules = network_fetcher.fetch_firewall_rules()
            bq_ops.insert_records("firewall_rules", rules)
            results["firewall_rules"] = len(rules)
        except Exception as e:
            logger.error(f"Firewall rules failed: {e}")
            results["firewall_rules"] = 0

        logger.info(f"Batch fetch complete: {results}")

    background_tasks.add_task(run_all_fetchers)

    return {
        "success": True,
        "message": "Batch fetch started in background",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
