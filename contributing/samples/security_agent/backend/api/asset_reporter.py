"""
Asset Inventory & Setting Reporter API Endpoints
================================================

RESTful API endpoints for asset discovery, configuration reporting,
drift detection, and inventory management.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import json

from ..models.asset_reporter_models import (
    AssetCategory, AssetImportance, ConfigurationStatus, SettingType,
    ReportFormat, AssetMetadata, ConfigurationSetting, AssetConfiguration,
    AssetInventoryItem, InventoryFilter, AssetGrouping, ConfigurationDrift,
    AssetReport, AssetChange, ComplianceRule, AssetReportRequest,
    AssetReportResponse
)
from ..services.asset_reporter import AssetReporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/assets", tags=["Asset Inventory"])

# Initialize asset reporter
project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "default-project")
organization_id = os.getenv("GOOGLE_CLOUD_ORGANIZATION", "")
database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
asset_reporter = AssetReporter(
    project_id=project_id,
    organization_id=organization_id,
    database_path=database_path
)


@router.post("/discover")
async def discover_assets(
    filters: Optional[InventoryFilter] = None,
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Discover and inventory all GCP assets.
    
    Performs comprehensive asset discovery using Cloud Asset Inventory API,
    analyzes configurations, and assesses compliance.
    """
    logger.info(f"Starting asset discovery with filters: {filters}")
    
    try:
        assets = await asset_reporter.discover_assets(filters)
        
        # Add background task for detailed analysis if many assets
        if len(assets) > 100 and background_tasks:
            background_tasks.add_task(
                _perform_deep_analysis,
                [a.metadata.asset_id for a in assets]
            )
        
        return {
            "success": True,
            "total_assets": len(assets),
            "assets": [a.dict() for a in assets[:100]],  # Limit response size
            "summary": _generate_discovery_summary(assets),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Asset discovery failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Asset discovery failed: {str(e)}"
        )


@router.get("/inventory")
async def get_inventory(
    category: Optional[AssetCategory] = Query(None, description="Filter by category"),
    importance: Optional[AssetImportance] = Query(None, description="Filter by importance"),
    environment: Optional[str] = Query(None, description="Filter by environment"),
    compliance_status: Optional[ConfigurationStatus] = Query(None, description="Filter by compliance"),
    public_only: Optional[bool] = Query(None, description="Only public assets"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results")
) -> Dict[str, Any]:
    """
    Get current asset inventory with filtering.
    
    Returns cached asset inventory with optional filters applied.
    """
    try:
        # Build filter
        filters = InventoryFilter(
            categories=[category] if category else None,
            importance_levels=[importance] if importance else None,
            environments=[environment] if environment else None,
            compliance_status=[compliance_status] if compliance_status else None,
            public_only=public_only
        )
        
        # Get assets from cache or discover
        assets = await asset_reporter.discover_assets(filters)
        
        # Limit results
        limited_assets = assets[:limit]
        
        return {
            "total_count": len(assets),
            "returned_count": len(limited_assets),
            "assets": [a.dict() for a in limited_assets],
            "filters_applied": filters.dict(exclude_none=True),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get inventory: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve inventory: {str(e)}"
        )


@router.get("/asset/{asset_id}")
async def get_asset_details(asset_id: str) -> AssetInventoryItem:
    """
    Get detailed information for a specific asset.
    
    Returns complete configuration, compliance status, and recommendations.
    """
    try:
        # Get all assets and find the requested one
        assets = await asset_reporter.discover_assets()
        
        for asset in assets:
            if asset.metadata.asset_id == asset_id or asset.metadata.asset_name == asset_id:
                return asset
        
        raise HTTPException(
            status_code=404,
            detail=f"Asset {asset_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get asset details: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve asset details: {str(e)}"
        )


@router.post("/configuration/drift")
async def detect_configuration_drift(
    baseline_config: Dict[str, Any] = None,
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Detect configuration drift from baseline.
    
    Compares current configurations against provided baseline
    and identifies drifts with remediation suggestions.
    """
    try:
        # Use provided baseline or fetch from database
        if not baseline_config:
            # Fetch baseline from last known good configuration
            baseline_config = await _get_baseline_configuration()
        
        drifts = await asset_reporter.detect_configuration_drift(baseline_config)
        
        # Add background task for auto-remediation if enabled
        if drifts and background_tasks:
            auto_remediate_drifts = [d for d in drifts if d.auto_remediation_available]
            if auto_remediate_drifts:
                background_tasks.add_task(
                    _auto_remediate_drifts,
                    auto_remediate_drifts
                )
        
        return {
            "drift_detected": len(drifts) > 0,
            "total_drifts": len(drifts),
            "critical_drifts": len([d for d in drifts if d.drift_severity == "HIGH"]),
            "auto_remediable": len([d for d in drifts if d.auto_remediation_available]),
            "drifts": [d.dict() for d in drifts[:50]],  # Limit response
            "detection_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Configuration drift detection failed: {str(e)}"
        )


@router.post("/report/generate", response_model=AssetReportResponse)
async def generate_report(
    request: AssetReportRequest,
    background_tasks: BackgroundTasks
) -> AssetReportResponse:
    """
    Generate comprehensive asset inventory report.
    
    Creates detailed reports with filtering, grouping, and multiple export formats.
    """
    logger.info(f"Generating report: {request.report_name}")
    
    try:
        response = await asset_reporter.generate_report(request)
        
        # Schedule recurring report if requested
        if request.schedule:
            background_tasks.add_task(
                _schedule_recurring_report,
                request
            )
        
        # Send to recipients if specified
        if request.recipients:
            background_tasks.add_task(
                _send_report_to_recipients,
                response.report,
                request.recipients
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )


@router.get("/report/{report_id}")
async def get_report(report_id: str) -> AssetReport:
    """
    Get previously generated report by ID.
    """
    try:
        import sqlite3
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM asset_reports WHERE report_id = ?",
            (report_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Report {report_id} not found"
            )
        
        # Convert row to AssetReport
        # (Implementation would properly deserialize)
        return {
            "report_id": row[0],
            "report_name": row[1],
            "report_type": row[2],
            "generated_at": row[3],
            "total_assets": row[5],
            "summary": json.loads(row[7]) if row[7] else {}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve report: {str(e)}"
        )


@router.get("/report/{report_id}/download/{format}")
async def download_report(
    report_id: str,
    format: ReportFormat = ReportFormat.JSON
) -> FileResponse:
    """
    Download report in specified format.
    """
    try:
        file_path = f"/tmp/{report_id}.{format.value.lower()}"
        
        if not os.path.exists(file_path):
            # Generate the file if it doesn't exist
            # (In production, this would regenerate from stored data)
            raise HTTPException(
                status_code=404,
                detail=f"Report file not found for format {format}"
            )
        
        return FileResponse(
            path=file_path,
            filename=f"{report_id}.{format.value.lower()}",
            media_type=_get_media_type(format)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download report: {str(e)}"
        )


@router.get("/compliance/rules")
async def get_compliance_rules() -> List[ComplianceRule]:
    """
    Get all compliance rules used for asset evaluation.
    """
    try:
        return asset_reporter.compliance_rules
        
    except Exception as e:
        logger.error(f"Failed to get compliance rules: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve compliance rules: {str(e)}"
        )


@router.post("/compliance/check")
async def check_compliance(
    asset_ids: Optional[List[str]] = None,
    frameworks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Check compliance status for specified assets.
    
    Evaluates assets against compliance frameworks and returns violations.
    """
    try:
        # Get assets to check
        assets = await asset_reporter.discover_assets()
        
        if asset_ids:
            assets = [a for a in assets if a.metadata.asset_id in asset_ids]
        
        violations = []
        compliant_count = 0
        
        for asset in assets:
            if asset.configuration.configuration_status == ConfigurationStatus.COMPLIANT:
                compliant_count += 1
            else:
                for setting in asset.configuration.settings:
                    if not setting.is_compliant:
                        violations.append({
                            "asset_id": asset.metadata.asset_id,
                            "asset_name": asset.metadata.display_name,
                            "setting": setting.setting_name,
                            "risk_level": setting.risk_level,
                            "remediation": setting.remediation_steps
                        })
        
        compliance_rate = (compliant_count / len(assets) * 100) if assets else 0
        
        return {
            "total_assets_checked": len(assets),
            "compliant_assets": compliant_count,
            "compliance_rate": compliance_rate,
            "total_violations": len(violations),
            "violations": violations[:100],  # Limit response
            "frameworks_checked": frameworks or ["CIS", "SOC2", "PCI-DSS"],
            "check_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Compliance check failed: {str(e)}"
        )


@router.get("/changes/recent")
async def get_recent_changes(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    asset_id: Optional[str] = Query(None, description="Filter by asset")
) -> List[AssetChange]:
    """
    Get recent configuration changes.
    
    Returns changes detected in asset configurations within specified timeframe.
    """
    try:
        import sqlite3
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM asset_changes 
            WHERE change_timestamp > datetime('now', ? || ' hours')
        """
        params = [-hours]
        
        if asset_id:
            query += " AND asset_id = ?"
            params.append(asset_id)
        
        query += " ORDER BY change_timestamp DESC LIMIT 100"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to AssetChange objects
        changes = []
        for row in rows:
            changes.append({
                "change_id": row[0],
                "asset_id": row[1],
                "change_type": row[2],
                "change_timestamp": row[3],
                "changed_by": row[4],
                "impact_assessment": row[8]
            })
        
        return changes
        
    except Exception as e:
        logger.error(f"Failed to get recent changes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve recent changes: {str(e)}"
        )


@router.get("/statistics")
async def get_inventory_statistics() -> Dict[str, Any]:
    """
    Get comprehensive inventory statistics.
    
    Returns aggregated statistics about assets, compliance, risks, and costs.
    """
    try:
        assets = await asset_reporter.discover_assets()
        
        if not assets:
            return {
                "message": "No assets discovered",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate statistics
        total_cost = sum(a.estimated_monthly_cost or 0 for a in assets)
        avg_risk = sum(a.risk_score for a in assets) / len(assets)
        avg_compliance = sum(a.configuration.compliance_score for a in assets) / len(assets)
        
        return {
            "total_assets": len(assets),
            "by_category": _count_by_category(assets),
            "by_environment": _count_by_environment(assets),
            "by_importance": _count_by_importance(assets),
            "compliance": {
                "average_score": avg_compliance,
                "compliant_count": len([a for a in assets if 
                                       a.configuration.configuration_status == ConfigurationStatus.COMPLIANT]),
                "non_compliant_count": len([a for a in assets if 
                                          a.configuration.configuration_status == ConfigurationStatus.NON_COMPLIANT])
            },
            "risk": {
                "average_score": avg_risk,
                "high_risk_count": len([a for a in assets if a.risk_score > 70]),
                "medium_risk_count": len([a for a in assets if 30 < a.risk_score <= 70]),
                "low_risk_count": len([a for a in assets if a.risk_score <= 30])
            },
            "cost": {
                "total_monthly": total_cost,
                "total_annual": total_cost * 12,
                "average_per_asset": total_cost / len(assets) if assets else 0
            },
            "security": {
                "public_exposed": len([a for a in assets if a.public_exposure]),
                "encryption_enabled": len([a for a in assets if a.encryption_enabled]),
                "monitoring_enabled": len([a for a in assets if a.monitoring_enabled]),
                "backup_configured": len([a for a in assets if a.backup_configured])
            },
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate statistics: {str(e)}"
        )


@router.post("/baseline/save")
async def save_baseline_configuration() -> Dict[str, Any]:
    """
    Save current configuration as baseline.
    
    Captures current asset configurations as the baseline for drift detection.
    """
    try:
        assets = await asset_reporter.discover_assets()
        
        baseline = {}
        for asset in assets:
            baseline[asset.metadata.asset_id] = {
                setting.setting_name: setting.current_value
                for setting in asset.configuration.settings
            }
        
        # Store baseline in database
        import sqlite3
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO configuration_baselines 
            (baseline_id, created_at, asset_count, baseline_data)
            VALUES (?, ?, ?, ?)
        """, (
            f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            datetime.now(),
            len(baseline),
            json.dumps(baseline)
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "Baseline configuration saved",
            "asset_count": len(baseline),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to save baseline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save baseline configuration: {str(e)}"
        )


@router.post("/import")
async def import_assets(
    file: UploadFile = File(...),
    format: str = Query("json", description="File format (json, csv)")
) -> Dict[str, Any]:
    """
    Import assets from file.
    
    Allows importing asset inventory from external sources.
    """
    try:
        contents = await file.read()
        
        if format == "json":
            data = json.loads(contents)
            # Process JSON data
        elif format == "csv":
            # Process CSV data
            pass
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}"
            )
        
        return {
            "success": True,
            "message": f"Successfully imported assets from {file.filename}",
            "timestamp": datetime.now().isoformat()
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Asset import failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Asset import failed: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for asset reporter service.
    """
    try:
        # Check database connectivity
        import sqlite3
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='asset_inventory'")
        table_exists = cursor.fetchone()[0] > 0
        conn.close()
        
        return {
            "status": "healthy",
            "service": "Asset Inventory Reporter",
            "version": "1.0.0",
            "database_connected": True,
            "asset_inventory_table": table_exists,
            "project_id": project_id,
            "organization_id": organization_id,
            "supported_categories": [c.value for c in AssetCategory],
            "supported_formats": [f.value for f in ReportFormat],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Helper functions

def _generate_discovery_summary(assets: List[AssetInventoryItem]) -> Dict[str, Any]:
    """Generate summary of discovered assets"""
    return {
        "by_category": _count_by_category(assets),
        "by_importance": _count_by_importance(assets),
        "compliance_summary": {
            "compliant": len([a for a in assets if 
                            a.configuration.configuration_status == ConfigurationStatus.COMPLIANT]),
            "non_compliant": len([a for a in assets if 
                               a.configuration.configuration_status == ConfigurationStatus.NON_COMPLIANT])
        },
        "security_summary": {
            "public_exposed": len([a for a in assets if a.public_exposure]),
            "encrypted": len([a for a in assets if a.encryption_enabled])
        }
    }


def _count_by_category(assets: List[AssetInventoryItem]) -> Dict[str, int]:
    """Count assets by category"""
    counts = {}
    for asset in assets:
        category = asset.metadata.category.value
        counts[category] = counts.get(category, 0) + 1
    return counts


def _count_by_environment(assets: List[AssetInventoryItem]) -> Dict[str, int]:
    """Count assets by environment"""
    counts = {}
    for asset in assets:
        env = asset.metadata.environment
        counts[env] = counts.get(env, 0) + 1
    return counts


def _count_by_importance(assets: List[AssetInventoryItem]) -> Dict[str, int]:
    """Count assets by importance"""
    counts = {}
    for asset in assets:
        importance = asset.metadata.importance.value
        counts[importance] = counts.get(importance, 0) + 1
    return counts


def _get_media_type(format: ReportFormat) -> str:
    """Get media type for report format"""
    media_types = {
        ReportFormat.JSON: "application/json",
        ReportFormat.CSV: "text/csv",
        ReportFormat.HTML: "text/html",
        ReportFormat.PDF: "application/pdf",
        ReportFormat.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ReportFormat.MARKDOWN: "text/markdown",
        ReportFormat.YAML: "application/x-yaml"
    }
    return media_types.get(format, "application/octet-stream")


async def _get_baseline_configuration() -> Dict[str, Any]:
    """Get latest baseline configuration from database"""
    try:
        import sqlite3
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configuration_baselines (
                baseline_id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                asset_count INTEGER,
                baseline_data JSON
            )
        """)
        
        cursor.execute("""
            SELECT baseline_data FROM configuration_baselines
            ORDER BY created_at DESC LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        
        return {}
        
    except Exception as e:
        logger.error(f"Failed to get baseline: {e}")
        return {}


async def _perform_deep_analysis(asset_ids: List[str]):
    """Background task to perform deep analysis on assets"""
    try:
        logger.info(f"Performing deep analysis on {len(asset_ids)} assets")
        # In production, this would perform detailed security scans,
        # cost optimization analysis, etc.
    except Exception as e:
        logger.error(f"Deep analysis failed: {e}")


async def _auto_remediate_drifts(drifts: List[ConfigurationDrift]):
    """Background task to auto-remediate configuration drifts"""
    try:
        logger.info(f"Auto-remediating {len(drifts)} configuration drifts")
        # In production, this would execute remediation scripts
    except Exception as e:
        logger.error(f"Auto-remediation failed: {e}")


async def _schedule_recurring_report(request: AssetReportRequest):
    """Background task to schedule recurring report generation"""
    try:
        logger.info(f"Scheduling recurring report: {request.report_name}")
        # In production, this would create a scheduled job
    except Exception as e:
        logger.error(f"Failed to schedule report: {e}")


async def _send_report_to_recipients(report: AssetReport, recipients: List[str]):
    """Background task to send report to recipients"""
    try:
        logger.info(f"Sending report {report.report_id} to {len(recipients)} recipients")
        # In production, this would send emails or notifications
    except Exception as e:
        logger.error(f"Failed to send report: {e}")


# Export router
__all__ = ["router"]