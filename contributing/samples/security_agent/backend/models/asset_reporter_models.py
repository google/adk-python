"""
Asset Inventory & Setting Reporter Models
=========================================

Data models for comprehensive asset inventory, configuration reporting,
and settings analysis across all GCP services.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class AssetCategory(str, Enum):
    """Categories of GCP assets"""
    COMPUTE = "COMPUTE"
    STORAGE = "STORAGE"
    NETWORKING = "NETWORKING"
    DATABASE = "DATABASE"
    ANALYTICS = "ANALYTICS"
    AI_ML = "AI_ML"
    SECURITY = "SECURITY"
    IDENTITY = "IDENTITY"
    SERVERLESS = "SERVERLESS"
    DEVELOPER_TOOLS = "DEVELOPER_TOOLS"
    MANAGEMENT = "MANAGEMENT"
    IOT = "IOT"
    HYBRID = "HYBRID"
    OTHER = "OTHER"


class AssetImportance(str, Enum):
    """Asset importance levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


class ConfigurationStatus(str, Enum):
    """Configuration compliance status"""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    UNKNOWN = "UNKNOWN"
    EXEMPT = "EXEMPT"


class SettingType(str, Enum):
    """Types of configuration settings"""
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    COST = "COST"
    COMPLIANCE = "COMPLIANCE"
    OPERATIONAL = "OPERATIONAL"
    NETWORKING = "NETWORKING"
    ACCESS_CONTROL = "ACCESS_CONTROL"
    MONITORING = "MONITORING"
    BACKUP = "BACKUP"
    ENCRYPTION = "ENCRYPTION"


class ReportFormat(str, Enum):
    """Output formats for reports"""
    JSON = "JSON"
    CSV = "CSV"
    HTML = "HTML"
    PDF = "PDF"
    EXCEL = "EXCEL"
    MARKDOWN = "MARKDOWN"
    TERRAFORM = "TERRAFORM"
    YAML = "YAML"


class AssetMetadata(BaseModel):
    """Metadata for discovered assets"""
    asset_id: str = Field(default_factory=lambda: f"asset_{uuid.uuid4().hex[:8]}")
    asset_type: str = Field(..., description="Full asset type (e.g., compute.googleapis.com/Instance)")
    asset_name: str = Field(..., description="Asset resource name")
    display_name: str = Field(..., description="User-friendly display name")
    category: AssetCategory = Field(..., description="Asset category")
    project_id: str = Field(..., description="Project containing the asset")
    location: Optional[str] = Field(None, description="Asset location/region")
    created_time: Optional[datetime] = Field(None, description="Asset creation time")
    update_time: Optional[datetime] = Field(None, description="Last update time")
    labels: Dict[str, str] = Field(default_factory=dict, description="Asset labels")
    tags: List[str] = Field(default_factory=list, description="Asset tags")
    importance: AssetImportance = Field(AssetImportance.MEDIUM, description="Business importance")
    owner: Optional[str] = Field(None, description="Asset owner/team")
    cost_center: Optional[str] = Field(None, description="Cost center for billing")
    environment: Optional[str] = Field(None, description="Environment (prod/staging/dev)")
    dependencies: List[str] = Field(default_factory=list, description="Asset dependencies")
    

class ConfigurationSetting(BaseModel):
    """Individual configuration setting"""
    setting_id: str = Field(default_factory=lambda: f"setting_{uuid.uuid4().hex[:8]}")
    setting_name: str = Field(..., description="Setting name")
    setting_type: SettingType = Field(..., description="Type of setting")
    current_value: Any = Field(..., description="Current setting value")
    recommended_value: Optional[Any] = Field(None, description="Recommended value")
    default_value: Optional[Any] = Field(None, description="Default value")
    is_compliant: bool = Field(..., description="Compliance status")
    compliance_reason: Optional[str] = Field(None, description="Compliance reasoning")
    risk_level: Optional[str] = Field(None, description="Risk if non-compliant")
    remediation_steps: List[str] = Field(default_factory=list, description="Steps to fix")
    documentation_link: Optional[str] = Field(None, description="Documentation URL")
    last_changed: Optional[datetime] = Field(None, description="Last change timestamp")
    changed_by: Optional[str] = Field(None, description="Who changed the setting")


class AssetConfiguration(BaseModel):
    """Complete configuration for an asset"""
    asset_id: str = Field(..., description="Asset identifier")
    configuration_status: ConfigurationStatus = Field(..., description="Overall compliance status")
    compliance_score: float = Field(..., description="Compliance score (0-100)")
    settings: List[ConfigurationSetting] = Field(default_factory=list, description="Configuration settings")
    security_findings: List[Dict[str, Any]] = Field(default_factory=list, description="Security issues")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    cost_metrics: Dict[str, float] = Field(default_factory=dict, description="Cost metrics")
    recommendations: List[str] = Field(default_factory=list, description="Configuration recommendations")
    last_scanned: datetime = Field(default_factory=datetime.now, description="Last scan time")
    scan_errors: List[str] = Field(default_factory=list, description="Errors during scan")


class AssetInventoryItem(BaseModel):
    """Single item in asset inventory"""
    metadata: AssetMetadata = Field(..., description="Asset metadata")
    configuration: AssetConfiguration = Field(..., description="Asset configuration")
    relationships: Dict[str, List[str]] = Field(default_factory=dict, description="Asset relationships")
    access_controls: Dict[str, Any] = Field(default_factory=dict, description="IAM and access settings")
    monitoring_enabled: bool = Field(False, description="Monitoring status")
    backup_configured: bool = Field(False, description="Backup status")
    encryption_enabled: bool = Field(False, description="Encryption status")
    public_exposure: bool = Field(False, description="Public exposure status")
    data_residency: Optional[str] = Field(None, description="Data residency requirements")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Applicable frameworks")
    risk_score: float = Field(0.0, description="Risk score (0-100)")
    estimated_monthly_cost: Optional[float] = Field(None, description="Estimated monthly cost")


class InventoryFilter(BaseModel):
    """Filters for asset inventory queries"""
    categories: Optional[List[AssetCategory]] = Field(None, description="Filter by categories")
    projects: Optional[List[str]] = Field(None, description="Filter by projects")
    locations: Optional[List[str]] = Field(None, description="Filter by locations")
    asset_types: Optional[List[str]] = Field(None, description="Filter by asset types")
    importance_levels: Optional[List[AssetImportance]] = Field(None, description="Filter by importance")
    compliance_status: Optional[List[ConfigurationStatus]] = Field(None, description="Filter by compliance")
    environments: Optional[List[str]] = Field(None, description="Filter by environments")
    labels: Optional[Dict[str, str]] = Field(None, description="Filter by labels")
    min_risk_score: Optional[float] = Field(None, description="Minimum risk score")
    max_risk_score: Optional[float] = Field(None, description="Maximum risk score")
    public_only: Optional[bool] = Field(None, description="Only public assets")
    created_after: Optional[datetime] = Field(None, description="Created after date")
    created_before: Optional[datetime] = Field(None, description="Created before date")


class AssetGrouping(BaseModel):
    """Grouping configuration for reports"""
    group_by: List[str] = Field(..., description="Fields to group by")
    aggregations: Dict[str, str] = Field(default_factory=dict, description="Aggregation functions")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: str = Field("DESC", description="Sort order (ASC/DESC)")
    include_subtotals: bool = Field(True, description="Include subtotals")
    include_percentages: bool = Field(True, description="Include percentages")


class ConfigurationDrift(BaseModel):
    """Configuration drift detection"""
    drift_id: str = Field(default_factory=lambda: f"drift_{uuid.uuid4().hex[:8]}")
    asset_id: str = Field(..., description="Asset with drift")
    setting_name: str = Field(..., description="Drifted setting")
    expected_value: Any = Field(..., description="Expected value")
    actual_value: Any = Field(..., description="Actual value")
    drift_detected_at: datetime = Field(default_factory=datetime.now, description="Detection time")
    drift_severity: str = Field(..., description="Severity of drift")
    auto_remediation_available: bool = Field(False, description="Can auto-fix")
    remediation_script: Optional[str] = Field(None, description="Fix script")
    business_impact: Optional[str] = Field(None, description="Business impact")


class AssetReport(BaseModel):
    """Generated asset inventory report"""
    report_id: str = Field(default_factory=lambda: f"report_{uuid.uuid4().hex[:8]}")
    report_name: str = Field(..., description="Report name")
    report_type: str = Field(..., description="Type of report")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation time")
    generated_by: Optional[str] = Field(None, description="Who generated report")
    filters_applied: Optional[InventoryFilter] = Field(None, description="Applied filters")
    grouping_config: Optional[AssetGrouping] = Field(None, description="Grouping configuration")
    total_assets: int = Field(..., description="Total assets in report")
    asset_summary: Dict[str, int] = Field(default_factory=dict, description="Summary by category")
    compliance_summary: Dict[str, int] = Field(default_factory=dict, description="Compliance summary")
    risk_summary: Dict[str, float] = Field(default_factory=dict, description="Risk summary")
    cost_summary: Dict[str, float] = Field(default_factory=dict, description="Cost summary")
    critical_findings: List[Dict[str, Any]] = Field(default_factory=list, description="Critical issues")
    recommendations: List[str] = Field(default_factory=list, description="Report recommendations")
    export_formats: List[ReportFormat] = Field(default_factory=list, description="Available formats")
    report_data: Optional[Any] = Field(None, description="Actual report data")


class AssetChange(BaseModel):
    """Asset change tracking"""
    change_id: str = Field(default_factory=lambda: f"change_{uuid.uuid4().hex[:8]}")
    asset_id: str = Field(..., description="Changed asset")
    change_type: str = Field(..., description="Type of change")
    change_timestamp: datetime = Field(default_factory=datetime.now, description="Change time")
    changed_by: Optional[str] = Field(None, description="Who made change")
    old_value: Optional[Any] = Field(None, description="Previous value")
    new_value: Optional[Any] = Field(None, description="New value")
    change_reason: Optional[str] = Field(None, description="Reason for change")
    approved_by: Optional[str] = Field(None, description="Approver")
    rollback_available: bool = Field(False, description="Can rollback")
    impact_assessment: Optional[str] = Field(None, description="Impact of change")


class ComplianceRule(BaseModel):
    """Compliance rule definition"""
    rule_id: str = Field(default_factory=lambda: f"rule_{uuid.uuid4().hex[:8]}")
    rule_name: str = Field(..., description="Rule name")
    framework: str = Field(..., description="Compliance framework")
    description: str = Field(..., description="Rule description")
    asset_types: List[str] = Field(..., description="Applicable asset types")
    condition: str = Field(..., description="Rule condition")
    severity: str = Field(..., description="Violation severity")
    remediation_guidance: str = Field(..., description="How to fix")
    automation_available: bool = Field(False, description="Can auto-remediate")
    exceptions: List[str] = Field(default_factory=list, description="Exception list")


class AssetReportRequest(BaseModel):
    """Request to generate asset report"""
    report_name: str = Field(..., description="Report name")
    report_type: str = Field("INVENTORY", description="Type of report")
    filters: Optional[InventoryFilter] = Field(None, description="Filters to apply")
    grouping: Optional[AssetGrouping] = Field(None, description="Grouping configuration")
    include_configurations: bool = Field(True, description="Include configs")
    include_compliance: bool = Field(True, description="Include compliance")
    include_costs: bool = Field(True, description="Include cost data")
    include_relationships: bool = Field(False, description="Include relationships")
    include_changes: bool = Field(False, description="Include recent changes")
    export_format: ReportFormat = Field(ReportFormat.JSON, description="Export format")
    schedule: Optional[str] = Field(None, description="Schedule for recurring")
    recipients: List[str] = Field(default_factory=list, description="Report recipients")


class AssetReportResponse(BaseModel):
    """Response from asset report generation"""
    report: AssetReport = Field(..., description="Generated report")
    assets: List[AssetInventoryItem] = Field(..., description="Assets in report")
    configuration_drifts: List[ConfigurationDrift] = Field(default_factory=list, description="Detected drifts")
    recent_changes: List[AssetChange] = Field(default_factory=list, description="Recent changes")
    compliance_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Violations")
    export_urls: Dict[str, str] = Field(default_factory=dict, description="Download URLs")
    next_scheduled_run: Optional[datetime] = Field(None, description="Next run time")
    processing_time_ms: int = Field(..., description="Generation time")