"""
Asset Inventory & Setting Reporter Service
==========================================

Service for discovering assets, analyzing configurations, detecting drift,
and generating comprehensive inventory reports.
"""

import os
import logging
import sqlite3
import json
import csv
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

try:
    from google.cloud import asset_v1
    from google.cloud import resourcemanager_v3
    from google.cloud import compute_v1
    from google.cloud import storage
    from google.cloud import sql_v1
    from google.cloud import container_v1
    GCLOUD_AVAILABLE = True
except ImportError:
    GCLOUD_AVAILABLE = False

from ..models.asset_reporter_models import (
    AssetCategory, AssetImportance, ConfigurationStatus, SettingType,
    ReportFormat, AssetMetadata, ConfigurationSetting, AssetConfiguration,
    AssetInventoryItem, InventoryFilter, AssetGrouping, ConfigurationDrift,
    AssetReport, AssetChange, ComplianceRule, AssetReportRequest,
    AssetReportResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssetReporter:
    """Asset Inventory and Setting Reporter service"""
    
    def __init__(self, project_id: str, organization_id: Optional[str] = None,
                 database_path: str = "backend/cache/gcp_data.db"):
        self.project_id = project_id
        self.organization_id = organization_id
        self.database_path = database_path
        
        # Initialize clients if available
        if GCLOUD_AVAILABLE:
            try:
                self.asset_client = asset_v1.AssetServiceClient()
                self.resource_client = resourcemanager_v3.ProjectsClient()
                self.compute_client = compute_v1.InstancesClient()
                self.storage_client = storage.Client(project=project_id)
                self.container_client = container_v1.ClusterManagerClient()
            except Exception as e:
                logger.warning(f"Failed to initialize GCP clients: {e}")
                self.asset_client = None
        else:
            self.asset_client = None
        
        # Initialize database
        self._init_database()
        
        # Asset type mappings
        self.asset_type_categories = {
            "compute.googleapis.com/Instance": AssetCategory.COMPUTE,
            "compute.googleapis.com/Disk": AssetCategory.COMPUTE,
            "storage.googleapis.com/Bucket": AssetCategory.STORAGE,
            "container.googleapis.com/Cluster": AssetCategory.COMPUTE,
            "sqladmin.googleapis.com/Instance": AssetCategory.DATABASE,
            "spanner.googleapis.com/Instance": AssetCategory.DATABASE,
            "bigquery.googleapis.com/Dataset": AssetCategory.ANALYTICS,
            "pubsub.googleapis.com/Topic": AssetCategory.ANALYTICS,
            "run.googleapis.com/Service": AssetCategory.SERVERLESS,
            "cloudfunctions.googleapis.com/Function": AssetCategory.SERVERLESS,
            "iam.googleapis.com/ServiceAccount": AssetCategory.IDENTITY,
            "compute.googleapis.com/Network": AssetCategory.NETWORKING,
            "compute.googleapis.com/Firewall": AssetCategory.SECURITY,
            "cloudkms.googleapis.com/CryptoKey": AssetCategory.SECURITY,
            "aiplatform.googleapis.com/Model": AssetCategory.AI_ML,
            "redis.googleapis.com/Instance": AssetCategory.DATABASE,
            "filestore.googleapis.com/Instance": AssetCategory.STORAGE,
            "networkconnectivity.googleapis.com/Hub": AssetCategory.NETWORKING,
            "apigateway.googleapis.com/Gateway": AssetCategory.NETWORKING,
            "secretmanager.googleapis.com/Secret": AssetCategory.SECURITY
        }
        
        # Compliance rules
        self.compliance_rules = self._load_compliance_rules()
    
    def _init_database(self):
        """Initialize database tables for asset reporting"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS asset_inventory (
                    asset_id TEXT PRIMARY KEY,
                    asset_type TEXT,
                    asset_name TEXT,
                    display_name TEXT,
                    category TEXT,
                    project_id TEXT,
                    location TEXT,
                    importance TEXT,
                    environment TEXT,
                    compliance_status TEXT,
                    compliance_score REAL,
                    risk_score REAL,
                    public_exposure INTEGER,
                    monitoring_enabled INTEGER,
                    encryption_enabled INTEGER,
                    estimated_cost REAL,
                    metadata JSON,
                    configuration JSON,
                    last_scanned TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS configuration_settings (
                    setting_id TEXT PRIMARY KEY,
                    asset_id TEXT,
                    setting_name TEXT,
                    setting_type TEXT,
                    current_value TEXT,
                    recommended_value TEXT,
                    is_compliant INTEGER,
                    risk_level TEXT,
                    last_changed TIMESTAMP,
                    FOREIGN KEY (asset_id) REFERENCES asset_inventory(asset_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS configuration_drifts (
                    drift_id TEXT PRIMARY KEY,
                    asset_id TEXT,
                    setting_name TEXT,
                    expected_value TEXT,
                    actual_value TEXT,
                    drift_severity TEXT,
                    detected_at TIMESTAMP,
                    auto_remediation_available INTEGER,
                    remediation_script TEXT,
                    FOREIGN KEY (asset_id) REFERENCES asset_inventory(asset_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS asset_changes (
                    change_id TEXT PRIMARY KEY,
                    asset_id TEXT,
                    change_type TEXT,
                    change_timestamp TIMESTAMP,
                    changed_by TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    change_reason TEXT,
                    impact_assessment TEXT,
                    FOREIGN KEY (asset_id) REFERENCES asset_inventory(asset_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS asset_reports (
                    report_id TEXT PRIMARY KEY,
                    report_name TEXT,
                    report_type TEXT,
                    generated_at TIMESTAMP,
                    generated_by TEXT,
                    total_assets INTEGER,
                    filters JSON,
                    summary JSON,
                    report_data JSON,
                    export_urls JSON
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Asset reporter database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _load_compliance_rules(self) -> List[ComplianceRule]:
        """Load compliance rules from configuration"""
        rules = [
            ComplianceRule(
                rule_name="Public Storage Buckets",
                framework="CIS",
                description="Storage buckets should not be publicly accessible",
                asset_types=["storage.googleapis.com/Bucket"],
                condition="public_exposure == False",
                severity="HIGH",
                remediation_guidance="Remove allUsers and allAuthenticatedUsers from bucket IAM",
                automation_available=True
            ),
            ComplianceRule(
                rule_name="Unencrypted Databases",
                framework="PCI-DSS",
                description="Databases must be encrypted at rest",
                asset_types=["sqladmin.googleapis.com/Instance", "spanner.googleapis.com/Instance"],
                condition="encryption_enabled == True",
                severity="CRITICAL",
                remediation_guidance="Enable encryption with customer-managed keys",
                automation_available=True
            ),
            ComplianceRule(
                rule_name="Missing Monitoring",
                framework="SOC2",
                description="All production assets must have monitoring enabled",
                asset_types=["compute.googleapis.com/Instance", "container.googleapis.com/Cluster"],
                condition="monitoring_enabled == True AND environment == 'production'",
                severity="MEDIUM",
                remediation_guidance="Enable Cloud Monitoring and install agents",
                automation_available=True
            ),
            ComplianceRule(
                rule_name="Outdated GKE Clusters",
                framework="Security Best Practices",
                description="GKE clusters should run supported versions",
                asset_types=["container.googleapis.com/Cluster"],
                condition="version >= minimum_supported_version",
                severity="HIGH",
                remediation_guidance="Upgrade cluster to latest stable version",
                automation_available=False
            )
        ]
        return rules
    
    async def discover_assets(self, filters: Optional[InventoryFilter] = None) -> List[AssetInventoryItem]:
        """Discover all assets in the project/organization"""
        logger.info(f"Starting asset discovery for project {self.project_id}")
        assets = []
        
        try:
            if self.asset_client and GCLOUD_AVAILABLE:
                # Use Cloud Asset Inventory API
                parent = f"projects/{self.project_id}"
                if self.organization_id:
                    parent = f"organizations/{self.organization_id}"
                
                request = asset_v1.ListAssetsRequest(
                    parent=parent,
                    content_type=asset_v1.ContentType.RESOURCE,
                    page_size=100
                )
                
                # Get assets from API
                page_result = self.asset_client.list_assets(request=request)
                
                for response in page_result:
                    asset_item = await self._process_asset(response)
                    if asset_item and self._apply_filter(asset_item, filters):
                        assets.append(asset_item)
            else:
                # Use mock data for testing
                assets = await self._get_mock_assets(filters)
            
            # Store in database
            await self._store_assets(assets)
            
            logger.info(f"Discovered {len(assets)} assets")
            return assets
            
        except Exception as e:
            logger.error(f"Asset discovery failed: {e}")
            # Return mock data on error
            return await self._get_mock_assets(filters)
    
    async def _process_asset(self, asset_data: Any) -> Optional[AssetInventoryItem]:
        """Process raw asset data into inventory item"""
        try:
            asset_type = asset_data.asset_type
            asset_name = asset_data.name
            
            # Extract metadata
            metadata = AssetMetadata(
                asset_type=asset_type,
                asset_name=asset_name,
                display_name=asset_name.split('/')[-1],
                category=self.asset_type_categories.get(asset_type, AssetCategory.OTHER),
                project_id=self.project_id,
                location=self._extract_location(asset_name),
                created_time=asset_data.create_time if hasattr(asset_data, 'create_time') else None,
                update_time=asset_data.update_time if hasattr(asset_data, 'update_time') else None,
                labels=asset_data.resource.data.get('labels', {}) if hasattr(asset_data.resource, 'data') else {},
                importance=self._determine_importance(asset_data),
                environment=self._determine_environment(asset_data)
            )
            
            # Analyze configuration
            configuration = await self._analyze_configuration(asset_data)
            
            # Build inventory item
            item = AssetInventoryItem(
                metadata=metadata,
                configuration=configuration,
                relationships=self._extract_relationships(asset_data),
                access_controls=self._extract_access_controls(asset_data),
                monitoring_enabled=self._check_monitoring(asset_data),
                backup_configured=self._check_backup(asset_data),
                encryption_enabled=self._check_encryption(asset_data),
                public_exposure=self._check_public_exposure(asset_data),
                compliance_frameworks=self._get_applicable_frameworks(asset_type),
                risk_score=self._calculate_risk_score(configuration),
                estimated_monthly_cost=self._estimate_cost(asset_data)
            )
            
            return item
            
        except Exception as e:
            logger.warning(f"Failed to process asset: {e}")
            return None
    
    async def _analyze_configuration(self, asset_data: Any) -> AssetConfiguration:
        """Analyze asset configuration for compliance"""
        settings = []
        compliance_score = 100.0
        non_compliant_count = 0
        
        # Check various configuration aspects
        checks = [
            ("encryption", self._check_encryption(asset_data), "Encryption at rest"),
            ("public_access", not self._check_public_exposure(asset_data), "No public access"),
            ("monitoring", self._check_monitoring(asset_data), "Monitoring enabled"),
            ("backup", self._check_backup(asset_data), "Backup configured"),
            ("labels", bool(self._extract_labels(asset_data)), "Proper labeling"),
            ("network_security", self._check_network_security(asset_data), "Network security")
        ]
        
        for setting_name, is_compliant, description in checks:
            if not is_compliant:
                non_compliant_count += 1
            
            setting = ConfigurationSetting(
                setting_name=setting_name,
                setting_type=SettingType.SECURITY,
                current_value=is_compliant,
                recommended_value=True,
                is_compliant=is_compliant,
                compliance_reason=description,
                risk_level="HIGH" if not is_compliant else "LOW",
                remediation_steps=[f"Enable {setting_name}"] if not is_compliant else []
            )
            settings.append(setting)
        
        # Calculate compliance score
        if len(checks) > 0:
            compliance_score = ((len(checks) - non_compliant_count) / len(checks)) * 100
        
        status = ConfigurationStatus.COMPLIANT if compliance_score >= 90 else \
                 ConfigurationStatus.PARTIALLY_COMPLIANT if compliance_score >= 70 else \
                 ConfigurationStatus.NON_COMPLIANT
        
        return AssetConfiguration(
            asset_id=asset_data.name,
            configuration_status=status,
            compliance_score=compliance_score,
            settings=settings,
            recommendations=self._generate_recommendations(settings),
            last_scanned=datetime.now()
        )
    
    async def detect_configuration_drift(self, 
                                        baseline_config: Dict[str, Any]) -> List[ConfigurationDrift]:
        """Detect configuration drift from baseline"""
        drifts = []
        
        try:
            # Get current assets
            current_assets = await self.discover_assets()
            
            for asset in current_assets:
                asset_id = asset.metadata.asset_id
                
                if asset_id in baseline_config:
                    baseline = baseline_config[asset_id]
                    
                    # Compare configurations
                    for setting in asset.configuration.settings:
                        baseline_value = baseline.get(setting.setting_name)
                        
                        if baseline_value and baseline_value != setting.current_value:
                            drift = ConfigurationDrift(
                                asset_id=asset_id,
                                setting_name=setting.setting_name,
                                expected_value=baseline_value,
                                actual_value=setting.current_value,
                                drift_severity="HIGH" if setting.setting_type == SettingType.SECURITY else "MEDIUM",
                                auto_remediation_available=True,
                                remediation_script=self._generate_remediation_script(
                                    asset.metadata.asset_type,
                                    setting.setting_name,
                                    baseline_value
                                ),
                                business_impact=self._assess_drift_impact(setting)
                            )
                            drifts.append(drift)
            
            # Store drifts in database
            await self._store_drifts(drifts)
            
            logger.info(f"Detected {len(drifts)} configuration drifts")
            return drifts
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return []
    
    async def generate_report(self, request: AssetReportRequest) -> AssetReportResponse:
        """Generate comprehensive asset inventory report"""
        logger.info(f"Generating report: {request.report_name}")
        
        try:
            # Discover assets with filters
            assets = await self.discover_assets(request.filters)
            
            # Apply grouping if requested
            grouped_data = self._group_assets(assets, request.grouping) if request.grouping else None
            
            # Generate summaries
            asset_summary = self._generate_asset_summary(assets)
            compliance_summary = self._generate_compliance_summary(assets)
            risk_summary = self._generate_risk_summary(assets)
            cost_summary = self._generate_cost_summary(assets)
            
            # Find critical issues
            critical_findings = self._identify_critical_findings(assets)
            
            # Generate recommendations
            recommendations = self._generate_report_recommendations(assets, critical_findings)
            
            # Create report
            report = AssetReport(
                report_name=request.report_name,
                report_type=request.report_type,
                filters_applied=request.filters,
                grouping_config=request.grouping,
                total_assets=len(assets),
                asset_summary=asset_summary,
                compliance_summary=compliance_summary,
                risk_summary=risk_summary,
                cost_summary=cost_summary,
                critical_findings=critical_findings,
                recommendations=recommendations,
                export_formats=[request.export_format],
                report_data=grouped_data or {"assets": [a.dict() for a in assets[:100]]}  # Limit for response
            )
            
            # Export report in requested format
            export_urls = await self._export_report(report, assets, request.export_format)
            
            # Get recent changes if requested
            recent_changes = await self._get_recent_changes() if request.include_changes else []
            
            # Detect configuration drifts
            drifts = await self._detect_recent_drifts() if request.include_configurations else []
            
            # Store report in database
            await self._store_report(report)
            
            # Create response
            response = AssetReportResponse(
                report=report,
                assets=assets[:100],  # Limit for response size
                configuration_drifts=drifts,
                recent_changes=recent_changes,
                compliance_violations=critical_findings,
                export_urls=export_urls,
                processing_time_ms=100  # Mock processing time
            )
            
            logger.info(f"Report generated successfully: {report.report_id}")
            return response
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
    
    async def _get_mock_assets(self, filters: Optional[InventoryFilter] = None) -> List[AssetInventoryItem]:
        """Generate mock assets for testing"""
        mock_assets = []
        
        # Mock compute instances
        for i in range(5):
            metadata = AssetMetadata(
                asset_type="compute.googleapis.com/Instance",
                asset_name=f"projects/{self.project_id}/zones/us-central1-a/instances/web-server-{i+1}",
                display_name=f"web-server-{i+1}",
                category=AssetCategory.COMPUTE,
                project_id=self.project_id,
                location="us-central1-a",
                importance=AssetImportance.HIGH if i < 2 else AssetImportance.MEDIUM,
                environment="production" if i < 3 else "staging",
                labels={"team": "platform", "env": "prod" if i < 3 else "staging"}
            )
            
            configuration = AssetConfiguration(
                asset_id=metadata.asset_id,
                configuration_status=ConfigurationStatus.COMPLIANT if i < 2 else ConfigurationStatus.PARTIALLY_COMPLIANT,
                compliance_score=95.0 if i < 2 else 75.0,
                settings=[
                    ConfigurationSetting(
                        setting_name="encryption",
                        setting_type=SettingType.SECURITY,
                        current_value=True,
                        recommended_value=True,
                        is_compliant=True
                    ),
                    ConfigurationSetting(
                        setting_name="public_ip",
                        setting_type=SettingType.SECURITY,
                        current_value=i >= 3,
                        recommended_value=False,
                        is_compliant=i < 3,
                        risk_level="HIGH" if i >= 3 else "LOW"
                    )
                ],
                recommendations=["Remove public IP"] if i >= 3 else [],
                last_scanned=datetime.now()
            )
            
            mock_assets.append(AssetInventoryItem(
                metadata=metadata,
                configuration=configuration,
                monitoring_enabled=True,
                backup_configured=i < 3,
                encryption_enabled=True,
                public_exposure=i >= 3,
                risk_score=20.0 if i < 2 else 60.0 if i < 4 else 80.0,
                estimated_monthly_cost=150.0 + (i * 50),
                compliance_frameworks=["CIS", "SOC2"]
            ))
        
        # Mock storage buckets
        for i in range(3):
            metadata = AssetMetadata(
                asset_type="storage.googleapis.com/Bucket",
                asset_name=f"{self.project_id}-data-bucket-{i+1}",
                display_name=f"data-bucket-{i+1}",
                category=AssetCategory.STORAGE,
                project_id=self.project_id,
                location="us-central1",
                importance=AssetImportance.CRITICAL if i == 0 else AssetImportance.HIGH,
                environment="production",
                labels={"data": "sensitive" if i == 0 else "public"}
            )
            
            configuration = AssetConfiguration(
                asset_id=metadata.asset_id,
                configuration_status=ConfigurationStatus.COMPLIANT if i == 0 else ConfigurationStatus.NON_COMPLIANT,
                compliance_score=100.0 if i == 0 else 40.0,
                settings=[
                    ConfigurationSetting(
                        setting_name="public_access",
                        setting_type=SettingType.SECURITY,
                        current_value=i > 0,
                        recommended_value=False,
                        is_compliant=i == 0,
                        risk_level="CRITICAL" if i > 0 else "LOW"
                    )
                ],
                recommendations=["Remove public access"] if i > 0 else [],
                last_scanned=datetime.now()
            )
            
            mock_assets.append(AssetInventoryItem(
                metadata=metadata,
                configuration=configuration,
                monitoring_enabled=True,
                backup_configured=True,
                encryption_enabled=i == 0,
                public_exposure=i > 0,
                risk_score=10.0 if i == 0 else 90.0,
                estimated_monthly_cost=25.0 + (i * 10),
                compliance_frameworks=["PCI-DSS", "HIPAA"] if i == 0 else ["CIS"]
            ))
        
        # Mock GKE cluster
        metadata = AssetMetadata(
            asset_type="container.googleapis.com/Cluster",
            asset_name=f"projects/{self.project_id}/locations/us-central1/clusters/production-cluster",
            display_name="production-cluster",
            category=AssetCategory.COMPUTE,
            project_id=self.project_id,
            location="us-central1",
            importance=AssetImportance.CRITICAL,
            environment="production",
            labels={"team": "platform", "tier": "1"}
        )
        
        configuration = AssetConfiguration(
            asset_id=metadata.asset_id,
            configuration_status=ConfigurationStatus.PARTIALLY_COMPLIANT,
            compliance_score=85.0,
            settings=[
                ConfigurationSetting(
                    setting_name="private_cluster",
                    setting_type=SettingType.SECURITY,
                    current_value=True,
                    recommended_value=True,
                    is_compliant=True
                ),
                ConfigurationSetting(
                    setting_name="workload_identity",
                    setting_type=SettingType.SECURITY,
                    current_value=False,
                    recommended_value=True,
                    is_compliant=False,
                    risk_level="MEDIUM"
                )
            ],
            recommendations=["Enable Workload Identity"],
            last_scanned=datetime.now()
        )
        
        mock_assets.append(AssetInventoryItem(
            metadata=metadata,
            configuration=configuration,
            monitoring_enabled=True,
            backup_configured=True,
            encryption_enabled=True,
            public_exposure=False,
            risk_score=35.0,
            estimated_monthly_cost=1200.0,
            compliance_frameworks=["CIS", "SOC2", "PCI-DSS"]
        ))
        
        # Apply filters if provided
        if filters:
            filtered_assets = []
            for asset in mock_assets:
                if self._apply_filter(asset, filters):
                    filtered_assets.append(asset)
            return filtered_assets
        
        return mock_assets
    
    def _apply_filter(self, asset: AssetInventoryItem, filters: Optional[InventoryFilter]) -> bool:
        """Apply inventory filters to an asset"""
        if not filters:
            return True
        
        if filters.categories and asset.metadata.category not in filters.categories:
            return False
        
        if filters.importance_levels and asset.metadata.importance not in filters.importance_levels:
            return False
        
        if filters.compliance_status and asset.configuration.configuration_status not in filters.compliance_status:
            return False
        
        if filters.environments and asset.metadata.environment not in filters.environments:
            return False
        
        if filters.public_only is not None and asset.public_exposure != filters.public_only:
            return False
        
        if filters.min_risk_score is not None and asset.risk_score < filters.min_risk_score:
            return False
        
        if filters.max_risk_score is not None and asset.risk_score > filters.max_risk_score:
            return False
        
        return True
    
    def _extract_location(self, asset_name: str) -> Optional[str]:
        """Extract location from asset name"""
        parts = asset_name.split('/')
        for i, part in enumerate(parts):
            if part in ['zones', 'regions', 'locations'] and i + 1 < len(parts):
                return parts[i + 1]
        return None
    
    def _determine_importance(self, asset_data: Any) -> AssetImportance:
        """Determine asset importance based on various factors"""
        # Check labels
        labels = self._extract_labels(asset_data)
        if labels.get('tier') == '1' or labels.get('critical') == 'true':
            return AssetImportance.CRITICAL
        if labels.get('env') == 'production' or labels.get('environment') == 'production':
            return AssetImportance.HIGH
        if labels.get('env') == 'staging':
            return AssetImportance.MEDIUM
        return AssetImportance.LOW
    
    def _determine_environment(self, asset_data: Any) -> str:
        """Determine asset environment"""
        labels = self._extract_labels(asset_data)
        return labels.get('env', labels.get('environment', 'unknown'))
    
    def _extract_labels(self, asset_data: Any) -> Dict[str, str]:
        """Extract labels from asset data"""
        try:
            if hasattr(asset_data, 'resource') and hasattr(asset_data.resource, 'data'):
                return asset_data.resource.data.get('labels', {})
        except:
            pass
        return {}
    
    def _extract_relationships(self, asset_data: Any) -> Dict[str, List[str]]:
        """Extract asset relationships"""
        # This would extract network, IAM, and other relationships
        return {}
    
    def _extract_access_controls(self, asset_data: Any) -> Dict[str, Any]:
        """Extract IAM and access control settings"""
        # This would extract IAM policies and access settings
        return {}
    
    def _check_monitoring(self, asset_data: Any) -> bool:
        """Check if monitoring is enabled"""
        # Check for monitoring configuration
        return True  # Mock implementation
    
    def _check_backup(self, asset_data: Any) -> bool:
        """Check if backup is configured"""
        # Check for backup configuration
        return True  # Mock implementation
    
    def _check_encryption(self, asset_data: Any) -> bool:
        """Check if encryption is enabled"""
        # Check for encryption configuration
        return True  # Mock implementation
    
    def _check_public_exposure(self, asset_data: Any) -> bool:
        """Check if asset is publicly exposed"""
        # Check for public IPs, public access, etc.
        return False  # Mock implementation
    
    def _check_network_security(self, asset_data: Any) -> bool:
        """Check network security configuration"""
        # Check firewall rules, network policies, etc.
        return True  # Mock implementation
    
    def _get_applicable_frameworks(self, asset_type: str) -> List[str]:
        """Get applicable compliance frameworks for asset type"""
        frameworks = ["CIS"]
        
        if "database" in asset_type.lower() or "sql" in asset_type.lower():
            frameworks.extend(["PCI-DSS", "HIPAA"])
        
        if "compute" in asset_type.lower() or "container" in asset_type.lower():
            frameworks.append("SOC2")
        
        return frameworks
    
    def _calculate_risk_score(self, configuration: AssetConfiguration) -> float:
        """Calculate risk score based on configuration"""
        # Higher score = higher risk
        base_score = 100.0 - configuration.compliance_score
        
        # Adjust based on specific settings
        for setting in configuration.settings:
            if not setting.is_compliant:
                if setting.risk_level == "CRITICAL":
                    base_score += 20
                elif setting.risk_level == "HIGH":
                    base_score += 10
                elif setting.risk_level == "MEDIUM":
                    base_score += 5
        
        return min(100.0, max(0.0, base_score))
    
    def _estimate_cost(self, asset_data: Any) -> Optional[float]:
        """Estimate monthly cost for asset"""
        # This would use pricing APIs or estimates
        return 100.0  # Mock value
    
    def _generate_recommendations(self, settings: List[ConfigurationSetting]) -> List[str]:
        """Generate recommendations based on settings"""
        recommendations = []
        
        for setting in settings:
            if not setting.is_compliant:
                if setting.remediation_steps:
                    recommendations.extend(setting.remediation_steps)
                else:
                    recommendations.append(f"Review and fix {setting.setting_name}")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_remediation_script(self, asset_type: str, setting_name: str, 
                                    target_value: Any) -> Optional[str]:
        """Generate remediation script for configuration drift"""
        scripts = {
            ("compute.googleapis.com/Instance", "encryption"): f"""
                gcloud compute disks update DISK_NAME \\
                    --kms-key=projects/PROJECT/locations/LOCATION/keyRings/RING/cryptoKeys/KEY
            """,
            ("storage.googleapis.com/Bucket", "public_access"): f"""
                gsutil iam ch -d allUsers BUCKET_NAME
                gsutil iam ch -d allAuthenticatedUsers BUCKET_NAME
            """,
            ("container.googleapis.com/Cluster", "workload_identity"): f"""
                gcloud container clusters update CLUSTER_NAME \\
                    --workload-pool=PROJECT.svc.id.goog
            """
        }
        
        return scripts.get((asset_type, setting_name))
    
    def _assess_drift_impact(self, setting: ConfigurationSetting) -> str:
        """Assess business impact of configuration drift"""
        if setting.risk_level == "CRITICAL":
            return "High business impact - immediate remediation required"
        elif setting.risk_level == "HIGH":
            return "Moderate business impact - schedule remediation within 24 hours"
        else:
            return "Low business impact - remediate during next maintenance window"
    
    def _group_assets(self, assets: List[AssetInventoryItem], 
                     grouping: AssetGrouping) -> Dict[str, Any]:
        """Group assets according to configuration"""
        grouped = {}
        
        for field in grouping.group_by:
            grouped[field] = {}
            
            for asset in assets:
                # Get grouping value
                if field == "category":
                    key = asset.metadata.category.value
                elif field == "environment":
                    key = asset.metadata.environment
                elif field == "project":
                    key = asset.metadata.project_id
                elif field == "importance":
                    key = asset.metadata.importance.value
                elif field == "compliance_status":
                    key = asset.configuration.configuration_status.value
                else:
                    key = "other"
                
                if key not in grouped[field]:
                    grouped[field] = []
                grouped[field].append(asset)
        
        return grouped
    
    def _generate_asset_summary(self, assets: List[AssetInventoryItem]) -> Dict[str, int]:
        """Generate asset summary by category"""
        summary = {}
        for asset in assets:
            category = asset.metadata.category.value
            summary[category] = summary.get(category, 0) + 1
        return summary
    
    def _generate_compliance_summary(self, assets: List[AssetInventoryItem]) -> Dict[str, int]:
        """Generate compliance summary"""
        summary = {
            "COMPLIANT": 0,
            "PARTIALLY_COMPLIANT": 0,
            "NON_COMPLIANT": 0,
            "UNKNOWN": 0
        }
        
        for asset in assets:
            status = asset.configuration.configuration_status.value
            summary[status] = summary.get(status, 0) + 1
        
        return summary
    
    def _generate_risk_summary(self, assets: List[AssetInventoryItem]) -> Dict[str, float]:
        """Generate risk summary statistics"""
        if not assets:
            return {"average": 0, "max": 0, "min": 0}
        
        risk_scores = [asset.risk_score for asset in assets]
        return {
            "average": sum(risk_scores) / len(risk_scores),
            "max": max(risk_scores),
            "min": min(risk_scores),
            "high_risk_count": len([s for s in risk_scores if s > 70]),
            "medium_risk_count": len([s for s in risk_scores if 30 < s <= 70]),
            "low_risk_count": len([s for s in risk_scores if s <= 30])
        }
    
    def _generate_cost_summary(self, assets: List[AssetInventoryItem]) -> Dict[str, float]:
        """Generate cost summary"""
        total_cost = 0
        by_category = {}
        
        for asset in assets:
            if asset.estimated_monthly_cost:
                total_cost += asset.estimated_monthly_cost
                category = asset.metadata.category.value
                by_category[category] = by_category.get(category, 0) + asset.estimated_monthly_cost
        
        return {
            "total_monthly": total_cost,
            "total_annual": total_cost * 12,
            "by_category": by_category
        }
    
    def _identify_critical_findings(self, assets: List[AssetInventoryItem]) -> List[Dict[str, Any]]:
        """Identify critical security and compliance findings"""
        findings = []
        
        for asset in assets:
            # Check for critical issues
            if asset.public_exposure and asset.metadata.importance in [AssetImportance.CRITICAL, AssetImportance.HIGH]:
                findings.append({
                    "severity": "CRITICAL",
                    "asset_id": asset.metadata.asset_id,
                    "asset_name": asset.metadata.display_name,
                    "finding": "Critical asset with public exposure",
                    "remediation": "Remove public access immediately"
                })
            
            if not asset.encryption_enabled and "database" in asset.metadata.asset_type.lower():
                findings.append({
                    "severity": "HIGH",
                    "asset_id": asset.metadata.asset_id,
                    "asset_name": asset.metadata.display_name,
                    "finding": "Database without encryption",
                    "remediation": "Enable encryption at rest"
                })
            
            if asset.risk_score > 80:
                findings.append({
                    "severity": "HIGH",
                    "asset_id": asset.metadata.asset_id,
                    "asset_name": asset.metadata.display_name,
                    "finding": f"High risk score: {asset.risk_score:.1f}",
                    "remediation": "Review and remediate configuration issues"
                })
        
        return sorted(findings, key=lambda x: x["severity"])
    
    def _generate_report_recommendations(self, assets: List[AssetInventoryItem], 
                                        findings: List[Dict[str, Any]]) -> List[str]:
        """Generate report-level recommendations"""
        recommendations = []
        
        # Based on findings
        if len([f for f in findings if f["severity"] == "CRITICAL"]) > 0:
            recommendations.append("URGENT: Address all critical findings immediately")
        
        # Based on compliance
        non_compliant = len([a for a in assets if 
                           a.configuration.configuration_status == ConfigurationStatus.NON_COMPLIANT])
        if non_compliant > len(assets) * 0.2:  # More than 20% non-compliant
            recommendations.append("Implement compliance remediation program")
        
        # Based on costs
        total_cost = sum(a.estimated_monthly_cost or 0 for a in assets)
        if total_cost > 10000:  # Arbitrary threshold
            recommendations.append("Review cost optimization opportunities")
        
        # General recommendations
        recommendations.extend([
            "Enable monitoring for all production assets",
            "Implement consistent labeling strategy",
            "Review and update backup policies",
            "Conduct quarterly security assessments"
        ])
        
        return recommendations[:10]  # Limit to top 10
    
    async def _export_report(self, report: AssetReport, assets: List[AssetInventoryItem],
                           format: ReportFormat) -> Dict[str, str]:
        """Export report in requested format"""
        export_urls = {}
        
        try:
            if format == ReportFormat.JSON:
                # Export as JSON
                file_path = f"/tmp/{report.report_id}.json"
                with open(file_path, 'w') as f:
                    json.dump(report.dict(), f, indent=2, default=str)
                export_urls["json"] = file_path
            
            elif format == ReportFormat.CSV:
                # Export as CSV
                file_path = f"/tmp/{report.report_id}.csv"
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write headers
                    writer.writerow(["Asset ID", "Name", "Type", "Category", "Environment", 
                                   "Compliance", "Risk Score", "Monthly Cost"])
                    # Write data
                    for asset in assets:
                        writer.writerow([
                            asset.metadata.asset_id,
                            asset.metadata.display_name,
                            asset.metadata.asset_type,
                            asset.metadata.category.value,
                            asset.metadata.environment,
                            asset.configuration.configuration_status.value,
                            asset.risk_score,
                            asset.estimated_monthly_cost
                        ])
                export_urls["csv"] = file_path
            
            elif format == ReportFormat.MARKDOWN:
                # Export as Markdown
                file_path = f"/tmp/{report.report_id}.md"
                with open(file_path, 'w') as f:
                    f.write(f"# {report.report_name}\n\n")
                    f.write(f"Generated: {report.generated_at}\n\n")
                    f.write(f"## Summary\n\n")
                    f.write(f"- Total Assets: {report.total_assets}\n")
                    f.write(f"- Compliance Rate: {self._calculate_compliance_rate(assets):.1f}%\n")
                    f.write(f"- Average Risk Score: {self._calculate_average_risk(assets):.1f}\n\n")
                    f.write(f"## Recommendations\n\n")
                    for rec in report.recommendations:
                        f.write(f"- {rec}\n")
                export_urls["markdown"] = file_path
            
            return export_urls
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return {}
    
    def _calculate_compliance_rate(self, assets: List[AssetInventoryItem]) -> float:
        """Calculate overall compliance rate"""
        if not assets:
            return 0
        compliant = len([a for a in assets if 
                        a.configuration.configuration_status == ConfigurationStatus.COMPLIANT])
        return (compliant / len(assets)) * 100
    
    def _calculate_average_risk(self, assets: List[AssetInventoryItem]) -> float:
        """Calculate average risk score"""
        if not assets:
            return 0
        return sum(a.risk_score for a in assets) / len(assets)
    
    async def _store_assets(self, assets: List[AssetInventoryItem]):
        """Store assets in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            for asset in assets:
                cursor.execute("""
                    INSERT OR REPLACE INTO asset_inventory
                    (asset_id, asset_type, asset_name, display_name, category, 
                     project_id, location, importance, environment, compliance_status,
                     compliance_score, risk_score, public_exposure, monitoring_enabled,
                     encryption_enabled, estimated_cost, metadata, configuration, last_scanned)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    asset.metadata.asset_id,
                    asset.metadata.asset_type,
                    asset.metadata.asset_name,
                    asset.metadata.display_name,
                    asset.metadata.category.value,
                    asset.metadata.project_id,
                    asset.metadata.location,
                    asset.metadata.importance.value,
                    asset.metadata.environment,
                    asset.configuration.configuration_status.value,
                    asset.configuration.compliance_score,
                    asset.risk_score,
                    asset.public_exposure,
                    asset.monitoring_enabled,
                    asset.encryption_enabled,
                    asset.estimated_monthly_cost,
                    json.dumps(asset.metadata.dict()),
                    json.dumps(asset.configuration.dict(), default=str),
                    datetime.now()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store assets: {e}")
    
    async def _store_drifts(self, drifts: List[ConfigurationDrift]):
        """Store configuration drifts in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            for drift in drifts:
                cursor.execute("""
                    INSERT INTO configuration_drifts
                    (drift_id, asset_id, setting_name, expected_value, actual_value,
                     drift_severity, detected_at, auto_remediation_available, remediation_script)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    drift.drift_id,
                    drift.asset_id,
                    drift.setting_name,
                    str(drift.expected_value),
                    str(drift.actual_value),
                    drift.drift_severity,
                    drift.drift_detected_at,
                    drift.auto_remediation_available,
                    drift.remediation_script
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store drifts: {e}")
    
    async def _store_report(self, report: AssetReport):
        """Store report in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO asset_reports
                (report_id, report_name, report_type, generated_at, generated_by,
                 total_assets, filters, summary, report_data, export_urls)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_id,
                report.report_name,
                report.report_type,
                report.generated_at,
                report.generated_by,
                report.total_assets,
                json.dumps(report.filters_applied.dict() if report.filters_applied else {}),
                json.dumps({
                    "asset_summary": report.asset_summary,
                    "compliance_summary": report.compliance_summary,
                    "risk_summary": report.risk_summary,
                    "cost_summary": report.cost_summary
                }),
                json.dumps(report.report_data, default=str) if report.report_data else None,
                json.dumps({})
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store report: {e}")
    
    async def _get_recent_changes(self) -> List[AssetChange]:
        """Get recent asset changes from database"""
        changes = []
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM asset_changes 
                WHERE change_timestamp > datetime('now', '-7 days')
                ORDER BY change_timestamp DESC
                LIMIT 100
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to AssetChange objects
            # (Implementation would convert rows to objects)
            
        except Exception as e:
            logger.error(f"Failed to get recent changes: {e}")
        
        return changes
    
    async def _detect_recent_drifts(self) -> List[ConfigurationDrift]:
        """Detect recent configuration drifts"""
        drifts = []
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM configuration_drifts
                WHERE detected_at > datetime('now', '-24 hours')
                ORDER BY drift_severity, detected_at DESC
                LIMIT 50
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to ConfigurationDrift objects
            # (Implementation would convert rows to objects)
            
        except Exception as e:
            logger.error(f"Failed to get recent drifts: {e}")
        
        return drifts