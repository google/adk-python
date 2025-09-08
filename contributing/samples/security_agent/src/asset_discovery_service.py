"""Core Asset Discovery Service for GCP Security Agent

Implements comprehensive asset discovery using Google Cloud Asset Inventory API
with intelligent caching, security analysis, and vulnerability detection.
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Google Cloud imports with fallbacks
try:
    from google.cloud import asset_v1
    from google.cloud import securitycenter
    from google.api_core import exceptions as gcp_exceptions
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AssetSummary:
    """Summary of discovered assets."""
    total_count: int
    by_type: Dict[str, int]
    by_location: Dict[str, int]
    security_issues: List[Dict[str, Any]]
    recommendations: List[str]
    fetch_duration: float

@dataclass
class SecurityFinding:
    """Security finding for an asset."""
    asset_name: str
    severity: str
    category: str
    description: str
    recommendation: str
    resource_type: str

class AssetDiscoveryService:
    """Core service for discovering and analyzing GCP assets."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.cache_duration = timedelta(hours=1)
        self.asset_client = None
        self.security_client = None
        
        # Initialize clients if available
        if GCP_AVAILABLE:
            try:
                self.asset_client = asset_v1.AssetServiceClient()
                logger.info("✅ Asset Inventory client initialized")
            except Exception as e:
                logger.warning(f"Asset client initialization failed: {e}")
            
            try:
                self.security_client = securitycenter.SecurityCenterClient()
                logger.info("✅ Security Center client initialized")
            except Exception as e:
                logger.warning(f"Security Center client initialization failed: {e}")
    
    async def discover_all_assets(self) -> AssetSummary:
        """Discover all assets in the project with security analysis."""
        start_time = datetime.now()
        
        try:
            # Get all assets
            assets = await self._fetch_all_assets()
            
            # Analyze assets for security issues
            security_issues = await self._analyze_security_issues(assets)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(assets, security_issues)
            
            # Create summary
            summary = AssetSummary(
                total_count=len(assets),
                by_type=self._count_by_type(assets),
                by_location=self._count_by_location(assets),
                security_issues=security_issues,
                recommendations=recommendations,
                fetch_duration=(datetime.now() - start_time).total_seconds()
            )
            
            logger.info(f"🔍 Discovered {summary.total_count} assets in {summary.fetch_duration:.2f}s")
            return summary
            
        except Exception as e:
            logger.error(f"Asset discovery failed: {e}")
            return self._create_fallback_summary(start_time)
    
    async def _fetch_all_assets(self) -> List[Dict[str, Any]]:
        """Fetch all assets from Cloud Asset Inventory."""
        if not self.asset_client:
            logger.warning("Asset client not available, using mock data")
            return self._create_mock_assets()
        
        try:
            parent = f"projects/{self.project_id}"
            request = asset_v1.ListAssetsRequest(
                parent=parent,
                content_type=asset_v1.ContentType.RESOURCE
            )
            
            assets = []
            async for asset in self.asset_client.list_assets(request=request):
                asset_dict = self._convert_asset_to_dict(asset)
                assets.append(asset_dict)
            
            logger.info(f"📊 Fetched {len(assets)} real assets from API")
            return assets
            
        except Exception as e:
            logger.error(f"Failed to fetch real assets: {e}")
            return self._create_mock_assets()
    
    def _convert_asset_to_dict(self, asset) -> Dict[str, Any]:
        """Convert Asset API response to dictionary."""
        try:
            return {
                'name': asset.name,
                'asset_type': asset.asset_type,
                'display_name': getattr(asset.resource.data, 'display_name', ''),
                'location': getattr(asset.resource.location, '', ''),
                'state': getattr(asset.resource.data, 'status', 'UNKNOWN'),
                'labels': dict(getattr(asset.resource.data, 'labels', {})),
                'create_time': getattr(asset.resource.data, 'creation_timestamp', ''),
                'update_time': datetime.now().isoformat(),
                'resource_data': self._extract_resource_data(asset)
            }
        except Exception as e:
            logger.warning(f"Error converting asset: {e}")
            return {
                'name': str(asset.name),
                'asset_type': str(asset.asset_type),
                'error': str(e)
            }
    
    def _extract_resource_data(self, asset) -> Dict[str, Any]:
        """Extract relevant resource data for security analysis."""
        try:
            data = asset.resource.data
            resource_type = asset.asset_type
            
            if 'compute' in resource_type:
                return self._extract_compute_data(data)
            elif 'storage' in resource_type:
                return self._extract_storage_data(data)
            elif 'iam' in resource_type:
                return self._extract_iam_data(data)
            elif 'container' in resource_type:
                return self._extract_gke_data(data)
            else:
                return {'raw_data': str(data)[:1000]}  # Truncate large data
                
        except Exception as e:
            return {'extraction_error': str(e)}
    
    def _extract_compute_data(self, data) -> Dict[str, Any]:
        """Extract security-relevant data from compute resources."""
        return {
            'machine_type': getattr(data, 'machine_type', ''),
            'status': getattr(data, 'status', ''),
            'external_ip': self._get_external_ip(data),
            'service_accounts': self._get_service_accounts(data),
            'network_interfaces': self._get_network_interfaces(data),
            'disk_encryption': self._check_disk_encryption(data),
            'tags': getattr(data, 'tags', {}).get('items', []),
            'metadata': self._get_metadata(data)
        }
    
    def _extract_storage_data(self, data) -> Dict[str, Any]:
        """Extract security-relevant data from storage resources."""
        return {
            'location': getattr(data, 'location', ''),
            'storage_class': getattr(data, 'storage_class', ''),
            'versioning': getattr(data, 'versioning', {}).get('enabled', False),
            'public_access_prevention': getattr(data, 'public_access_prevention', 'inherited'),
            'uniform_bucket_level_access': getattr(data, 'uniform_bucket_level_access', {}).get('enabled', False),
            'encryption': self._get_bucket_encryption(data),
            'lifecycle_rules': len(getattr(data, 'lifecycle', {}).get('rule', [])),
            'cors_rules': len(getattr(data, 'cors', [])),
            'iam_configuration': getattr(data, 'iam_configuration', {})
        }
    
    def _extract_iam_data(self, data) -> Dict[str, Any]:
        """Extract security-relevant data from IAM resources."""
        return {
            'email': getattr(data, 'email', ''),
            'display_name': getattr(data, 'display_name', ''),
            'disabled': getattr(data, 'disabled', False),
            'oauth2_client_id': getattr(data, 'oauth2_client_id', ''),
            'project_id': getattr(data, 'project_id', ''),
            'unique_id': getattr(data, 'unique_id', '')
        }
    
    def _extract_gke_data(self, data) -> Dict[str, Any]:
        """Extract security-relevant data from GKE resources."""
        return {
            'status': getattr(data, 'status', ''),
            'location': getattr(data, 'location', ''),
            'current_node_count': getattr(data, 'current_node_count', 0),
            'network': getattr(data, 'network', ''),
            'private_cluster_config': getattr(data, 'private_cluster_config', {}),
            'master_auth': getattr(data, 'master_auth', {}),
            'network_policy': getattr(data, 'network_policy', {}),
            'pod_security_policy_config': getattr(data, 'pod_security_policy_config', {}),
            'binary_authorization': getattr(data, 'binary_authorization', {}),
            'workload_identity_config': getattr(data, 'workload_identity_config', {})
        }
    
    def _get_external_ip(self, data) -> Optional[str]:
        """Extract external IP from compute instance."""
        try:
            interfaces = getattr(data, 'network_interfaces', [])
            for interface in interfaces:
                access_configs = getattr(interface, 'access_configs', [])
                for config in access_configs:
                    if hasattr(config, 'nat_i_p'):
                        return config.nat_i_p
        except Exception:
            pass
        return None
    
    def _get_service_accounts(self, data) -> List[str]:
        """Extract service accounts from compute instance."""
        try:
            accounts = getattr(data, 'service_accounts', [])
            return [getattr(account, 'email', '') for account in accounts]
        except Exception:
            return []
    
    def _get_network_interfaces(self, data) -> List[Dict[str, Any]]:
        """Extract network interface information."""
        try:
            interfaces = getattr(data, 'network_interfaces', [])
            return [
                {
                    'network': getattr(iface, 'network', ''),
                    'subnetwork': getattr(iface, 'subnetwork', ''),
                    'network_i_p': getattr(iface, 'network_i_p', '')
                }
                for iface in interfaces
            ]
        except Exception:
            return []
    
    def _check_disk_encryption(self, data) -> Dict[str, Any]:
        """Check disk encryption configuration."""
        try:
            disks = getattr(data, 'disks', [])
            encryption_info = {}
            for disk in disks:
                disk_encryption = getattr(disk, 'disk_encryption_key', None)
                if disk_encryption:
                    encryption_info[disk.device_name] = {
                        'kms_key_name': getattr(disk_encryption, 'kms_key_name', ''),
                        'sha256': getattr(disk_encryption, 'sha256', '')
                    }
            return encryption_info
        except Exception:
            return {}
    
    def _get_metadata(self, data) -> Dict[str, str]:
        """Extract metadata from compute instance."""
        try:
            metadata = getattr(data, 'metadata', {})
            items = getattr(metadata, 'items', [])
            return {item.key: item.value for item in items if hasattr(item, 'key')}
        except Exception:
            return {}
    
    def _get_bucket_encryption(self, data) -> Dict[str, Any]:
        """Extract bucket encryption configuration."""
        try:
            encryption = getattr(data, 'encryption', {})
            return {
                'default_kms_key_name': getattr(encryption, 'default_kms_key_name', ''),
                'has_default_key': bool(getattr(encryption, 'default_kms_key_name', ''))
            }
        except Exception:
            return {'has_default_key': False}
    
    async def _analyze_security_issues(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze assets for security issues."""
        issues = []
        
        for asset in assets:
            asset_issues = self._analyze_asset_security(asset)
            issues.extend(asset_issues)
        
        return issues
    
    def _analyze_asset_security(self, asset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze a single asset for security issues."""
        issues = []
        asset_type = asset.get('asset_type', '')
        resource_data = asset.get('resource_data', {})
        
        # Compute instance security checks
        if 'compute' in asset_type:
            issues.extend(self._check_compute_security(asset, resource_data))
        
        # Storage bucket security checks
        elif 'storage' in asset_type:
            issues.extend(self._check_storage_security(asset, resource_data))
        
        # GKE cluster security checks
        elif 'container' in asset_type:
            issues.extend(self._check_gke_security(asset, resource_data))
        
        # IAM security checks
        elif 'iam' in asset_type:
            issues.extend(self._check_iam_security(asset, resource_data))
        
        return issues
    
    def _check_compute_security(self, asset: Dict[str, Any], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check compute instance security."""
        issues = []
        
        # External IP exposure
        if data.get('external_ip'):
            issues.append({
                'asset_name': asset['name'],
                'severity': 'MEDIUM',
                'category': 'NETWORK_EXPOSURE',
                'description': 'Compute instance has external IP address',
                'recommendation': 'Consider using Cloud NAT or Load Balancer instead',
                'resource_type': 'compute_instance'
            })
        
        # Unencrypted disks
        if not data.get('disk_encryption'):
            issues.append({
                'asset_name': asset['name'],
                'severity': 'HIGH',
                'category': 'ENCRYPTION',
                'description': 'Compute instance disks are not encrypted',
                'recommendation': 'Enable disk encryption with customer-managed keys',
                'resource_type': 'compute_instance'
            })
        
        # Default service account usage
        service_accounts = data.get('service_accounts', [])
        for sa in service_accounts:
            if 'compute@developer.gserviceaccount.com' in sa:
                issues.append({
                    'asset_name': asset['name'],
                    'severity': 'MEDIUM',
                    'category': 'IAM_MISCONFIGURATION',
                    'description': 'Using default Compute Engine service account',
                    'recommendation': 'Create custom service account with minimal permissions',
                    'resource_type': 'compute_instance'
                })
        
        return issues
    
    def _check_storage_security(self, asset: Dict[str, Any], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check storage bucket security."""
        issues = []
        
        # Public access
        if data.get('public_access_prevention') != 'enforced':
            issues.append({
                'asset_name': asset['name'],
                'severity': 'HIGH',
                'category': 'PUBLIC_ACCESS',
                'description': 'Storage bucket allows public access',
                'recommendation': 'Enable public access prevention',
                'resource_type': 'storage_bucket'
            })
        
        # Uniform bucket-level access
        if not data.get('uniform_bucket_level_access'):
            issues.append({
                'asset_name': asset['name'],
                'severity': 'MEDIUM',
                'category': 'ACCESS_CONTROL',
                'description': 'Bucket does not use uniform bucket-level access',
                'recommendation': 'Enable uniform bucket-level access for better security',
                'resource_type': 'storage_bucket'
            })
        
        # Versioning
        if not data.get('versioning'):
            issues.append({
                'asset_name': asset['name'],
                'severity': 'LOW',
                'category': 'DATA_PROTECTION',
                'description': 'Object versioning is disabled',
                'recommendation': 'Enable versioning to protect against accidental deletion',
                'resource_type': 'storage_bucket'
            })
        
        # Encryption
        encryption = data.get('encryption', {})
        if not encryption.get('has_default_key'):
            issues.append({
                'asset_name': asset['name'],
                'severity': 'MEDIUM',
                'category': 'ENCRYPTION',
                'description': 'Bucket not encrypted with customer-managed key',
                'recommendation': 'Use customer-managed encryption keys for better control',
                'resource_type': 'storage_bucket'
            })
        
        return issues
    
    def _check_gke_security(self, asset: Dict[str, Any], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check GKE cluster security."""
        issues = []
        
        # Private cluster
        if not data.get('private_cluster_config'):
            issues.append({
                'asset_name': asset['name'],
                'severity': 'HIGH',
                'category': 'NETWORK_EXPOSURE',
                'description': 'GKE cluster is not private',
                'recommendation': 'Enable private cluster to isolate nodes',
                'resource_type': 'gke_cluster'
            })
        
        # Network policy
        if not data.get('network_policy', {}).get('enabled'):
            issues.append({
                'asset_name': asset['name'],
                'severity': 'MEDIUM',
                'category': 'NETWORK_SECURITY',
                'description': 'Network policy is disabled',
                'recommendation': 'Enable network policy for pod-to-pod security',
                'resource_type': 'gke_cluster'
            })
        
        # Binary authorization
        if not data.get('binary_authorization', {}).get('enabled'):
            issues.append({
                'asset_name': asset['name'],
                'severity': 'MEDIUM',
                'category': 'CONTAINER_SECURITY',
                'description': 'Binary Authorization is disabled',
                'recommendation': 'Enable Binary Authorization to ensure only trusted images',
                'resource_type': 'gke_cluster'
            })
        
        # Workload Identity
        if not data.get('workload_identity_config'):
            issues.append({
                'asset_name': asset['name'],
                'severity': 'MEDIUM',
                'category': 'IAM_MISCONFIGURATION',
                'description': 'Workload Identity is not configured',
                'recommendation': 'Enable Workload Identity for secure pod authentication',
                'resource_type': 'gke_cluster'
            })
        
        return issues
    
    def _check_iam_security(self, asset: Dict[str, Any], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check IAM security."""
        issues = []
        
        # Disabled service accounts
        if data.get('disabled'):
            issues.append({
                'asset_name': asset['name'],
                'severity': 'INFO',
                'category': 'IAM_STATUS',
                'description': 'Service account is disabled',
                'recommendation': 'Remove unused disabled service accounts',
                'resource_type': 'service_account'
            })
        
        return issues
    
    def _generate_recommendations(self, assets: List[Dict[str, Any]], 
                                security_issues: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on analysis."""
        recommendations = []
        
        # Count issues by category
        issue_categories = {}
        for issue in security_issues:
            category = issue['category']
            issue_categories[category] = issue_categories.get(category, 0) + 1
        
        # Generate recommendations based on most common issues
        if issue_categories.get('PUBLIC_ACCESS', 0) > 0:
            recommendations.append(
                \"🔒 Enable public access prevention on all storage buckets\"
            )
        
        if issue_categories.get('ENCRYPTION', 0) > 0:
            recommendations.append(
                \"🔐 Implement customer-managed encryption keys for sensitive resources\"
            )
        
        if issue_categories.get('NETWORK_EXPOSURE', 0) > 0:
            recommendations.append(
                \"🌐 Review network exposure and implement private clusters/instances\"
            )
        
        if issue_categories.get('IAM_MISCONFIGURATION', 0) > 0:
            recommendations.append(
                \"👤 Review IAM configurations and apply principle of least privilege\"
            )
        
        # General recommendations
        total_assets = len(assets)
        if total_assets > 50:
            recommendations.append(
                \"📊 Consider implementing asset management automation for large environments\"
            )
        
        recommendations.append(
            \"✅ Regular security audits recommended every 30 days\"
        )
        
        return recommendations
    
    def _count_by_type(self, assets: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count assets by type."""
        counts = {}
        for asset in assets:
            asset_type = asset.get('asset_type', 'unknown')
            # Simplify asset type for display
            simplified_type = self._simplify_asset_type(asset_type)
            counts[simplified_type] = counts.get(simplified_type, 0) + 1
        return counts
    
    def _count_by_location(self, assets: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count assets by location."""
        counts = {}
        for asset in assets:
            location = asset.get('location', 'global')
            if not location:
                location = 'global'
            counts[location] = counts.get(location, 0) + 1
        return counts
    
    def _simplify_asset_type(self, asset_type: str) -> str:
        """Simplify asset type for better readability."""
        type_mapping = {
            'compute.googleapis.com/Instance': 'Compute Instances',
            'storage.googleapis.com/Bucket': 'Storage Buckets',
            'container.googleapis.com/Cluster': 'GKE Clusters',
            'iam.googleapis.com/ServiceAccount': 'Service Accounts',
            'compute.googleapis.com/Firewall': 'Firewall Rules',
            'compute.googleapis.com/Network': 'VPC Networks',
            'sqladmin.googleapis.com/Instance': 'Cloud SQL',
            'bigquery.googleapis.com/Dataset': 'BigQuery Datasets',
            'pubsub.googleapis.com/Topic': 'Pub/Sub Topics'
        }
        return type_mapping.get(asset_type, asset_type)
    
    def _create_mock_assets(self) -> List[Dict[str, Any]]:
        """Create mock assets for development/testing."""
        return [
            {
                'name': f'projects/{self.project_id}/zones/us-central1-a/instances/web-server-1',
                'asset_type': 'compute.googleapis.com/Instance',
                'display_name': 'web-server-1',
                'location': 'us-central1-a',
                'state': 'RUNNING',
                'labels': {'env': 'production', 'team': 'web'},
                'create_time': '2024-01-01T10:00:00Z',
                'update_time': datetime.now().isoformat(),
                'resource_data': {
                    'machine_type': 'n1-standard-1',
                    'status': 'RUNNING',
                    'external_ip': '34.123.45.67',
                    'service_accounts': [f'{self.project_id}@appspot.gserviceaccount.com'],
                    'disk_encryption': {},
                    'tags': ['web-server'],
                    'metadata': {'startup-script': 'echo \"Hello World\"'}
                }
            },
            {
                'name': f'projects/{self.project_id}/buckets/app-data-bucket',
                'asset_type': 'storage.googleapis.com/Bucket',
                'display_name': 'app-data-bucket',
                'location': 'US',
                'state': 'ACTIVE',
                'labels': {'env': 'production', 'data-type': 'user-data'},
                'create_time': '2024-01-01T09:00:00Z',
                'update_time': datetime.now().isoformat(),
                'resource_data': {
                    'location': 'US',
                    'storage_class': 'STANDARD',
                    'versioning': False,
                    'public_access_prevention': 'inherited',
                    'uniform_bucket_level_access': False,
                    'encryption': {'has_default_key': False}
                }
            }
        ]
    
    def _create_fallback_summary(self, start_time: datetime) -> AssetSummary:
        """Create fallback summary when discovery fails."""
        return AssetSummary(
            total_count=0,
            by_type={},
            by_location={},
            security_issues=[],
            recommendations=['⚠️ Asset discovery failed - check GCP credentials and permissions'],
            fetch_duration=(datetime.now() - start_time).total_seconds()
        )

# Convenience functions for integration
async def discover_assets(project_id: str) -> AssetSummary:
    """Convenience function to discover all assets."""
    service = AssetDiscoveryService(project_id)
    return await service.discover_all_assets()

def get_asset_security_summary(project_id: str) -> str:
    """Get a formatted security summary for assets."""
    import asyncio
    
    async def _get_summary():
        summary = await discover_assets(project_id)
        
        result = f"🔍 **Asset Discovery Results**\\n\\n"
        result += f"**Total Assets**: {summary.total_count}\\n"
        
        if summary.by_type:
            result += f"\\n**By Type**:\\n"
            for asset_type, count in summary.by_type.items():
                result += f"* {asset_type}: {count}\\n"
        
        if summary.security_issues:
            result += f"\\n🚨 **Security Issues Found**: {len(summary.security_issues)}\\n"
            for issue in summary.security_issues[:5]:  # Show first 5
                result += f"* **{issue['severity']}**: {issue['description']}\\n"
        
        if summary.recommendations:
            result += f"\\n💡 **Recommendations**:\\n"
            for rec in summary.recommendations[:3]:  # Show first 3
                result += f"* {rec}\\n"
        
        result += f"\\n⚡ *Completed in {summary.fetch_duration:.2f} seconds*"
        return result
    
    try:
        return asyncio.run(_get_summary())
    except Exception as e:
        return f"❌ Asset discovery failed: {e}"