"""
Direct GCP API implementation for real resource discovery.

This module provides direct access to GCP APIs without ADK agents,
ensuring we can get real data from your project.
"""

import os
import logging
from typing import Dict, Any, List
from google.cloud import asset_v1
from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import iam
from google.oauth2 import service_account
import google.auth

logger = logging.getLogger(__name__)


class GCPDirectClient:
    """Direct client for GCP APIs."""
    
    def __init__(self, project_id: str):
        """Initialize GCP client with project ID."""
        self.project_id = project_id
        
        # Try to get credentials
        try:
            # First check for service account file
            import os
            from pathlib import Path
            
            # Check multiple possible locations for service account key
            secrets_dir = Path(__file__).parent.parent / "config" / "secrets"
            possible_paths = []
            
            # First check if GOOGLE_APPLICATION_CREDENTIALS is set
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                possible_paths.append(Path(os.getenv('GOOGLE_APPLICATION_CREDENTIALS')))
            
            # Then check for any JSON files in secrets directory
            if secrets_dir.exists():
                possible_paths.extend(secrets_dir.glob("*.json"))
            
            # Finally check for default gcloud credentials
            possible_paths.append(Path(os.path.expanduser("~/.config/gcloud/application_default_credentials.json")))
            
            service_account_file = None
            for path in possible_paths:
                if path.exists():
                    service_account_file = str(path)
                    logger.info(f"Found service account key at: {service_account_file}")
                    break
            
            if service_account_file:
                from google.oauth2 import service_account
                self.credentials = service_account.Credentials.from_service_account_file(
                    service_account_file
                )
                logger.info(f"Using service account credentials for project {project_id}")
            else:
                # Fall back to default credentials (gcloud auth)
                self.credentials, _ = google.auth.default()
                logger.info(f"Using default credentials for project {project_id}")
                
        except Exception as e:
            logger.warning(f"Could not get credentials: {e}")
            self.credentials = None
    
    async def discover_resources(self) -> Dict[str, Any]:
        """Discover all resources in the project."""
        resources = {
            "project_id": self.project_id,
            "compute": [],
            "storage": [],
            "iam": [],
            "summary": {}
        }
        
        try:
            # Try Cloud Asset Inventory first (most comprehensive)
            try:
                asset_client = asset_v1.AssetServiceClient(credentials=self.credentials)
                parent = f"projects/{self.project_id}"
                
                # List all assets
                assets = asset_client.list_assets(
                    request={"parent": parent, "page_size": 100}
                )
                
                asset_count = 0
                asset_types = {}
                
                for asset in assets:
                    asset_count += 1
                    asset_type = asset.asset_type.split('/')[-1]
                    
                    if asset_type not in asset_types:
                        asset_types[asset_type] = 0
                    asset_types[asset_type] += 1
                    
                    # Add to appropriate category
                    if 'compute' in asset.asset_type.lower():
                        resources["compute"].append({
                            "name": asset.name.split('/')[-1],
                            "type": asset_type,
                            "full_name": asset.name
                        })
                    elif 'storage' in asset.asset_type.lower() or 'bucket' in asset.asset_type.lower():
                        resources["storage"].append({
                            "name": asset.name.split('/')[-1],
                            "type": asset_type,
                            "full_name": asset.name
                        })
                    elif 'iam' in asset.asset_type.lower() or 'serviceAccount' in asset.asset_type.lower():
                        resources["iam"].append({
                            "name": asset.name.split('/')[-1],
                            "type": asset_type,
                            "full_name": asset.name
                        })
                
                resources["summary"] = {
                    "total_assets": asset_count,
                    "asset_types": asset_types,
                    "compute_count": len(resources["compute"]),
                    "storage_count": len(resources["storage"]),
                    "iam_count": len(resources["iam"])
                }
                
                logger.info(f"Discovered {asset_count} assets via Cloud Asset Inventory")
                
            except Exception as asset_error:
                logger.warning(f"Cloud Asset Inventory not available: {asset_error}")
                
                # Fallback to individual APIs
                # Try Compute Engine
                try:
                    compute_client = compute_v1.InstancesClient(credentials=self.credentials)
                    
                    # List all zones first
                    zones_client = compute_v1.ZonesClient(credentials=self.credentials)
                    zones = zones_client.list(project=self.project_id)
                    
                    for zone in zones:
                        instances = compute_client.list(project=self.project_id, zone=zone.name)
                        for instance in instances:
                            resources["compute"].append({
                                "name": instance.name,
                                "type": "Instance",
                                "zone": zone.name,
                                "status": instance.status,
                                "machine_type": instance.machine_type.split('/')[-1]
                            })
                    
                    logger.info(f"Found {len(resources['compute'])} compute instances")
                    
                except Exception as compute_error:
                    logger.warning(f"Could not list compute instances: {compute_error}")
                
                # Try Cloud Storage
                try:
                    storage_client = storage.Client(project=self.project_id, credentials=self.credentials)
                    buckets = storage_client.list_buckets()
                    
                    for bucket in buckets:
                        resources["storage"].append({
                            "name": bucket.name,
                            "type": "Bucket",
                            "location": bucket.location,
                            "storage_class": bucket.storage_class
                        })
                    
                    logger.info(f"Found {len(resources['storage'])} storage buckets")
                    
                except Exception as storage_error:
                    logger.warning(f"Could not list storage buckets: {storage_error}")
                
                # Try IAM
                try:
                    iam_client = iam.IAMClient(credentials=self.credentials)
                    service_accounts = iam_client.list_service_accounts(
                        name=f"projects/{self.project_id}"
                    )
                    
                    for sa in service_accounts:
                        resources["iam"].append({
                            "name": sa.email,
                            "type": "ServiceAccount",
                            "display_name": sa.display_name or "N/A"
                        })
                    
                    logger.info(f"Found {len(resources['iam'])} service accounts")
                    
                except Exception as iam_error:
                    logger.warning(f"Could not list IAM resources: {iam_error}")
                
                # Update summary
                resources["summary"] = {
                    "total_assets": len(resources["compute"]) + len(resources["storage"]) + len(resources["iam"]),
                    "compute_count": len(resources["compute"]),
                    "storage_count": len(resources["storage"]),
                    "iam_count": len(resources["iam"]),
                    "method": "individual_apis"
                }
        
        except Exception as e:
            logger.error(f"Error discovering resources: {e}")
            resources["error"] = str(e)
        
        return resources
    
    def analyze_security(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security posture of discovered resources."""
        findings = []
        
        # Check for public storage buckets
        for bucket in resources.get("storage", []):
            # In a real implementation, check bucket IAM policies
            findings.append({
                "resource": bucket["name"],
                "type": "storage",
                "severity": "INFO",
                "finding": f"Storage bucket '{bucket['name']}' should be reviewed for public access"
            })
        
        # Check for default service accounts
        for sa in resources.get("iam", []):
            if "compute@developer.gserviceaccount.com" in sa.get("name", ""):
                findings.append({
                    "resource": sa["name"],
                    "type": "iam",
                    "severity": "MEDIUM",
                    "finding": "Default compute service account in use - consider using custom service accounts"
                })
        
        # Check compute instances
        for instance in resources.get("compute", []):
            if instance.get("status") == "RUNNING":
                findings.append({
                    "resource": instance["name"],
                    "type": "compute",
                    "severity": "INFO",
                    "finding": f"Instance '{instance['name']}' is running - ensure it's needed"
                })
        
        return {
            "total_findings": len(findings),
            "findings": findings,
            "summary": {
                "critical": len([f for f in findings if f["severity"] == "CRITICAL"]),
                "high": len([f for f in findings if f["severity"] == "HIGH"]),
                "medium": len([f for f in findings if f["severity"] == "MEDIUM"]),
                "low": len([f for f in findings if f["severity"] == "LOW"]),
                "info": len([f for f in findings if f["severity"] == "INFO"])
            }
        }


async def process_gcp_query(project_id: str, query: str) -> str:
    """Process a query using direct GCP API calls."""
    client = GCPDirectClient(project_id)
    
    query_lower = query.lower()
    
    # Check for permission queries
    if any(word in query_lower for word in ["permission", "role", "iam policy", "access", "what can"]) and any(word in query_lower for word in ["service account", "these", "they"]):
        # Get IAM policy bindings
        response = f"[STATS] **IAM Permissions for Service Accounts in {project_id}:**\n\n"
        
        try:
            from google.cloud import resourcemanager_v3
            
            # Get project IAM policy
            projects_client = resourcemanager_v3.ProjectsClient(credentials=client.credentials)
            project_name = f"projects/{project_id}"
            
            policy = projects_client.get_iam_policy(
                request={"resource": project_name}
            )
            
            # Map service accounts to their roles
            sa_roles = {}
            for binding in policy.bindings:
                role = binding.role
                for member in binding.members:
                    if "serviceAccount:" in member:
                        sa_email = member.replace("serviceAccount:", "")
                        if sa_email not in sa_roles:
                            sa_roles[sa_email] = []
                        sa_roles[sa_email].append(role)
            
            # Display permissions for each service account
            if sa_roles:
                for sa_email, roles in sa_roles.items():
                    response += f"**{sa_email}**\n"
                    response += f"  Roles ({len(roles)}):\n"
                    for role in sorted(roles):
                        role_name = role.replace("roles/", "")
                        response += f"  * `{role_name}`\n"
                    response += "\n"
            else:
                response += "No service account permissions found at project level.\n"
                response += "Note: Service accounts may have permissions at resource level.\n"
            
            # Add recommendations
            response += "\n**Security Recommendations:**\n"
            
            # Check for overly permissive roles
            dangerous_roles = ["owner", "editor", "iam.securityAdmin"]
            for sa_email, roles in sa_roles.items():
                for role in roles:
                    if any(danger in role.lower() for danger in dangerous_roles):
                        response += f"[WARNING] **Warning**: {sa_email} has `{role}` - consider using more restrictive roles\n"
            
            # Check for default service accounts with permissions
            if any("compute@developer" in sa or "@appspot" in sa for sa in sa_roles.keys()):
                response += "[WARNING] **Warning**: Default service accounts have permissions - consider using custom service accounts\n"
                
        except Exception as e:
            logger.error(f"Error getting IAM policies: {e}")
            response += f"Error retrieving IAM policies: {str(e)}\n"
            response += "\nEnsure the Resource Manager API is enabled and you have appropriate permissions.\n"
        
        return response
    
    # Check for service account specific queries
    elif any(word in query_lower for word in ["service account", "serviceaccount", "iam account", "iam resource"]):
        # Get IAM-specific information
        resources = await client.discover_resources()
        
        if "error" in resources:
            return f"[ERROR] Error accessing GCP APIs: {resources['error']}"
        
        response = f"[STATS] **Service Accounts in project {project_id}:**\n\n"
        
        # Extract service accounts from IAM resources
        service_accounts = []
        if resources.get("iam"):
            for iam_resource in resources["iam"]:
                if "@" in iam_resource.get("name", ""):
                    service_accounts.append(iam_resource["name"])
        
        # Also check in the raw data for service accounts
        try:
            from google.cloud import iam_admin_v1
            iam_client = iam_admin_v1.IAMClient(credentials=client.credentials)
            
            # List service accounts directly
            request = iam_admin_v1.ListServiceAccountsRequest(
                name=f"projects/{project_id}"
            )
            
            sa_list = iam_client.list_service_accounts(request=request)
            
            response += "**Active Service Accounts:**\n"
            count = 0
            for sa in sa_list:
                count += 1
                response += f"* **{sa.display_name or 'Unnamed'}**\n"
                response += f"  - Email: `{sa.email}`\n"
                response += f"  - Unique ID: {sa.unique_id}\n"
                response += f"  - Created: {sa.name.split('/')[-1]}\n\n"
                
                if count >= 10:  # Limit output
                    response += f"...and more service accounts\n"
                    break
            
            if count == 0:
                response += "No service accounts found.\n"
            else:
                response += f"\n**Total Service Accounts:** {count}\n"
                
        except Exception as e:
            logger.warning(f"Could not list service accounts directly: {e}")
            
            # Fall back to showing what we found in resources
            if service_accounts:
                response += "**Service Accounts found in resources:**\n"
                for sa in service_accounts:
                    response += f"* {sa}\n"
            else:
                response += "No service accounts found in resource inventory.\n"
                response += "\nTry enabling the IAM API or checking permissions.\n"
        
        return response
    
    elif any(word in query_lower for word in ["resource", "asset", "what do i have", "inventory", "list"]):
        # Resource discovery
        resources = await client.discover_resources()
        
        if "error" in resources:
            return f"[ERROR] Error accessing GCP APIs: {resources['error']}\n\nPlease ensure:\n1. You're authenticated with gcloud\n2. APIs are enabled for project {project_id}\n3. You have necessary permissions"
        
        response = f"[STATS] **GCP Resources in project {project_id}:**\n\n"
        
        if resources.get("summary"):
            summary = resources["summary"]
            response += f"**Summary:**\n"
            response += f"* Total assets: {summary.get('total_assets', 0)}\n"
            response += f"* Compute instances: {summary.get('compute_count', 0)}\n"
            response += f"* Storage buckets: {summary.get('storage_count', 0)}\n"
            response += f"* IAM resources: {summary.get('iam_count', 0)}\n\n"
        
        if resources["compute"]:
            response += f"**Compute Resources ({len(resources['compute'])}):**\n"
            for r in resources["compute"][:5]:
                response += f"* {r['name']} ({r.get('type', 'Instance')})\n"
            if len(resources["compute"]) > 5:
                response += f"  ...and {len(resources['compute']) - 5} more\n"
            response += "\n"
        
        if resources["storage"]:
            response += f"**Storage Resources ({len(resources['storage'])}):**\n"
            for r in resources["storage"][:5]:
                response += f"* {r['name']} ({r.get('location', 'unknown location')})\n"
            if len(resources["storage"]) > 5:
                response += f"  ...and {len(resources['storage']) - 5} more\n"
            response += "\n"
        
        if resources["iam"]:
            response += f"**IAM Resources ({len(resources['iam'])}):**\n"
            for r in resources["iam"][:5]:
                response += f"* {r['name']}\n"
            if len(resources["iam"]) > 5:
                response += f"  ...and {len(resources['iam']) - 5} more\n"
        
        if not any([resources["compute"], resources["storage"], resources["iam"]]):
            response += "[INFO] No resources found. This could mean:\n"
            response += "* The project has no resources yet\n"
            response += "* APIs need to be enabled\n"
            response += "* Permissions need to be granted\n"
        
        return response
    
    elif any(word in query_lower for word in ["security", "vulnerabilit", "risk", "finding"]):
        # Security analysis
        resources = await client.discover_resources()
        
        if "error" in resources:
            return f"[ERROR] Error accessing GCP APIs: {resources['error']}"
        
        analysis = client.analyze_security(resources)
        
        response = f"[SECURITY] **Security Analysis for project {project_id}:**\n\n"
        response += f"**Summary:**\n"
        response += f"* Total findings: {analysis['total_findings']}\n"
        
        summary = analysis['summary']
        if summary['critical'] > 0:
            response += f"* [CRITICAL] Critical: {summary['critical']}\n"
        if summary['high'] > 0:
            response += f"* [HIGH] High: {summary['high']}\n"
        if summary['medium'] > 0:
            response += f"* [MEDIUM] Medium: {summary['medium']}\n"
        if summary['low'] > 0:
            response += f"* [LOW] Low: {summary['low']}\n"
        if summary['info'] > 0:
            response += f"* [INFO] Info: {summary['info']}\n"
        
        response += "\n**Key Findings:**\n"
        for finding in analysis['findings'][:10]:
            icon = {"CRITICAL": "[CRITICAL]", "HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]", "INFO": "[INFO]"}.get(finding['severity'], "*")
            response += f"{icon} {finding['finding']}\n"
        
        if len(analysis['findings']) > 10:
            response += f"\n...and {len(analysis['findings']) - 10} more findings\n"
        
        return response
    
    else:
        # General help
        return (
            f"[SEARCH] **GCP Security Assistant for project: {project_id}**\n\n"
            "I can help you with:\n\n"
            "* **Resource Discovery**: 'What resources do I have?'\n"
            "* **Security Analysis**: 'Check my security posture'\n"
            "* **IAM Review**: 'Show my service accounts'\n"
            "* **Vulnerability Scan**: 'Find security issues'\n\n"
            "What would you like to explore?"
        )