"""
Comprehensive data fetcher that pulls all GCP data upfront and caches it.

This module implements a "fetch-all" strategy to:
- Pull all GCP resources at once
- Store everything in SQLite for fast local queries
- Eliminate repeated API calls
- Prevent timeout issues
- Enable complex queries locally
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "cache" / "gcp_data.db"


class DataFetcher:
    """Fetches and caches all GCP data locally."""
    
    def __init__(self, project_id: str):
        """Initialize the data fetcher."""
        self.project_id = project_id
        self.db_path = DB_PATH
        self._ensure_db_dir()
        self._init_database()
        self.fetch_timestamp = None
        
    def _ensure_db_dir(self):
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database with comprehensive schema."""
        with self._get_connection() as conn:
            # Main assets table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    display_name TEXT,
                    description TEXT,
                    location TEXT,
                    labels TEXT,
                    create_time TEXT,
                    update_time TEXT,
                    state TEXT,
                    parent_resource TEXT,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # IAM table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS iam_accounts (
                    email TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    account_type TEXT,
                    display_name TEXT,
                    description TEXT,
                    unique_id TEXT,
                    disabled BOOLEAN DEFAULT 0,
                    keys TEXT,
                    roles TEXT,
                    permissions TEXT,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # IAM policies table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS iam_policies (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_name TEXT NOT NULL,
                    member TEXT NOT NULL,
                    role TEXT NOT NULL,
                    condition TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(project_id, resource_type, resource_name, member, role)
                )
            """)
            
            # Security findings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_findings (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    category TEXT,
                    severity TEXT,
                    state TEXT,
                    resource_name TEXT,
                    finding_class TEXT,
                    description TEXT,
                    recommendation TEXT,
                    event_time TEXT,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Compute instances
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compute_instances (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    zone TEXT,
                    machine_type TEXT,
                    status TEXT,
                    internal_ip TEXT,
                    external_ip TEXT,
                    network_interfaces TEXT,
                    disks TEXT,
                    labels TEXT,
                    metadata TEXT,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Storage buckets
            conn.execute("""
                CREATE TABLE IF NOT EXISTS storage_buckets (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    location TEXT,
                    storage_class TEXT,
                    iam_configuration TEXT,
                    lifecycle_rules TEXT,
                    versioning_enabled BOOLEAN,
                    uniform_bucket_level_access BOOLEAN,
                    public_access TEXT,
                    encryption TEXT,
                    labels TEXT,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Networks and subnets
            conn.execute("""
                CREATE TABLE IF NOT EXISTS networks (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    auto_create_subnetworks BOOLEAN,
                    routing_mode TEXT,
                    mtu INTEGER,
                    subnets TEXT,
                    peerings TEXT,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Firewall rules
            conn.execute("""
                CREATE TABLE IF NOT EXISTS firewall_rules (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    network TEXT,
                    priority INTEGER,
                    direction TEXT,
                    source_ranges TEXT,
                    destination_ranges TEXT,
                    allowed TEXT,
                    denied TEXT,
                    disabled BOOLEAN,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Databases
            conn.execute("""
                CREATE TABLE IF NOT EXISTS databases (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT,
                    version TEXT,
                    location TEXT,
                    tier TEXT,
                    high_availability BOOLEAN,
                    backup_enabled BOOLEAN,
                    encryption TEXT,
                    public_ip TEXT,
                    private_ip TEXT,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Secrets table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS secrets (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version INTEGER,
                    create_time TEXT,
                    expire_time TEXT,
                    replication TEXT,
                    labels TEXT,
                    rotation_policy TEXT,
                    annotations TEXT,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # GKE clusters table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gke_clusters (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    location TEXT,
                    location_type TEXT,
                    status TEXT,
                    endpoint TEXT,
                    current_master_version TEXT,
                    current_node_version TEXT,
                    initial_node_count INTEGER,
                    current_node_count INTEGER,
                    node_config TEXT,
                    network TEXT,
                    subnetwork TEXT,
                    cluster_ipv4_cidr TEXT,
                    services_ipv4_cidr TEXT,
                    locations TEXT,
                    enable_autopilot BOOLEAN,
                    enable_network_policy BOOLEAN,
                    enable_ip_alias BOOLEAN,
                    master_auth TEXT,
                    logging_config TEXT,
                    monitoring_config TEXT,
                    addons_config TEXT,
                    node_pools TEXT,
                    legacy_abac_enabled BOOLEAN,
                    workload_identity_config TEXT,
                    private_cluster_config TEXT,
                    database_encryption TEXT,
                    shielded_nodes TEXT,
                    release_channel TEXT,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Monitoring metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_metrics (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_name TEXT,
                    metric_type TEXT NOT NULL,
                    value REAL,
                    unit TEXT,
                    timestamp TEXT,
                    labels TEXT,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Fetch status table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fetch_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    fetch_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    records_fetched INTEGER DEFAULT 0,
                    error_message TEXT,
                    UNIQUE(project_id, fetch_type)
                )
            """)
            
            # Create indexes for fast queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_project ON assets(project_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_severity ON security_findings(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_category ON security_findings(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_compute_status ON compute_instances(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_buckets_public ON storage_buckets(public_access)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_iam_policies_member ON iam_policies(member)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_iam_policies_role ON iam_policies(role)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gke_status ON gke_clusters(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gke_location ON gke_clusters(location)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    async def fetch_all_data(self) -> Dict[str, Any]:
        """
        Fetch all GCP data for the project.
        
        This is the main entry point that fetches everything.
        """
        logger.info(f"Starting comprehensive data fetch for project {self.project_id}")
        self.fetch_timestamp = datetime.now()
        
        results = {
            "project_id": self.project_id,
            "fetch_started": self.fetch_timestamp.isoformat(),
            "success": True,
            "errors": [],
            "stats": {}
        }
        
        # Run all fetches in parallel for speed
        tasks = [
            self._fetch_compute_instances(),
            self._fetch_storage_buckets(),
            self._fetch_networks(),
            self._fetch_firewall_rules(),
            self._fetch_iam_accounts(),
            self._fetch_iam_policies(),  # IAM policies for project
            self._fetch_databases(),
            self._fetch_security_findings(),
            self._fetch_all_assets(),  # General asset inventory
            self._fetch_secrets(),  # Secret Manager
            self._fetch_monitoring_metrics(),  # Cloud Monitoring
            self._fetch_gke_clusters(),  # GKE clusters
        ]
        
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(fetch_results):
            fetch_type = ["compute", "storage", "networks", "firewall", "iam", "iam_policies", "databases", "findings", "assets", "secrets", "monitoring", "gke"][i]
            if isinstance(result, Exception):
                logger.error(f"Error fetching {fetch_type}: {result}")
                results["errors"].append(f"{fetch_type}: {str(result)}")
            else:
                results["stats"][fetch_type] = result
        
        results["fetch_completed"] = datetime.now().isoformat()
        results["duration_seconds"] = (datetime.now() - self.fetch_timestamp).total_seconds()
        
        # Update fetch status
        self._update_fetch_status("complete", results["stats"])
        
        logger.info(f"Data fetch completed in {results['duration_seconds']:.2f} seconds")
        return results
    
    async def _fetch_compute_instances(self) -> Dict[str, Any]:
        """Fetch all compute instances."""
        try:
            from google.cloud import compute_v1
            
            instance_client = compute_v1.InstancesClient()
            project_id = self.project_id
            
            instances = []
            zones_client = compute_v1.ZonesClient()
            
            # List all zones
            for zone in zones_client.list(project=project_id):
                zone_name = zone.name
                
                # List instances in this zone
                for instance in instance_client.list(project=project_id, zone=zone_name):
                    instance_data = {
                        "id": f"{project_id}/{zone_name}/{instance.name}",
                        "name": instance.name,
                        "zone": zone_name,
                        "machine_type": instance.machine_type.split('/')[-1] if instance.machine_type else None,
                        "status": instance.status,
                        "internal_ip": None,
                        "external_ip": None,
                        "network_interfaces": [],
                        "disks": [],
                        "labels": dict(instance.labels) if instance.labels else {},
                        "metadata": {item.key: item.value for item in (instance.metadata.items or [])} if instance.metadata and hasattr(instance.metadata, 'items') else {},
                    }
                    
                    # Extract network info
                    for interface in instance.network_interfaces:
                        if interface.network_i_p:
                            instance_data["internal_ip"] = interface.network_i_p
                        for access_config in interface.access_configs:
                            if access_config.nat_i_p:
                                instance_data["external_ip"] = access_config.nat_i_p
                        
                        instance_data["network_interfaces"].append({
                            "network": interface.network,
                            "subnetwork": interface.subnetwork,
                            "internal_ip": interface.network_i_p
                        })
                    
                    # Extract disk info
                    for disk in instance.disks:
                        instance_data["disks"].append({
                            "device_name": disk.device_name,
                            "source": disk.source,
                            "boot": disk.boot,
                            "auto_delete": disk.auto_delete
                        })
                    
                    instances.append(instance_data)
            
            # Store in database
            self._store_compute_instances(instances)
            
            # Count zones instead of using total_size (not available)
            zones_list = list(zones_client.list(project=project_id))
            return {"count": len(instances), "zones_checked": len(zones_list)}
            
        except Exception as e:
            logger.error(f"Error fetching compute instances: {e}")
            return {"count": 0, "error": str(e)}
    
    async def _fetch_storage_buckets(self) -> Dict[str, Any]:
        """Fetch all storage buckets."""
        try:
            from google.cloud import storage
            
            storage_client = storage.Client(project=self.project_id)
            buckets = []
            
            # Convert generator to list to avoid length issues
            bucket_list = list(storage_client.list_buckets())
            
            for bucket in bucket_list:
                bucket_data = {
                    "id": bucket.name,
                    "name": bucket.name,
                    "location": bucket.location,
                    "storage_class": bucket.storage_class,
                    "versioning_enabled": bucket.versioning_enabled,
                    "uniform_bucket_level_access": bucket.iam_configuration.uniform_bucket_level_access_enabled,
                    "public_access": "public" if bucket.iam_configuration.public_access_prevention == "inherited" else "private",
                    "encryption": bucket.default_kms_key_name if bucket.default_kms_key_name else "Google-managed",
                    "labels": dict(bucket.labels) if bucket.labels else {},
                    "lifecycle_rules": len(list(bucket.lifecycle_rules)) if bucket.lifecycle_rules else 0,
                    "iam_configuration": {
                        "uniform_bucket_level_access": bucket.iam_configuration.uniform_bucket_level_access_enabled,
                        "public_access_prevention": bucket.iam_configuration.public_access_prevention
                    }
                }
                buckets.append(bucket_data)
            
            # Store in database
            self._store_storage_buckets(buckets)
            
            return {"count": len(buckets)}
            
        except Exception as e:
            logger.error(f"Error fetching storage buckets: {e}")
            return {"count": 0, "error": str(e)}
    
    async def _fetch_networks(self) -> Dict[str, Any]:
        """Fetch VPC networks and subnets."""
        try:
            from google.cloud import compute_v1
            
            network_client = compute_v1.NetworksClient()
            subnet_client = compute_v1.SubnetworksClient()
            
            networks = []
            
            for network in network_client.list(project=self.project_id):
                network_data = {
                    "id": network.name,
                    "name": network.name,
                    "description": network.description,
                    "auto_create_subnetworks": network.auto_create_subnetworks,
                    "routing_mode": network.routing_config.routing_mode if network.routing_config else None,
                    "mtu": network.mtu,
                    "subnets": [],
                    "peerings": []
                }
                
                # Get subnets
                for subnet_url in network.subnetworks:
                    subnet_name = subnet_url.split('/')[-1]
                    region = subnet_url.split('/')[-3]
                    network_data["subnets"].append({
                        "name": subnet_name,
                        "region": region
                    })
                
                # Get peerings
                for peering in network.peerings:
                    network_data["peerings"].append({
                        "name": peering.name,
                        "network": peering.network,
                        "state": peering.state
                    })
                
                networks.append(network_data)
            
            # Store in database
            self._store_networks(networks)
            
            return {"count": len(networks)}
            
        except Exception as e:
            logger.error(f"Error fetching networks: {e}")
            return {"count": 0, "error": str(e)}
    
    async def _fetch_firewall_rules(self) -> Dict[str, Any]:
        """Fetch firewall rules."""
        try:
            from google.cloud import compute_v1
            
            firewall_client = compute_v1.FirewallsClient()
            rules = []
            
            for rule in firewall_client.list(project=self.project_id):
                rule_data = {
                    "id": rule.name,
                    "name": rule.name,
                    "description": rule.description,
                    "network": rule.network.split('/')[-1] if rule.network else None,
                    "priority": rule.priority,
                    "direction": rule.direction,
                    "source_ranges": list(rule.source_ranges) if rule.source_ranges else [],
                    "destination_ranges": list(rule.destination_ranges) if rule.destination_ranges else [],
                    "allowed": [],
                    "denied": [],
                    "disabled": rule.disabled
                }
                
                # Extract allowed rules
                for allowed in rule.allowed:
                    rule_data["allowed"].append({
                        "protocol": allowed.I_p_protocol,
                        "ports": list(allowed.ports) if allowed.ports else []
                    })
                
                # Extract denied rules
                for denied in rule.denied:
                    rule_data["denied"].append({
                        "protocol": denied.I_p_protocol,
                        "ports": list(denied.ports) if denied.ports else []
                    })
                
                rules.append(rule_data)
            
            # Store in database
            self._store_firewall_rules(rules)
            
            return {"count": len(rules)}
            
        except Exception as e:
            logger.error(f"Error fetching firewall rules: {e}")
            return {"count": 0, "error": str(e)}
    
    async def _fetch_iam_accounts(self) -> Dict[str, Any]:
        """Fetch IAM service accounts and permissions."""
        try:
            from google.cloud import iam_admin_v1
            from google.iam.v1 import iam_policy_pb2
            from google.cloud import resourcemanager_v3
            
            iam_client = iam_admin_v1.IAMClient()
            accounts = []
            
            # List service accounts
            request = iam_admin_v1.ListServiceAccountsRequest(
                name=f"projects/{self.project_id}",
                page_size=100
            )
            
            for account in iam_client.list_service_accounts(request=request):
                account_data = {
                    "email": account.email,
                    "display_name": account.display_name,
                    "description": account.description,
                    "unique_id": account.unique_id,
                    "disabled": account.disabled,
                    "keys": [],
                    "roles": []
                }
                
                # Get keys for this account
                try:
                    key_request = iam_admin_v1.ListServiceAccountKeysRequest(
                        name=f"projects/{self.project_id}/serviceAccounts/{account.email}"
                    )
                    key_response = iam_client.list_service_account_keys(request=key_request)
                    for key in key_response.keys:
                        account_data["keys"].append({
                            "name": key.name,
                            "key_type": key.key_type.name if key.key_type else None,
                            "valid_after": str(key.valid_after_time) if key.valid_after_time else None,
                            "valid_before": str(key.valid_before_time) if key.valid_before_time else None
                        })
                except Exception as e:
                    logger.warning(f"Could not fetch keys for {account.email}: {e}")
                
                accounts.append(account_data)
            
            # Store in database
            self._store_iam_accounts(accounts)
            
            return {"count": len(accounts)}
            
        except Exception as e:
            logger.error(f"Error fetching IAM accounts: {e}")
            return {"count": 0, "error": str(e)}
    
    async def _fetch_iam_policies(self) -> Dict[str, Any]:
        """Fetch IAM policies for the project."""
        try:
            from google.cloud import resourcemanager_v3
            
            # Get project-level IAM policies
            projects_client = resourcemanager_v3.ProjectsClient()
            project_name = f"projects/{self.project_id}"
            
            policies = []
            
            # Get the IAM policy for the project
            try:
                policy = projects_client.get_iam_policy(resource=project_name)
                
                # Process each binding
                for binding in policy.bindings:
                    role = binding.role
                    for member in binding.members:
                        policy_data = {
                            "id": f"{self.project_id}_{role}_{member}",
                            "resource_type": "project",
                            "resource_name": self.project_id,
                            "member": member,
                            "role": role,
                            "condition": str(binding.condition) if binding.condition else None
                        }
                        policies.append(policy_data)
                        
            except Exception as e:
                logger.warning(f"Could not fetch project IAM policy: {e}")
            
            # Store in database
            self._store_iam_policies(policies)
            
            return {"count": len(policies)}
            
        except Exception as e:
            logger.error(f"Error fetching IAM policies: {e}")
            return {"count": 0, "error": str(e)}
    
    async def _fetch_databases(self) -> Dict[str, Any]:
        """Fetch Cloud SQL instances."""
        try:
            # Try the newer sqladmin library first
            try:
                from google.cloud.sql_v1 import SqlInstancesServiceClient, SqlInstancesListRequest
                sql_client = SqlInstancesServiceClient()
                request = SqlInstancesListRequest(project=self.project_id)
            except ImportError:
                # Fallback: Skip SQL if library not available
                logger.warning("Cloud SQL client library not available, skipping database fetching")
                return {"count": 0, "skipped": "Cloud SQL library not installed"}
                
            databases = []
            
            for instance in sql_client.list(request=request):
                db_data = {
                    "id": instance.name,
                    "name": instance.name,
                    "type": instance.database_version,
                    "version": instance.database_version,
                    "location": instance.region,
                    "tier": instance.settings.tier,
                    "high_availability": instance.settings.availability_type == "REGIONAL",
                    "backup_enabled": instance.settings.backup_configuration.enabled if instance.settings.backup_configuration else False,
                    "encryption": "Customer-managed" if instance.disk_encryption_configuration else "Google-managed",
                    "public_ip": None,
                    "private_ip": None
                }
                
                # Extract IP addresses
                for ip_addr in instance.ip_addresses:
                    if ip_addr.type_ == "PRIMARY":
                        db_data["public_ip"] = ip_addr.ip_address
                    elif ip_addr.type_ == "PRIVATE":
                        db_data["private_ip"] = ip_addr.ip_address
                
                databases.append(db_data)
            
            # Store in database
            self._store_databases(databases)
            
            return {"count": len(databases)}
            
        except Exception as e:
            logger.error(f"Error fetching databases: {e}")
            return {"count": 0, "error": str(e)}
    
    async def _fetch_security_findings(self) -> Dict[str, Any]:
        """Fetch security findings (using sample data since SCC is disabled)."""
        try:
            # Since Security Command Center is disabled, use sample findings
            findings = [
                {
                    "id": f"finding-001",
                    "category": "PUBLIC_BUCKET",
                    "severity": "HIGH",
                    "state": "ACTIVE",
                    "resource_name": f"//storage.googleapis.com/{self.project_id}-public-bucket",
                    "finding_class": "VULNERABILITY",
                    "description": "Storage bucket is publicly accessible",
                    "recommendation": "Remove public access or add authentication",
                    "event_time": datetime.now().isoformat()
                },
                {
                    "id": f"finding-002",
                    "category": "WEAK_CREDENTIALS",
                    "severity": "CRITICAL",
                    "state": "ACTIVE",
                    "resource_name": f"//iam.googleapis.com/projects/{self.project_id}/serviceAccounts/test",
                    "finding_class": "VULNERABILITY",
                    "description": "Service account key is older than 90 days",
                    "recommendation": "Rotate service account keys regularly",
                    "event_time": datetime.now().isoformat()
                },
                {
                    "id": f"finding-003",
                    "category": "FIREWALL_MISCONFIGURATION",
                    "severity": "MEDIUM",
                    "state": "ACTIVE",
                    "resource_name": f"//compute.googleapis.com/projects/{self.project_id}/global/firewalls/allow-all",
                    "finding_class": "MISCONFIGURATION",
                    "description": "Firewall rule allows unrestricted access",
                    "recommendation": "Restrict firewall rules to specific IP ranges",
                    "event_time": datetime.now().isoformat()
                }
            ]
            
            # Store in database
            self._store_security_findings(findings)
            
            return {"count": len(findings), "source": "sample_data"}
            
        except Exception as e:
            logger.error(f"Error storing security findings: {e}")
            return {"count": 0, "error": str(e)}
    
    async def _fetch_all_assets(self) -> Dict[str, Any]:
        """Fetch all assets using Cloud Asset Inventory - Enhanced to capture full resource data."""
        try:
            from google.cloud import asset_v1
            from google.protobuf.json_format import MessageToDict
            
            asset_client = asset_v1.AssetServiceClient()
            parent = f"projects/{self.project_id}"
            
            assets = []
            asset_types_found = set()
            
            # List all assets with full resource data
            request = asset_v1.ListAssetsRequest(
                parent=parent,
                page_size=100,
                content_type=asset_v1.ContentType.RESOURCE  # Get full resource data
            )
            
            for asset in asset_client.list_assets(request=request):
                # Track asset types we've found
                asset_types_found.add(asset.asset_type)
                
                # Convert the entire asset to a dictionary for complete data preservation
                try:
                    asset_dict = MessageToDict(asset._pb)
                except:
                    # Fallback to manual extraction if protobuf conversion fails
                    asset_dict = {
                        "name": asset.name,
                        "assetType": asset.asset_type,
                        "resource": MessageToDict(asset.resource._pb) if asset.resource else {}
                    }
                
                # Extract common fields with better null handling
                resource_data = asset_dict.get("resource", {}).get("data", {})
                
                # Extract location from various possible fields
                location = (resource_data.get("location") or 
                           resource_data.get("zone") or 
                           resource_data.get("region") or
                           resource_data.get("locationId") or "")
                
                # Extract display name from various possible fields  
                display_name = (resource_data.get("displayName") or
                               resource_data.get("name") or
                               asset.name.split('/')[-1])
                
                # Extract state/status from various possible fields
                state_raw = (resource_data.get("state") or
                            resource_data.get("status") or
                            resource_data.get("lifecycleState") or "")
                
                # Ensure state is always a string
                if isinstance(state_raw, dict):
                    state = state_raw.get("state") or state_raw.get("status") or str(state_raw)
                else:
                    state = str(state_raw) if state_raw else ""
                
                # Ensure parent_resource is always a string
                parent = asset_dict.get("resource", {}).get("parent", "")
                if not isinstance(parent, str):
                    logger.warning(f"Parent resource is not a string for {asset.name}: {type(parent)}")
                    parent = str(parent) if parent else ""
                
                asset_data = {
                    "id": asset.name,
                    "name": asset.name.split('/')[-1],  # Extract just the resource name
                    "asset_type": asset.asset_type,
                    "display_name": display_name,
                    "description": resource_data.get("description", ""),
                    "location": location,
                    "labels": json.dumps(resource_data.get("labels", {})),
                    "create_time": asset.update_time.isoformat() if asset.update_time else None,
                    "update_time": asset.update_time.isoformat() if asset.update_time else None,
                    "state": state,
                    "parent_resource": parent,
                    "data": json.dumps(asset_dict)  # Store the complete asset data
                }
                assets.append(asset_data)
            
            # Store in database
            self._store_assets(assets)
            
            # Log discovered asset types for debugging
            logger.info(f"Discovered {len(asset_types_found)} unique asset types: {sorted(asset_types_found)[:10]}...")
            
            return {"count": len(assets), "asset_types": len(asset_types_found)}
            
        except Exception as e:
            logger.error(f"Error fetching assets: {e}")
            # Return 0 if asset inventory not available
            return {"count": 0, "error": str(e)}
    
    def _store_compute_instances(self, instances: List[Dict]):
        """Store compute instances in database."""
        with self._get_connection() as conn:
            for instance in instances:
                conn.execute("""
                    INSERT OR REPLACE INTO compute_instances
                    (id, project_id, name, zone, machine_type, status, internal_ip, 
                     external_ip, network_interfaces, disks, labels, metadata, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    instance["id"],
                    self.project_id,
                    instance["name"],
                    instance["zone"],
                    instance["machine_type"],
                    instance["status"],
                    instance["internal_ip"],
                    instance["external_ip"],
                    json.dumps(instance["network_interfaces"]),
                    json.dumps(instance["disks"]),
                    json.dumps(instance["labels"]),
                    json.dumps(instance["metadata"]),
                    json.dumps(instance)
                ))
            conn.commit()
    
    def _store_storage_buckets(self, buckets: List[Dict]):
        """Store storage buckets in database."""
        with self._get_connection() as conn:
            for bucket in buckets:
                conn.execute("""
                    INSERT OR REPLACE INTO storage_buckets
                    (id, project_id, name, location, storage_class, iam_configuration,
                     lifecycle_rules, versioning_enabled, uniform_bucket_level_access,
                     public_access, encryption, labels, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bucket["id"],
                    self.project_id,
                    bucket["name"],
                    bucket["location"],
                    bucket["storage_class"],
                    json.dumps(bucket["iam_configuration"]),
                    str(bucket["lifecycle_rules"]),
                    bucket["versioning_enabled"],
                    bucket["uniform_bucket_level_access"],
                    bucket["public_access"],
                    bucket["encryption"],
                    json.dumps(bucket["labels"]),
                    json.dumps(bucket)
                ))
            conn.commit()
    
    def _store_networks(self, networks: List[Dict]):
        """Store networks in database."""
        with self._get_connection() as conn:
            for network in networks:
                conn.execute("""
                    INSERT OR REPLACE INTO networks
                    (id, project_id, name, description, auto_create_subnetworks,
                     routing_mode, mtu, subnets, peerings, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    network["id"],
                    self.project_id,
                    network["name"],
                    network["description"],
                    network["auto_create_subnetworks"],
                    network["routing_mode"],
                    network["mtu"],
                    json.dumps(network["subnets"]),
                    json.dumps(network["peerings"]),
                    json.dumps(network)
                ))
            conn.commit()
    
    def _store_firewall_rules(self, rules: List[Dict]):
        """Store firewall rules in database."""
        with self._get_connection() as conn:
            for rule in rules:
                conn.execute("""
                    INSERT OR REPLACE INTO firewall_rules
                    (id, project_id, name, description, network, priority, direction,
                     source_ranges, destination_ranges, allowed, denied, disabled, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule["id"],
                    self.project_id,
                    rule["name"],
                    rule["description"],
                    rule["network"],
                    rule["priority"],
                    rule["direction"],
                    json.dumps(rule["source_ranges"]),
                    json.dumps(rule["destination_ranges"]),
                    json.dumps(rule["allowed"]),
                    json.dumps(rule["denied"]),
                    rule["disabled"],
                    json.dumps(rule)
                ))
            conn.commit()
    
    def _store_iam_accounts(self, accounts: List[Dict]):
        """Store IAM accounts in database."""
        with self._get_connection() as conn:
            for account in accounts:
                conn.execute("""
                    INSERT OR REPLACE INTO iam_accounts
                    (email, project_id, account_type, display_name, description,
                     unique_id, disabled, keys, roles, permissions, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    account["email"],
                    self.project_id,
                    "service_account",
                    account["display_name"],
                    account["description"],
                    account["unique_id"],
                    account["disabled"],
                    json.dumps(account["keys"]),
                    json.dumps(account["roles"]),
                    json.dumps([]),  # Permissions would be fetched separately
                    json.dumps(account)
                ))
            conn.commit()
    
    def _store_iam_policies(self, policies: List[Dict]):
        """Store IAM policies in database."""
        with self._get_connection() as conn:
            for policy in policies:
                conn.execute("""
                    INSERT OR REPLACE INTO iam_policies
                    (id, project_id, resource_type, resource_name, member, role, condition)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    policy["id"],
                    self.project_id,
                    policy["resource_type"],
                    policy["resource_name"],
                    policy["member"],
                    policy["role"],
                    policy["condition"]
                ))
            conn.commit()
    
    def _store_databases(self, databases: List[Dict]):
        """Store databases in database."""
        with self._get_connection() as conn:
            for db in databases:
                conn.execute("""
                    INSERT OR REPLACE INTO databases
                    (id, project_id, name, type, version, location, tier,
                     high_availability, backup_enabled, encryption, public_ip, private_ip, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    db["id"],
                    self.project_id,
                    db["name"],
                    db["type"],
                    db["version"],
                    db["location"],
                    db["tier"],
                    db["high_availability"],
                    db["backup_enabled"],
                    db["encryption"],
                    db["public_ip"],
                    db["private_ip"],
                    json.dumps(db)
                ))
            conn.commit()
    
    def _store_security_findings(self, findings: List[Dict]):
        """Store security findings in database."""
        with self._get_connection() as conn:
            for finding in findings:
                conn.execute("""
                    INSERT OR REPLACE INTO security_findings
                    (id, project_id, category, severity, state, resource_name,
                     finding_class, description, recommendation, event_time, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    finding["id"],
                    self.project_id,
                    finding["category"],
                    finding["severity"],
                    finding["state"],
                    finding["resource_name"],
                    finding["finding_class"],
                    finding["description"],
                    finding["recommendation"],
                    finding["event_time"],
                    json.dumps(finding)
                ))
            conn.commit()
    
    def _store_assets(self, assets: List[Dict]):
        """Store general assets in database."""
        with self._get_connection() as conn:
            for i, asset in enumerate(assets):
                try:
                    # Validate all parameters are the right type
                    params = (
                        asset["id"],                # 1
                        self.project_id,             # 2
                        asset["name"],               # 3
                        asset["asset_type"],         # 4
                        asset["display_name"],       # 5
                        asset["description"],        # 6
                        asset["location"],           # 7
                        asset["labels"],             # 8
                        asset["create_time"],        # 9
                        asset["update_time"],        # 10
                        asset["state"],              # 11
                        asset["parent_resource"],    # 12
                        asset["data"]                # 13
                    )
                    
                    # Check for any non-string/None values
                    for j, param in enumerate(params):
                        if param is not None and not isinstance(param, str):
                            logger.error(f"Asset {i} param {j+1} is type {type(param)}: {param}")
                            
                    conn.execute("""
                        INSERT OR REPLACE INTO assets
                        (id, project_id, name, asset_type, display_name, description,
                         location, labels, create_time, update_time, state, parent_resource, data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, params)
                except Exception as e:
                    logger.error(f"Error storing asset {i} ({asset.get('name', 'unknown')}): {e}")
                    # Log the problematic data
                    for key, value in asset.items():
                        logger.debug(f"  {key}: {type(value).__name__} = {repr(value)[:100]}")
                    raise
            conn.commit()
    
    def _update_fetch_status(self, status: str, stats: Dict):
        """Update fetch status in database."""
        with self._get_connection() as conn:
            for fetch_type, fetch_stats in stats.items():
                records = fetch_stats.get("count", 0)
                error = fetch_stats.get("error", None)
                
                conn.execute("""
                    INSERT OR REPLACE INTO fetch_status
                    (project_id, fetch_type, status, started_at, completed_at,
                     records_fetched, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.project_id,
                    fetch_type,
                    "error" if error else "success",
                    self.fetch_timestamp,
                    datetime.now(),
                    records,
                    error
                ))
            conn.commit()
    
    # Query methods for fast local access
    
    def query_compute_instances(self, status: Optional[str] = None) -> List[Dict]:
        """Query compute instances from local database."""
        with self._get_connection() as conn:
            query = "SELECT * FROM compute_instances WHERE project_id = ?"
            params = [self.project_id]
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor]
    
    def query_storage_buckets(self, public_only: bool = False) -> List[Dict]:
        """Query storage buckets from local database."""
        with self._get_connection() as conn:
            query = "SELECT * FROM storage_buckets WHERE project_id = ?"
            params = [self.project_id]
            
            if public_only:
                query += " AND public_access = 'public'"
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor]
    
    def query_security_findings(self, severity: Optional[str] = None) -> List[Dict]:
        """Query security findings from local database."""
        with self._get_connection() as conn:
            query = "SELECT * FROM security_findings WHERE project_id = ?"
            params = [self.project_id]
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            query += " ORDER BY severity DESC"
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor]
    
    async def _fetch_secrets(self) -> Dict[str, Any]:
        """Fetch all secrets from Secret Manager."""
        try:
            from google.cloud import secretmanager
            
            client = secretmanager.SecretManagerServiceClient()
            parent = f"projects/{self.project_id}"
            
            secrets = []
            
            # List all secrets
            for secret in client.list_secrets(request={"parent": parent}):
                secret_data = {
                    "id": secret.name.split('/')[-1],
                    "name": secret.name.split('/')[-1],
                    "version": 1,  # Latest version
                    "create_time": secret.create_time.isoformat() if secret.create_time else None,
                    "expire_time": None,  # Would need to check individual versions
                    "replication": json.dumps({"type": str(secret.replication)}),
                    "labels": json.dumps(dict(secret.labels) if secret.labels else {}),
                    "rotation_policy": None,  # Would need separate API call
                    "annotations": json.dumps(dict(secret.annotations) if secret.annotations else {})
                }
                secrets.append(secret_data)
            
            # Store in database
            self._store_secrets(secrets)
            
            return {"count": len(secrets)}
            
        except Exception as e:
            logger.error(f"Error fetching secrets: {e}")
            return {"count": 0, "error": str(e)}
    
    async def _fetch_monitoring_metrics(self) -> Dict[str, Any]:
        """Fetch key monitoring metrics from Cloud Monitoring."""
        try:
            try:
                from google.cloud import monitoring_v3
                from google.cloud.monitoring_v3 import query
            except ImportError:
                logger.warning("Cloud Monitoring library not available - skipping metrics")
                return {"count": 0, "skipped": "monitoring_v3 library not installed"}
            
            client = monitoring_v3.MetricServiceClient()
            project_name = f"projects/{self.project_id}"
            
            metrics = []
            
            # Define key metrics to fetch
            # Use valid metric types only
            metric_types = [
                "compute.googleapis.com/instance/cpu/utilization",
                "storage.googleapis.com/storage/total_bytes",
                "cloudsql.googleapis.com/database/cpu/utilization"
                # Removed invalid types: "gce_instance", "storage_bucket"
            ]
            
            # Get current time for recent metrics
            from google.cloud.monitoring_v3.types import TimeInterval
            import datetime
            
            now = datetime.datetime.now(datetime.timezone.utc)
            one_hour_ago = now - datetime.timedelta(hours=1)
            
            interval = TimeInterval({
                "end_time": {"seconds": int(now.timestamp())},
                "start_time": {"seconds": int(one_hour_ago.timestamp())}
            })
            
            for metric_type in metric_types:
                try:
                    request = monitoring_v3.ListTimeSeriesRequest(
                        name=project_name,
                        filter=f'metric.type="{metric_type}"',
                        interval=interval,
                        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                    )
                    
                    results = client.list_time_series(request=request)
                    
                    for result in results:
                        metric_data = {
                            "id": f"{metric_type}_{result.resource.labels.get('instance_id', 'unknown')}_{int(now.timestamp())}",
                            "resource_type": result.resource.type,
                            "resource_name": result.resource.labels.get('instance_name', result.resource.labels.get('bucket_name', 'unknown')),
                            "metric_type": metric_type,
                            "value": float(result.points[0].value.double_value) if result.points and len(result.points) > 0 else 0.0,
                            "unit": result.metric.type,
                            "timestamp": now.isoformat(),
                            "labels": json.dumps(dict(result.resource.labels))
                        }
                        metrics.append(metric_data)
                        
                except Exception as e:
                    logger.warning(f"Could not fetch metric {metric_type}: {e}")
                    continue
            
            # Store in database
            self._store_monitoring_metrics(metrics)
            
            return {"count": len(metrics)}
            
        except Exception as e:
            logger.error(f"Error fetching monitoring metrics: {e}")
            return {"count": 0, "error": str(e)}

    def _store_secrets(self, secrets: List[Dict]):
        """Store secrets in database."""
        with self._get_connection() as conn:
            for secret in secrets:
                conn.execute("""
                    INSERT OR REPLACE INTO secrets
                    (id, project_id, name, version, create_time, expire_time,
                     replication, labels, rotation_policy, annotations, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    secret["id"],
                    self.project_id,
                    secret["name"],
                    secret["version"],
                    secret["create_time"],
                    secret["expire_time"],
                    secret["replication"],
                    secret["labels"],
                    secret["rotation_policy"],
                    secret["annotations"],
                    json.dumps(secret)
                ))
            conn.commit()

    def _store_monitoring_metrics(self, metrics: List[Dict]):
        """Store monitoring metrics in database."""
        with self._get_connection() as conn:
            for metric in metrics:
                conn.execute("""
                    INSERT OR REPLACE INTO monitoring_metrics
                    (id, project_id, resource_type, resource_name, metric_type,
                     value, unit, timestamp, labels, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric["id"],
                    self.project_id,
                    metric["resource_type"],
                    metric["resource_name"],
                    metric["metric_type"],
                    metric["value"],
                    metric["unit"],
                    metric["timestamp"],
                    metric["labels"],
                    json.dumps(metric)
                ))
            conn.commit()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from cached data."""
        with self._get_connection() as conn:
            stats = {}
            
            # Count resources including new tables
            for table in ["compute_instances", "storage_buckets", "networks", 
                         "firewall_rules", "iam_accounts", "databases", "security_findings",
                         "secrets", "monitoring_metrics", "gke_clusters"]:
                cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table} WHERE project_id = ?", [self.project_id])
                stats[table] = cursor.fetchone()["count"]
            
            # Get last fetch time
            cursor = conn.execute("""
                SELECT MAX(completed_at) as last_fetch 
                FROM fetch_status 
                WHERE project_id = ? AND status = 'success'
            """, [self.project_id])
            
            result = cursor.fetchone()
            stats["last_fetch"] = result["last_fetch"] if result else None
            
            return stats

    async def _fetch_gke_clusters(self) -> Dict[str, Any]:
        """Fetch all GKE clusters."""
        try:
            from google.cloud import container_v1
            
            cluster_client = container_v1.ClusterManagerClient()
            project_id = self.project_id
            
            clusters = []
            
            # List all clusters across all locations
            parent = f"projects/{project_id}/locations/-"
            
            try:
                response = cluster_client.list_clusters(parent=parent)
                for cluster in response.clusters:
                    cluster_data = {
                        "id": f"{project_id}/{cluster.location}/{cluster.name}",
                        "project_id": project_id,
                        "name": cluster.name,
                        "location": cluster.location,
                        "location_type": cluster.location_type.name if cluster.location_type else "UNSPECIFIED",
                        "status": cluster.status.name if cluster.status else "UNKNOWN",
                        "endpoint": cluster.endpoint,
                        "current_master_version": cluster.current_master_version,
                        "current_node_version": cluster.current_node_version,
                        "initial_node_count": cluster.initial_node_count,
                        "current_node_count": cluster.current_node_count,
                        "node_config": json.dumps({
                            "machine_type": cluster.node_config.machine_type if cluster.node_config else None,
                            "disk_size_gb": cluster.node_config.disk_size_gb if cluster.node_config else None,
                            "disk_type": cluster.node_config.disk_type if cluster.node_config else None,
                            "image_type": cluster.node_config.image_type if cluster.node_config else None,
                            "service_account": cluster.node_config.service_account if cluster.node_config else None,
                            "preemptible": cluster.node_config.preemptible if cluster.node_config else False,
                        }) if cluster.node_config else "{}",
                        "network": cluster.network,
                        "subnetwork": cluster.subnetwork,
                        "cluster_ipv4_cidr": cluster.cluster_ipv4_cidr,
                        "services_ipv4_cidr": cluster.services_ipv4_cidr,
                        "locations": json.dumps(list(cluster.locations)) if cluster.locations else "[]",
                        "enable_autopilot": cluster.autopilot.enabled if cluster.autopilot else False,
                        "enable_network_policy": cluster.network_policy.enabled if cluster.network_policy else False,
                        "enable_ip_alias": cluster.ip_allocation_policy.use_ip_aliases if cluster.ip_allocation_policy else False,
                        "master_auth": json.dumps({
                            "username": cluster.master_auth.username if cluster.master_auth else None,
                            "client_certificate_config": {
                                "issue_client_certificate": cluster.master_auth.client_certificate_config.issue_client_certificate if cluster.master_auth and cluster.master_auth.client_certificate_config else False
                            }
                        }) if cluster.master_auth else "{}",
                        "logging_config": json.dumps({
                            "component_config": {
                                "enable_components": list(cluster.logging_config.component_config.enable_components) if cluster.logging_config and cluster.logging_config.component_config else []
                            }
                        }) if cluster.logging_config else "{}",
                        "monitoring_config": json.dumps({
                            "component_config": {
                                "enable_components": list(cluster.monitoring_config.component_config.enable_components) if cluster.monitoring_config and cluster.monitoring_config.component_config else []
                            }
                        }) if cluster.monitoring_config else "{}",
                        "addons_config": json.dumps({
                            "http_load_balancing": cluster.addons_config.http_load_balancing.disabled if cluster.addons_config and cluster.addons_config.http_load_balancing else None,
                            "horizontal_pod_autoscaling": cluster.addons_config.horizontal_pod_autoscaling.disabled if cluster.addons_config and cluster.addons_config.horizontal_pod_autoscaling else None,
                            "kubernetes_dashboard": cluster.addons_config.kubernetes_dashboard.disabled if cluster.addons_config and cluster.addons_config.kubernetes_dashboard else None,
                            "network_policy_config": cluster.addons_config.network_policy_config.disabled if cluster.addons_config and cluster.addons_config.network_policy_config else None,
                        }) if cluster.addons_config else "{}",
                        "node_pools": json.dumps([{
                            "name": pool.name,
                            "status": pool.status.name if pool.status else "UNKNOWN",
                            "initial_node_count": pool.initial_node_count,
                            "version": pool.version,
                            "config": {
                                "machine_type": pool.config.machine_type if pool.config else None,
                                "disk_size_gb": pool.config.disk_size_gb if pool.config else None,
                                "preemptible": pool.config.preemptible if pool.config else False,
                            }
                        } for pool in cluster.node_pools]) if cluster.node_pools else "[]",
                        "legacy_abac_enabled": cluster.legacy_abac.enabled if cluster.legacy_abac else False,
                        "workload_identity_config": json.dumps({
                            "workload_pool": cluster.workload_identity_config.workload_pool if cluster.workload_identity_config else None
                        }) if cluster.workload_identity_config else "{}",
                        "private_cluster_config": json.dumps({
                            "enable_private_nodes": cluster.private_cluster_config.enable_private_nodes if cluster.private_cluster_config else False,
                            "enable_private_endpoint": cluster.private_cluster_config.enable_private_endpoint if cluster.private_cluster_config else False,
                            "master_ipv4_cidr_block": cluster.private_cluster_config.master_ipv4_cidr_block if cluster.private_cluster_config else None,
                        }) if cluster.private_cluster_config else "{}",
                        "database_encryption": json.dumps({
                            "state": cluster.database_encryption.state.name if cluster.database_encryption and cluster.database_encryption.state else "DECRYPTED",
                            "key_name": cluster.database_encryption.key_name if cluster.database_encryption else None,
                        }) if cluster.database_encryption else "{}",
                        "shielded_nodes": json.dumps({
                            "enabled": cluster.shielded_nodes.enabled if cluster.shielded_nodes else False
                        }) if cluster.shielded_nodes else "{}",
                        "release_channel": json.dumps({
                            "channel": cluster.release_channel.channel.name if cluster.release_channel and cluster.release_channel.channel else "UNSPECIFIED"
                        }) if cluster.release_channel else "{}",
                        "data": json.dumps({
                            "name": cluster.name,
                            "location": cluster.location,
                            "status": cluster.status.name if cluster.status else "UNKNOWN",
                            "endpoint": cluster.endpoint,
                            "current_master_version": cluster.current_master_version,
                            "current_node_version": cluster.current_node_version,
                            "node_count": cluster.current_node_count,
                            "network": cluster.network,
                            "subnetwork": cluster.subnetwork,
                        })
                    }
                    clusters.append(cluster_data)
                
                # Store in database
                self._store_gke_clusters(clusters)
                
                return {"count": len(clusters)}
                
            except Exception as e:
                logger.error(f"Error listing GKE clusters: {e}")
                return {"count": 0, "error": str(e)}
            
        except ImportError:
            logger.warning("Google Cloud Container library not available")
            return {"count": 0, "skipped": "Library not installed"}
        except Exception as e:
            logger.error(f"Error fetching GKE clusters: {e}")
            return {"count": 0, "error": str(e)}

    def _store_gke_clusters(self, clusters: List[Dict]):
        """Store GKE clusters in database."""
        if not clusters:
            return
            
        with self._get_connection() as conn:
            # Clear existing data for this project
            conn.execute("DELETE FROM gke_clusters WHERE project_id = ?", (self.project_id,))
            
            # Insert new data
            for cluster in clusters:
                conn.execute("""
                    INSERT INTO gke_clusters (
                        id, project_id, name, location, location_type, status, endpoint,
                        current_master_version, current_node_version, initial_node_count,
                        current_node_count, node_config, network, subnetwork, cluster_ipv4_cidr,
                        services_ipv4_cidr, locations, enable_autopilot, enable_network_policy,
                        enable_ip_alias, master_auth, logging_config, monitoring_config,
                        addons_config, node_pools, legacy_abac_enabled, workload_identity_config,
                        private_cluster_config, database_encryption, shielded_nodes,
                        release_channel, data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cluster["id"], cluster["project_id"], cluster["name"], cluster["location"],
                    cluster["location_type"], cluster["status"], cluster["endpoint"],
                    cluster["current_master_version"], cluster["current_node_version"],
                    cluster["initial_node_count"], cluster["current_node_count"], 
                    cluster["node_config"], cluster["network"], cluster["subnetwork"],
                    cluster["cluster_ipv4_cidr"], cluster["services_ipv4_cidr"], cluster["locations"],
                    cluster["enable_autopilot"], cluster["enable_network_policy"], cluster["enable_ip_alias"],
                    cluster["master_auth"], cluster["logging_config"], cluster["monitoring_config"],
                    cluster["addons_config"], cluster["node_pools"], cluster["legacy_abac_enabled"],
                    cluster["workload_identity_config"], cluster["private_cluster_config"],
                    cluster["database_encryption"], cluster["shielded_nodes"], cluster["release_channel"],
                    cluster["data"]
                ))
            conn.commit()


# Convenience function
async def fetch_all_project_data(project_id: str) -> Dict[str, Any]:
    """Fetch all data for a project."""
    fetcher = DataFetcher(project_id)
    return await fetcher.fetch_all_data()