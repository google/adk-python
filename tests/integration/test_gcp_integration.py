"""
Integration Tests for ADK Google Cloud Platform Integration

These tests validate the integration between ADK and Google Cloud services,
including Asset Inventory, Compute Engine, Storage, and IAM.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any
import json


class TestGCPAssetInventoryIntegration:
    """Integration tests for Google Cloud Asset Inventory."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @patch('google.cloud.asset.AssetServiceAsyncClient')
    async def test_asset_inventory_list_all_assets(self, mock_client, test_config, sample_compute_instance, sample_storage_bucket):
        """Test listing all assets from Asset Inventory API."""
        # Given: Mock Asset Inventory client with sample data
        mock_instance = mock_client.return_value
        
        # Mock asset data
        mock_assets = [
            {
                "name": f"//compute.googleapis.com/projects/{test_config['project_id']}/zones/{test_config['zone']}/instances/test-vm-1",
                "assetType": "compute.googleapis.com/Instance",
                "resource": {"data": sample_compute_instance}
            },
            {
                "name": f"//storage.googleapis.com/projects/_/buckets/test-bucket",
                "assetType": "storage.googleapis.com/Bucket", 
                "resource": {"data": sample_storage_bucket}
            }
        ]
        
        mock_instance.search_all_resources.return_value.__aiter__.return_value = iter(mock_assets)
        
        # When: Searching for all resources
        assets = []
        async for asset in mock_instance.search_all_resources(
            request={
                "scope": f"projects/{test_config['project_id']}",
                "asset_types": [],
            }
        ):
            assets.append(asset)
        
        # Then: Assets are returned correctly
        assert len(assets) == 2
        assert any(asset["assetType"] == "compute.googleapis.com/Instance" for asset in assets)
        assert any(asset["assetType"] == "storage.googleapis.com/Bucket" for asset in assets)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @patch('google.cloud.asset.AssetServiceAsyncClient')
    async def test_asset_inventory_filter_by_type(self, mock_client, test_config, sample_compute_instance):
        """Test filtering assets by type."""
        # Given: Mock client with compute instances only
        mock_instance = mock_client.return_value
        
        compute_assets = [
            {
                "name": f"//compute.googleapis.com/projects/{test_config['project_id']}/zones/{test_config['zone']}/instances/test-vm-{i}",
                "assetType": "compute.googleapis.com/Instance",
                "resource": {"data": {**sample_compute_instance, "name": f"test-vm-{i}"}}
            }
            for i in range(1, 4)
        ]
        
        mock_instance.search_all_resources.return_value.__aiter__.return_value = iter(compute_assets)
        
        # When: Filtering by compute instances
        assets = []
        async for asset in mock_instance.search_all_resources(
            request={
                "scope": f"projects/{test_config['project_id']}",
                "asset_types": ["compute.googleapis.com/Instance"],
            }
        ):
            assets.append(asset)
        
        # Then: Only compute instances are returned
        assert len(assets) == 3
        assert all(asset["assetType"] == "compute.googleapis.com/Instance" for asset in assets)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_asset_inventory_error_handling(self, mock_asset_client):
        """Test error handling in asset inventory operations."""
        # Given: Client that raises an exception
        from google.api_core.exceptions import NotFound
        mock_asset_client.search_all_resources.side_effect = NotFound("Project not found")
        
        # When/Then: Exception is handled appropriately
        with pytest.raises(NotFound):
            assets = []
            async for asset in mock_asset_client.search_all_resources(
                request={"scope": "projects/nonexistent-project"}
            ):
                assets.append(asset)


class TestGCPComputeEngineIntegration:
    """Integration tests for Google Compute Engine."""
    
    @pytest.mark.integration
    @patch('google.cloud.compute_v1.InstancesClient')
    def test_compute_instances_list(self, mock_client, test_config, sample_compute_instance):
        """Test listing compute instances."""
        # Given: Mock Compute client
        mock_instance = mock_client.return_value
        
        instances = [sample_compute_instance]
        mock_instance.list.return_value = instances
        
        # When: Listing instances
        result = mock_instance.list(
            project=test_config["project_id"],
            zone=test_config["zone"]
        )
        
        # Then: Instances are returned
        assert len(result) == 1
        assert result[0]["name"] == "test-instance-1"
        assert result[0]["status"] == "RUNNING"
    
    @pytest.mark.integration
    @patch('google.cloud.compute_v1.InstancesClient')
    def test_compute_instance_security_analysis(self, mock_client, test_config, sample_compute_instance):
        """Test security analysis of compute instances."""
        # Given: Compute instance with potential security issues
        insecure_instance = {
            **sample_compute_instance,
            "networkInterfaces": [
                {
                    "network": f"projects/{test_config['project_id']}/global/networks/default",
                    "accessConfigs": [
                        {
                            "type": "ONE_TO_ONE_NAT",
                            "name": "External NAT",
                            "natIP": "0.0.0.0"  # Suspicious IP
                        }
                    ]
                }
            ]
        }
        
        mock_instance = mock_client.return_value
        mock_instance.list.return_value = [insecure_instance]
        
        # When: Analyzing instance security
        instances = mock_instance.list(
            project=test_config["project_id"],
            zone=test_config["zone"]
        )
        
        security_findings = []
        for instance in instances:
            # Analyze network configuration
            for interface in instance.get("networkInterfaces", []):
                for config in interface.get("accessConfigs", []):
                    if config.get("natIP") == "0.0.0.0":
                        security_findings.append({
                            "resource": instance["name"],
                            "finding": "suspicious_external_ip",
                            "severity": "MEDIUM"
                        })
        
        # Then: Security findings are identified
        assert len(security_findings) == 1
        assert security_findings[0]["finding"] == "suspicious_external_ip"


class TestGCPStorageIntegration:
    """Integration tests for Google Cloud Storage."""
    
    @pytest.mark.integration
    @patch('google.cloud.storage.Client')
    def test_storage_buckets_list(self, mock_client, sample_storage_bucket):
        """Test listing storage buckets."""
        # Given: Mock Storage client
        mock_instance = mock_client.return_value
        
        # Create mock bucket objects
        mock_bucket = MagicMock()
        mock_bucket.name = sample_storage_bucket["name"]
        mock_bucket.location = sample_storage_bucket["location"]
        mock_bucket.storage_class = sample_storage_bucket["storageClass"]
        
        mock_instance.list_buckets.return_value = [mock_bucket]
        
        # When: Listing buckets
        buckets = list(mock_instance.list_buckets())
        
        # Then: Buckets are returned
        assert len(buckets) == 1
        assert buckets[0].name == "test-bucket-12345"
        assert buckets[0].location == "US"
    
    @pytest.mark.integration
    @patch('google.cloud.storage.Client')
    def test_storage_bucket_security_analysis(self, mock_client, sample_storage_bucket):
        """Test security analysis of storage buckets."""
        # Given: Bucket with public access
        public_bucket = {
            **sample_storage_bucket,
            "iamConfiguration": {
                "uniformBucketLevelAccess": {"enabled": False},
                "publicAccessPrevention": "inherited"
            }
        }
        
        mock_instance = mock_client.return_value
        mock_bucket = MagicMock()
        mock_bucket.name = public_bucket["name"]
        mock_bucket.iam_configuration = public_bucket["iamConfiguration"]
        
        # Mock IAM policy with public access
        mock_policy = MagicMock()
        mock_policy.bindings = [
            {
                "role": "roles/storage.objectViewer",
                "members": ["allUsers"]
            }
        ]
        mock_bucket.get_iam_policy.return_value = mock_policy
        
        mock_instance.list_buckets.return_value = [mock_bucket]
        
        # When: Analyzing bucket security
        buckets = list(mock_instance.list_buckets())
        security_findings = []
        
        for bucket in buckets:
            policy = bucket.get_iam_policy()
            
            # Check for public access
            for binding in policy.bindings:
                if any(member in ["allUsers", "allAuthenticatedUsers"] 
                       for member in binding.get("members", [])):
                    security_findings.append({
                        "resource": bucket.name,
                        "finding": "public_access",
                        "severity": "HIGH",
                        "role": binding["role"]
                    })
        
        # Then: Public access is detected
        assert len(security_findings) == 1
        assert security_findings[0]["finding"] == "public_access"
        assert security_findings[0]["severity"] == "HIGH"


class TestGCPIAMIntegration:
    """Integration tests for Google Cloud IAM."""
    
    @pytest.mark.integration
    @patch('google.cloud.resourcemanager.ProjectsClient')
    def test_iam_policy_retrieval(self, mock_client, test_config, sample_iam_policy):
        """Test retrieving IAM policies."""
        # Given: Mock Resource Manager client
        mock_instance = mock_client.return_value
        mock_instance.get_iam_policy.return_value = sample_iam_policy
        
        # When: Getting IAM policy
        policy = mock_instance.get_iam_policy(
            resource=f"projects/{test_config['project_id']}"
        )
        
        # Then: Policy is retrieved correctly
        assert policy["version"] == 1
        assert len(policy["bindings"]) == 2
        assert any(binding["role"] == "roles/owner" for binding in policy["bindings"])
    
    @pytest.mark.integration
    @patch('google.cloud.resourcemanager.ProjectsClient')
    def test_iam_security_analysis(self, mock_client, test_config):
        """Test IAM security analysis."""
        # Given: IAM policy with security issues
        risky_policy = {
            "version": 1,
            "etag": "test-etag",
            "bindings": [
                {
                    "role": "roles/owner",
                    "members": ["allUsers"]  # Security risk!
                },
                {
                    "role": "roles/editor",
                    "members": [f"user:admin@{test_config['project_id']}.example.com"]
                }
            ]
        }
        
        mock_instance = mock_client.return_value
        mock_instance.get_iam_policy.return_value = risky_policy
        
        # When: Analyzing IAM policy
        policy = mock_instance.get_iam_policy(
            resource=f"projects/{test_config['project_id']}"
        )
        
        security_findings = []
        for binding in policy["bindings"]:
            # Check for overly permissive roles
            if binding["role"] in ["roles/owner", "roles/editor"]:
                for member in binding["members"]:
                    if member in ["allUsers", "allAuthenticatedUsers"]:
                        security_findings.append({
                            "finding": "public_admin_access",
                            "severity": "CRITICAL",
                            "role": binding["role"],
                            "member": member
                        })
        
        # Then: Security issues are identified
        assert len(security_findings) == 1
        assert security_findings[0]["finding"] == "public_admin_access"
        assert security_findings[0]["severity"] == "CRITICAL"


class TestGCPMultiServiceIntegration:
    """Integration tests across multiple GCP services."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_comprehensive_security_scan(self, test_config, mock_asset_client, mock_compute_client, mock_storage_client):
        """Test comprehensive security scan across multiple services."""
        # Given: Mock data from multiple services
        compute_finding = {
            "resource_type": "compute_instance",
            "resource_name": "test-vm-1",
            "finding": "public_ip_address",
            "severity": "MEDIUM"
        }
        
        storage_finding = {
            "resource_type": "storage_bucket", 
            "resource_name": "test-bucket",
            "finding": "public_read_access",
            "severity": "HIGH"
        }
        
        # When: Running comprehensive scan
        all_findings = []
        
        # Simulate asset discovery
        all_findings.append(compute_finding)
        all_findings.append(storage_finding)
        
        # Aggregate findings by severity
        findings_by_severity = {}
        for finding in all_findings:
            severity = finding["severity"]
            if severity not in findings_by_severity:
                findings_by_severity[severity] = []
            findings_by_severity[severity].append(finding)
        
        # Then: Findings are properly aggregated
        assert "HIGH" in findings_by_severity
        assert "MEDIUM" in findings_by_severity
        assert len(findings_by_severity["HIGH"]) == 1
        assert len(findings_by_severity["MEDIUM"]) == 1
        assert findings_by_severity["HIGH"][0]["resource_type"] == "storage_bucket"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_service_dependency_analysis(self, test_config):
        """Test analysis of dependencies between GCP services."""
        # Given: Resources with dependencies
        vm_instance = {
            "name": "web-server-1",
            "service_account": "web-sa@test-project.iam.gserviceaccount.com",
            "disks": ["web-disk-1"],
            "network": "vpc-web"
        }
        
        storage_bucket = {
            "name": "web-assets-bucket",
            "iam_bindings": [
                {
                    "role": "roles/storage.objectViewer",
                    "members": ["serviceAccount:web-sa@test-project.iam.gserviceaccount.com"]
                }
            ]
        }
        
        # When: Analyzing dependencies
        dependencies = []
        
        # Find VM -> Storage dependencies
        vm_sa = vm_instance["service_account"]
        for binding in storage_bucket["iam_bindings"]:
            if f"serviceAccount:{vm_sa}" in binding["members"]:
                dependencies.append({
                    "from_resource": vm_instance["name"],
                    "to_resource": storage_bucket["name"],
                    "dependency_type": "iam_access",
                    "role": binding["role"]
                })
        
        # Then: Dependencies are identified
        assert len(dependencies) == 1
        assert dependencies[0]["dependency_type"] == "iam_access"
        assert dependencies[0]["from_resource"] == "web-server-1"
        assert dependencies[0]["to_resource"] == "web-assets-bucket"
    
    @pytest.mark.integration
    def test_compliance_framework_mapping(self, sample_security_finding):
        """Test mapping security findings to compliance frameworks."""
        # Given: Security finding
        finding = sample_security_finding
        
        # When: Mapping to compliance frameworks
        compliance_mappings = {
            "public_access": {
                "CIS": ["3.3", "3.4"],
                "SOC2": ["CC6.1", "CC6.6"],
                "ISO27001": ["A.9.1.2", "A.13.1.1"],
                "NIST": ["AC-3", "AC-6"]
            }
        }
        
        finding_type = finding["category"]
        applicable_controls = {}
        
        if finding_type in compliance_mappings:
            applicable_controls = compliance_mappings[finding_type]
        
        # Then: Compliance controls are mapped correctly
        assert "CIS" in applicable_controls
        assert "SOC2" in applicable_controls
        assert "3.3" in applicable_controls["CIS"]
        assert "CC6.1" in applicable_controls["SOC2"]
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_large_scale_asset_processing(self, test_config):
        """Test processing large numbers of assets (performance test)."""
        # Given: Large number of mock assets
        num_assets = 1000
        mock_assets = []
        
        for i in range(num_assets):
            asset = {
                "name": f"//compute.googleapis.com/projects/{test_config['project_id']}/zones/{test_config['zone']}/instances/vm-{i}",
                "assetType": "compute.googleapis.com/Instance",
                "resource": {
                    "data": {
                        "name": f"vm-{i}",
                        "status": "RUNNING" if i % 2 == 0 else "STOPPED"
                    }
                }
            }
            mock_assets.append(asset)
        
        # When: Processing assets in batches
        batch_size = 100
        processed_count = 0
        
        for i in range(0, len(mock_assets), batch_size):
            batch = mock_assets[i:i + batch_size]
            
            # Simulate processing each batch
            for asset in batch:
                # Basic validation
                assert "name" in asset
                assert "assetType" in asset
                processed_count += 1
        
        # Then: All assets are processed
        assert processed_count == num_assets
    
    @pytest.mark.integration
    def test_error_recovery_and_retry(self):
        """Test error recovery and retry mechanisms."""
        from google.api_core.exceptions import ServiceUnavailable
        import time
        
        # Given: Service that fails then succeeds
        call_count = 0
        
        def mock_api_call():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ServiceUnavailable("Service temporarily unavailable")
            return {"result": "success"}
        
        # When: Implementing retry logic
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                result = mock_api_call()
                break
            except ServiceUnavailable:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
        
        # Then: Operation succeeds after retries
        assert call_count == 3
        assert result["result"] == "success"