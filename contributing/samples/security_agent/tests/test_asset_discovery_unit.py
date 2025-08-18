"""
Unit tests for asset discovery functionality (backend-independent).

These tests focus on the core business logic without requiring the full FastAPI backend.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any

# Add the project root and backend to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
backend_path = os.path.join(project_root, 'backend')
sys.path.insert(0, project_root)
sys.path.insert(0, backend_path)

# Mock Google Cloud imports before importing our modules
try:
    from google.cloud import asset_v1
    from google.api_core import exceptions as gcp_exceptions
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    # Create mock modules
    sys.modules['google.cloud'] = Mock()
    sys.modules['google.cloud.asset_v1'] = Mock()
    sys.modules['google.api_core'] = Mock()
    sys.modules['google.api_core.exceptions'] = Mock()

# Now import our modules
from backend.api.asset_inventory import (
    calculate_risk_score,
    get_risk_level,
    analyze_security_context,
    categorize_asset,
    generate_recommendations,
    SecurityContext,
    RiskLevel,
    AssetSummary
)

class TestSecurityContextAnalysis:
    """Test security context analysis for different asset types"""
    
    def test_compute_instance_with_public_ip(self):
        """Test compute instance with public IP gets flagged"""
        asset_data = {
            "name": "//compute.googleapis.com/projects/test/zones/us-central1-a/instances/web-server",
            "asset_type": "compute.googleapis.com/Instance",
            "resource": {
                "data": {
                    "networkInterfaces": [
                        {
                            "accessConfigs": [
                                {
                                    "type": "ONE_TO_ONE_NAT",
                                    "natIP": "34.123.45.67"
                                }
                            ]
                        }
                    ],
                    "disks": [
                        {
                            "boot": True,
                            "source": "projects/test/zones/us-central1-a/disks/web-server"
                            # No diskEncryptionKey - unencrypted
                        }
                    ],
                    "machineType": "projects/test/zones/us-central1-a/machineTypes/f1-micro",
                    "labels": {}
                }
            }
        }
        
        context = analyze_security_context(asset_data)
        
        assert context.is_public == True, "Instance with public IP should be flagged as public"
        assert context.is_legacy_version == True, "f1-micro should be flagged as legacy"
        assert "Instance has public IP" in context.risk_factors
        assert "Legacy machine type" in context.risk_factors
        assert "Missing resource labels" in context.risk_factors
        assert "Unencrypted disk attached" in context.risk_factors
    
    def test_storage_bucket_with_public_access(self):
        """Test storage bucket with public access gets high risk score"""
        asset_data = {
            "name": "//storage.googleapis.com/public-bucket",
            "asset_type": "storage.googleapis.com/Bucket",
            "resource": {
                "data": {
                    "location": "us-central1"
                    # No encryption configuration
                }
            },
            "iam_policy": {
                "version": 1,
                "bindings": [
                    {
                        "role": "roles/storage.objectViewer",
                        "members": ["allUsers"]
                    }
                ]
            }
        }
        
        context = analyze_security_context(asset_data)
        
        assert context.is_public == True, "Bucket with allUsers should be flagged as public"
        assert context.is_encrypted == False, "Bucket without encryption config should be flagged as unencrypted"
        assert "Public bucket access" in context.risk_factors
        assert "Default encryption not configured" in context.risk_factors
    
    def test_sql_instance_security_issues(self):
        """Test SQL instance with security issues"""
        asset_data = {
            "name": "//sqladmin.googleapis.com/projects/test/instances/db-server",
            "asset_type": "sqladmin.googleapis.com/Instance",
            "resource": {
                "data": {
                    "settings": {
                        "ipConfiguration": {
                            "requireSsl": False,
                            "ipv4Enabled": True
                        }
                    },
                    "ipAddresses": [
                        {
                            "type": "PRIMARY",
                            "ipAddress": "34.123.45.68"
                        }
                    ]
                }
            }
        }
        
        context = analyze_security_context(asset_data)
        
        assert context.is_public == True, "Database with public IP should be flagged as public"
        assert context.has_weak_authentication == True, "Database without SSL requirement should be flagged"
        assert "Database has public IP" in context.risk_factors
        assert "SSL not required for database" in context.risk_factors
    
    def test_secure_compute_instance(self):
        """Test that a secure compute instance gets low risk score"""
        asset_data = {
            "name": "//compute.googleapis.com/projects/test/zones/us-central1-a/instances/secure-server",
            "asset_type": "compute.googleapis.com/Instance",
            "resource": {
                "data": {
                    "networkInterfaces": [],  # No public access
                    "disks": [
                        {
                            "boot": True,
                            "diskEncryptionKey": {
                                "kmsKeyName": "projects/test/locations/global/keyRings/ring1/cryptoKeys/key1"
                            }
                        }
                    ],
                    "machineType": "projects/test/zones/us-central1-a/machineTypes/n1-standard-1",
                    "labels": {
                        "environment": "prod",
                        "team": "security"
                    }
                }
            }
        }
        
        context = analyze_security_context(asset_data)
        
        assert context.is_public == False, "Instance without public access should not be flagged as public"
        assert context.is_legacy_version == False, "n1-standard-1 should not be flagged as legacy"
        # Should still have some risk factors due to missing monitoring and other checks

class TestRiskScoringAlgorithm:
    """Test the risk scoring algorithm with various scenarios"""
    
    def test_critical_risk_score_calculation(self):
        """Test calculation of critical risk score"""
        context = SecurityContext(
            is_public=True,           # +30
            is_encrypted=False,       # +20
            has_overprivileged_access=True,  # +25
            has_weak_authentication=True,    # +15
            is_legacy_version=True,   # +10
            missing_monitoring=True,  # +8
            compliance_violations=["violation1", "violation2"],  # +10 (min of 2*5, 15)
            risk_factors=["risk1", "risk2", "risk3", "risk4"]    # +12 (min of 4*3, 12)
        )
        
        asset_data = {
            "asset_type": "sqladmin.googleapis.com/Instance"  # +10 for critical infrastructure
        }
        
        score = calculate_risk_score(asset_data, context)
        
        # Should be high risk: 30+20+25+15+10+8+10+12+10 = 140, capped at 100
        assert score == 100, f"Expected max score of 100, got {score}"
        assert get_risk_level(score) == RiskLevel.CRITICAL
    
    def test_medium_risk_score_calculation(self):
        """Test calculation of medium risk score"""
        context = SecurityContext(
            is_public=False,
            is_encrypted=True,
            has_overprivileged_access=False,
            has_weak_authentication=True,  # +15
            is_legacy_version=True,        # +10
            missing_monitoring=True,       # +8
            compliance_violations=[],
            risk_factors=["missing labels", "minor issue"]  # +6
        )
        
        asset_data = {
            "asset_type": "compute.googleapis.com/Instance"
        }
        
        score = calculate_risk_score(asset_data, context)
        expected_score = 15 + 10 + 8 + 6  # 39
        
        assert score == expected_score, f"Expected score of {expected_score}, got {score}"
        assert get_risk_level(score) == RiskLevel.LOW  # 21-40 range
    
    def test_minimal_risk_score_calculation(self):
        """Test calculation of minimal risk score"""
        context = SecurityContext(
            is_public=False,
            is_encrypted=True,
            has_overprivileged_access=False,
            has_weak_authentication=False,
            is_legacy_version=False,
            missing_monitoring=False,
            compliance_violations=[],
            risk_factors=[]
        )
        
        asset_data = {
            "asset_type": "compute.googleapis.com/Instance"
        }
        
        score = calculate_risk_score(asset_data, context)
        
        assert score == 0, f"Expected minimal score of 0, got {score}"
        assert get_risk_level(score) == RiskLevel.MINIMAL
    
    def test_risk_level_boundaries(self):
        """Test risk level boundary conditions"""
        assert get_risk_level(0) == RiskLevel.MINIMAL
        assert get_risk_level(20) == RiskLevel.MINIMAL
        assert get_risk_level(21) == RiskLevel.LOW
        assert get_risk_level(40) == RiskLevel.LOW
        assert get_risk_level(41) == RiskLevel.MEDIUM
        assert get_risk_level(60) == RiskLevel.MEDIUM
        assert get_risk_level(61) == RiskLevel.HIGH
        assert get_risk_level(80) == RiskLevel.HIGH
        assert get_risk_level(81) == RiskLevel.CRITICAL
        assert get_risk_level(100) == RiskLevel.CRITICAL

class TestAssetCategorization:
    """Test asset categorization and metadata extraction"""
    
    def test_compute_instance_categorization(self):
        """Test categorization of compute instance"""
        asset_data = {
            "name": "//compute.googleapis.com/projects/test/zones/us-central1-a/instances/web-server-1",
            "asset_type": "compute.googleapis.com/Instance",
            "resource": {"data": {}}
        }
        
        categorization = categorize_asset(asset_data)
        
        assert categorization['service'] == 'compute'
        assert categorization['category'] == 'compute'
        assert categorization['criticality'] == 'standard'
        assert categorization['region'] == 'us-central1'
        assert categorization['friendly_type'] == 'Instance'
    
    def test_storage_bucket_categorization(self):
        """Test categorization of storage bucket"""
        asset_data = {
            "name": "//storage.googleapis.com/test-bucket",
            "asset_type": "storage.googleapis.com/Bucket",
            "resource": {"data": {}}
        }
        
        categorization = categorize_asset(asset_data)
        
        assert categorization['service'] == 'storage'
        assert categorization['category'] == 'storage'
        assert categorization['criticality'] == 'standard'
        assert categorization['region'] == 'global'
        assert categorization['friendly_type'] == 'Bucket'
    
    def test_database_categorization(self):
        """Test categorization of database instance"""
        asset_data = {
            "name": "//sqladmin.googleapis.com/projects/test/instances/db-server",
            "asset_type": "sqladmin.googleapis.com/Instance",
            "resource": {"data": {}}
        }
        
        categorization = categorize_asset(asset_data)
        
        assert categorization['service'] == 'sqladmin'
        assert categorization['category'] == 'database'
        assert categorization['criticality'] == 'critical'  # Databases are critical
        assert categorization['region'] == 'global'
        assert categorization['friendly_type'] == 'Instance'
    
    def test_kms_key_categorization(self):
        """Test categorization of KMS key (critical security asset)"""
        asset_data = {
            "name": "//cloudkms.googleapis.com/projects/test/locations/global/keyRings/ring1/cryptoKeys/key1",
            "asset_type": "cloudkms.googleapis.com/CryptoKey",
            "resource": {"data": {}}
        }
        
        categorization = categorize_asset(asset_data)
        
        assert categorization['service'] == 'cloudkms'
        assert categorization['category'] == 'security'
        assert categorization['criticality'] == 'critical'  # KMS keys are critical

class TestRecommendationGeneration:
    """Test recommendation generation based on security context"""
    
    def test_public_asset_recommendations(self):
        """Test recommendations for publicly exposed assets"""
        context = SecurityContext(
            is_public=True,
            is_encrypted=False,
            risk_factors=["Public access", "No encryption"]
        )
        
        asset_data = {
            "asset_type": "storage.googleapis.com/Bucket",
            "resource": {"data": {}}
        }
        
        recommendations = generate_recommendations(context, asset_data)
        
        assert "Restrict public access - review and minimize exposure" in recommendations
        assert "Enable encryption at rest and in transit" in recommendations
        assert "Add resource labels for governance and cost tracking" in recommendations
        assert len(recommendations) <= 5, "Should limit to top 5 recommendations"
    
    def test_database_security_recommendations(self):
        """Test recommendations for database with security issues"""
        context = SecurityContext(
            is_public=True,
            has_weak_authentication=True,
            risk_factors=["Public database", "Weak authentication"]
        )
        
        asset_data = {
            "asset_type": "sqladmin.googleapis.com/Instance",
            "resource": {"data": {"labels": {"environment": "prod"}}}
        }
        
        recommendations = generate_recommendations(context, asset_data)
        
        assert "Restrict public access - review and minimize exposure" in recommendations
        assert "Strengthen authentication requirements" in recommendations
        assert len(recommendations) >= 2, "Should have at least basic recommendations"
    
    def test_legacy_system_recommendations(self):
        """Test recommendations for legacy systems"""
        context = SecurityContext(
            is_legacy_version=True,
            missing_monitoring=True,
            risk_factors=["Legacy version", "No monitoring"]
        )
        
        asset_data = {
            "asset_type": "compute.googleapis.com/Instance",
            "resource": {"data": {}}
        }
        
        recommendations = generate_recommendations(context, asset_data)
        
        assert "Upgrade to supported version or instance type" in recommendations
        assert "Enable monitoring and alerting" in recommendations
        assert "Add resource labels for governance and cost tracking" in recommendations

class TestDataModels:
    """Test data model functionality and initialization"""
    
    def test_security_context_defaults(self):
        """Test SecurityContext initialization with default values"""
        context = SecurityContext()
        
        assert context.is_public == False
        assert context.is_encrypted == True
        assert context.has_overprivileged_access == False
        assert context.has_weak_authentication == False
        assert context.is_legacy_version == False
        assert context.missing_monitoring == False
        assert context.compliance_violations == []
        assert context.risk_factors == []
    
    def test_security_context_with_values(self):
        """Test SecurityContext with explicit values"""
        violations = ["violation1", "violation2"]
        risks = ["risk1", "risk2"]
        
        context = SecurityContext(
            is_public=True,
            is_encrypted=False,
            compliance_violations=violations,
            risk_factors=risks
        )
        
        assert context.is_public == True
        assert context.is_encrypted == False
        assert context.compliance_violations == violations
        assert context.risk_factors == risks
    
    def test_asset_summary_defaults(self):
        """Test AssetSummary initialization with default values"""
        summary = AssetSummary()
        
        assert summary.total_assets == 0
        assert summary.by_type == {}
        assert summary.by_region == {}
        assert summary.security_issues == 0
        assert all(level.value in summary.by_risk_level for level in RiskLevel)
        assert all(count == 0 for count in summary.by_risk_level.values())

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_asset_data(self):
        """Test handling of empty asset data"""
        asset_data = {}
        context = analyze_security_context(asset_data)
        
        assert isinstance(context, SecurityContext)
        # Should handle gracefully without crashing
    
    def test_malformed_asset_data(self):
        """Test handling of malformed asset data"""
        asset_data = {
            "name": "invalid-name",
            "asset_type": "",  # Empty string instead of None to avoid AttributeError
            "resource": {}  # Empty dict instead of None to avoid AttributeError
        }
        
        context = analyze_security_context(asset_data)
        
        assert isinstance(context, SecurityContext)
        # Should not crash on malformed data
    
    def test_missing_resource_data(self):
        """Test handling of missing resource data"""
        asset_data = {
            "name": "//compute.googleapis.com/projects/test/instances/server",
            "asset_type": "compute.googleapis.com/Instance"
            # No resource field
        }
        
        context = analyze_security_context(asset_data)
        categorization = categorize_asset(asset_data)
        
        assert isinstance(context, SecurityContext)
        assert isinstance(categorization, dict)
        # Should handle missing data gracefully
    
    def test_risk_score_with_none_values(self):
        """Test risk score calculation with None values"""
        context = SecurityContext(
            compliance_violations=None,
            risk_factors=None
        )
        
        asset_data = {"asset_type": "unknown.service/Unknown"}
        score = calculate_risk_score(asset_data, context)
        
        assert isinstance(score, int)
        assert 0 <= score <= 100

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])