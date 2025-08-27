"""
Mock GCP Data for Testing
Provides realistic test data for Security Agent testing
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any

class MockGCPData:
    """Mock GCP API responses for testing."""
    
    @staticmethod
    def get_mock_assets() -> List[Dict[str, Any]]:
        """Mock Cloud Asset Inventory data."""
        return [
            {
                "name": "//compute.googleapis.com/projects/test-project/zones/us-central1-a/instances/web-server-1",
                "asset_type": "compute.googleapis.com/Instance",
                "resource": {
                    "data": {
                        "name": "web-server-1",
                        "status": "RUNNING",
                        "machineType": "n1-standard-1",
                        "tags": {"items": ["web", "production"]},
                        "networkInterfaces": [{
                            "accessConfigs": [{
                                "type": "ONE_TO_ONE_NAT",
                                "natIP": "34.68.1.1"
                            }]
                        }]
                    }
                },
                "ancestors": ["projects/test-project"],
                "update_time": datetime.now().isoformat()
            },
            {
                "name": "//storage.googleapis.com/test-bucket-public",
                "asset_type": "storage.googleapis.com/Bucket",
                "resource": {
                    "data": {
                        "name": "test-bucket-public",
                        "location": "US",
                        "storageClass": "STANDARD",
                        "iamConfiguration": {
                            "publicAccessPrevention": "inherited"
                        },
                        "lifecycle": {"rule": []},
                        "versioning": {"enabled": False}
                    }
                },
                "iam_policy": {
                    "bindings": [{
                        "role": "roles/storage.objectViewer",
                        "members": ["allUsers"]
                    }]
                },
                "ancestors": ["projects/test-project"],
                "update_time": datetime.now().isoformat()
            },
            {
                "name": "//storage.googleapis.com/test-bucket-private",
                "asset_type": "storage.googleapis.com/Bucket",
                "resource": {
                    "data": {
                        "name": "test-bucket-private",
                        "location": "US",
                        "storageClass": "STANDARD",
                        "iamConfiguration": {
                            "publicAccessPrevention": "enforced"
                        },
                        "encryption": {
                            "defaultKmsKeyName": "projects/test-project/locations/us/keyRings/test/cryptoKeys/test-key"
                        },
                        "versioning": {"enabled": True}
                    }
                },
                "ancestors": ["projects/test-project"],
                "update_time": datetime.now().isoformat()
            }
        ]

    @staticmethod
    def get_mock_security_findings() -> List[Dict[str, Any]]:
        """Mock Security Command Center findings."""
        return [
            {
                "name": "organizations/123/sources/456/findings/critical-001",
                "finding_class": "VULNERABILITY",
                "severity": "CRITICAL",
                "state": "ACTIVE",
                "category": "PUBLIC_BUCKET",
                "resource_name": "//storage.googleapis.com/test-bucket-public",
                "event_time": datetime.now().isoformat(),
                "create_time": (datetime.now() - timedelta(days=7)).isoformat(),
                "source_properties": {
                    "description": "Storage bucket is publicly accessible",
                    "recommendation": "Enable uniform bucket-level access and remove allUsers binding",
                    "affected_resource": "test-bucket-public",
                    "risk_score": 9.5
                }
            },
            {
                "name": "organizations/123/sources/456/findings/high-001",
                "finding_class": "MISCONFIGURATION",
                "severity": "HIGH",
                "state": "ACTIVE",
                "category": "OVERLY_PERMISSIVE_IAM",
                "resource_name": "//iam.googleapis.com/projects/test-project/serviceAccounts/wide-permissions@test.iam",
                "event_time": datetime.now().isoformat(),
                "create_time": (datetime.now() - timedelta(days=3)).isoformat(),
                "source_properties": {
                    "description": "Service account has overly permissive roles",
                    "recommendation": "Apply principle of least privilege",
                    "affected_roles": ["roles/owner", "roles/editor"],
                    "risk_score": 7.8
                }
            },
            {
                "name": "organizations/123/sources/456/findings/medium-001",
                "finding_class": "OBSERVATION",
                "severity": "MEDIUM",
                "state": "ACTIVE",
                "category": "MISSING_ENCRYPTION",
                "resource_name": "//compute.googleapis.com/projects/test-project/disks/unencrypted-disk",
                "event_time": datetime.now().isoformat(),
                "create_time": (datetime.now() - timedelta(days=14)).isoformat(),
                "source_properties": {
                    "description": "Disk is not encrypted with customer-managed keys",
                    "recommendation": "Enable encryption with Cloud KMS",
                    "risk_score": 5.2
                }
            }
        ]

    @staticmethod
    def get_mock_iam_policies() -> List[Dict[str, Any]]:
        """Mock IAM policies."""
        return [
            {
                "resource": "projects/test-project",
                "policy": {
                    "bindings": [
                        {
                            "role": "roles/owner",
                            "members": ["user:admin@example.com"]
                        },
                        {
                            "role": "roles/editor",
                            "members": [
                                "serviceAccount:wide-permissions@test.iam.gserviceaccount.com",
                                "user:developer@example.com"
                            ]
                        },
                        {
                            "role": "roles/viewer",
                            "members": [
                                "group:security-team@example.com",
                                "serviceAccount:readonly@test.iam.gserviceaccount.com"
                            ]
                        }
                    ],
                    "etag": "BwXpK7E2M0Y=",
                    "version": 1
                }
            }
        ]

    @staticmethod
    def get_mock_storage_buckets() -> List[Dict[str, Any]]:
        """Mock Storage bucket details."""
        return [
            {
                "name": "test-bucket-public",
                "location": "US",
                "storage_class": "STANDARD",
                "public_access": True,
                "uniform_access": False,
                "versioning": False,
                "encryption": None,
                "lifecycle_rules": 0,
                "retention_policy": None,
                "labels": {"env": "test", "public": "true"},
                "created": (datetime.now() - timedelta(days=180)).isoformat(),
                "updated": datetime.now().isoformat()
            },
            {
                "name": "test-bucket-private",
                "location": "US",
                "storage_class": "STANDARD",
                "public_access": False,
                "uniform_access": True,
                "versioning": True,
                "encryption": "CMEK",
                "lifecycle_rules": 2,
                "retention_policy": "30 days",
                "labels": {"env": "production", "compliance": "required"},
                "created": (datetime.now() - timedelta(days=90)).isoformat(),
                "updated": datetime.now().isoformat()
            },
            {
                "name": "backup-bucket",
                "location": "EU",
                "storage_class": "NEARLINE",
                "public_access": False,
                "uniform_access": True,
                "versioning": True,
                "encryption": "CMEK",
                "lifecycle_rules": 3,
                "retention_policy": "365 days",
                "labels": {"type": "backup", "compliance": "gdpr"},
                "created": (datetime.now() - timedelta(days=365)).isoformat(),
                "updated": (datetime.now() - timedelta(days=1)).isoformat()
            }
        ]

    @staticmethod
    def get_mock_recommendations() -> List[Dict[str, Any]]:
        """Mock Recommender API recommendations."""
        return [
            {
                "name": "projects/test-project/locations/global/recommenders/google.iam.policy.Recommender/recommendations/rec-001",
                "description": "Remove unused IAM role binding",
                "primary_impact": {
                    "category": "SECURITY",
                    "security_projection": {
                        "risk_reduction": "HIGH"
                    }
                },
                "state_info": {
                    "state": "ACTIVE"
                },
                "content": {
                    "operation_groups": [{
                        "operations": [{
                            "action": "remove",
                            "resource_type": "iam.googleapis.com/Binding",
                            "path": "/bindings/*/members/*",
                            "value": "serviceAccount:unused@test.iam.gserviceaccount.com"
                        }]
                    }]
                },
                "priority": "P2",
                "associated_insights": []
            },
            {
                "name": "projects/test-project/locations/global/recommenders/google.compute.instance.IdleResourceRecommender/recommendations/rec-002",
                "description": "Delete idle VM instance",
                "primary_impact": {
                    "category": "COST",
                    "cost_projection": {
                        "cost": {
                            "currency_code": "USD",
                            "units": "150"
                        }
                    }
                },
                "state_info": {
                    "state": "ACTIVE"
                },
                "content": {
                    "operation_groups": [{
                        "operations": [{
                            "action": "delete",
                            "resource_type": "compute.googleapis.com/Instance",
                            "path": "",
                            "resource": "//compute.googleapis.com/projects/test-project/zones/us-central1-a/instances/idle-vm"
                        }]
                    }]
                },
                "priority": "P3",
                "associated_insights": []
            }
        ]

    @staticmethod
    def get_mock_org_policies() -> List[Dict[str, Any]]:
        """Mock Organization policies."""
        return [
            {
                "name": "projects/test-project/policies/compute.requireShieldedVm",
                "spec": {
                    "rules": [{
                        "enforce": True
                    }]
                }
            },
            {
                "name": "projects/test-project/policies/storage.publicAccessPrevention",
                "spec": {
                    "rules": [{
                        "enforce": False
                    }]
                }
            },
            {
                "name": "projects/test-project/policies/iam.disableServiceAccountCreation",
                "spec": {
                    "rules": [{
                        "enforce": False
                    }]
                }
            }
        ]

    @staticmethod
    def get_mock_api_keys() -> List[Dict[str, Any]]:
        """Mock API keys."""
        return [
            {
                "name": "projects/test-project/locations/global/keys/test-api-key-001",
                "display_name": "Test API Key 1",
                "restrictions": {
                    "api_targets": [],
                    "browser_key_restrictions": None,
                    "server_key_restrictions": None,
                    "android_key_restrictions": None,
                    "ios_key_restrictions": None
                },
                "create_time": (datetime.now() - timedelta(days=60)).isoformat(),
                "uid": "key-001"
            },
            {
                "name": "projects/test-project/locations/global/keys/restricted-key-002",
                "display_name": "Restricted API Key",
                "restrictions": {
                    "api_targets": [
                        {"service": "maps.googleapis.com"}
                    ],
                    "browser_key_restrictions": {
                        "allowed_referrers": ["https://example.com/*"]
                    }
                },
                "create_time": (datetime.now() - timedelta(days=30)).isoformat(),
                "uid": "key-002"
            }
        ]

    @staticmethod
    def get_dashboard_metrics() -> Dict[str, Any]:
        """Get mock metrics for dashboard display."""
        return {
            "critical_findings": 1,
            "high_risk_assets": 2,
            "compliance_score": 72,
            "security_recommendations": 5,
            "public_buckets": 1,
            "overly_permissive_roles": 3,
            "unencrypted_resources": 1,
            "total_assets": 12,
            "last_scan": datetime.now().isoformat(),
            "trend_data": {
                "dates": [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7, -1, -1)],
                "critical": [2, 1, 1, 2, 1, 1, 1, 1],
                "high": [5, 4, 3, 3, 2, 2, 2, 2],
                "medium": [8, 8, 7, 6, 5, 4, 3, 3],
                "low": [12, 11, 10, 10, 9, 8, 7, 7]
            }
        }

    @staticmethod
    def get_compliance_status() -> Dict[str, Any]:
        """Get mock compliance status."""
        return {
            "soc2": {
                "status": "partial",
                "score": 75,
                "controls_passed": 45,
                "controls_total": 60,
                "critical_gaps": ["Access logging not fully implemented", "Encryption at rest not enforced"]
            },
            "gdpr": {
                "status": "compliant",
                "score": 92,
                "controls_passed": 23,
                "controls_total": 25,
                "critical_gaps": []
            },
            "hipaa": {
                "status": "non_compliant",
                "score": 45,
                "controls_passed": 18,
                "controls_total": 40,
                "critical_gaps": ["PHI encryption required", "Audit logs retention insufficient", "Access controls too permissive"]
            },
            "pci_dss": {
                "status": "not_applicable",
                "score": 0,
                "controls_passed": 0,
                "controls_total": 0,
                "critical_gaps": []
            }
        }