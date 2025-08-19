"""
Comprehensive test suite for Security API endpoints - TASK-003.

Tests security analysis, vulnerability scanning, CVSS scoring, threat detection,
compliance checks, and security command center integration.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json

# Import the security module and related components
from backend.api.security import router
from backend.main import app

client = TestClient(app)

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_security_client():
    """Mock Google Cloud Security Command Center client."""
    with patch('backend.api.security.securitycenter.SecurityCenterClient') as mock_client:
        yield mock_client

@pytest.fixture
def mock_finding():
    """Mock security finding object."""
    finding = Mock()
    finding.name = "organizations/123456789/sources/1234567890/findings/test-finding-1"
    finding.parent = "organizations/123456789/sources/1234567890"
    finding.resource_name = "//compute.googleapis.com/projects/test-project/zones/us-central1-a/instances/test-vm"
    finding.state = "ACTIVE"
    finding.category = "MALWARE"
    finding.external_uri = "https://console.cloud.google.com/security/findings"
    finding.source_properties = {
        "severity": "HIGH",
        "description": "Malware detected on instance",
        "recommendation": "Isolate and investigate the instance"
    }
    finding.security_marks = {}
    finding.event_time = datetime.now()
    finding.create_time = datetime.now()
    finding.severity = "HIGH"
    return finding

@pytest.fixture
def mock_asset():
    """Mock asset object."""
    asset = Mock()
    asset.name = "//compute.googleapis.com/projects/test-project/zones/us-central1-a/instances/test-vm"
    asset.asset_type = "compute.googleapis.com/Instance"
    asset.resource = Mock()
    asset.resource.version = "v1"
    asset.resource.discovery_document_uri = "https://www.googleapis.com/discovery/v1/apis/compute/v1/rest"
    asset.resource.discovery_name = "Instance"
    asset.resource.resource_url = "https://compute.googleapis.com/compute/v1/projects/test-project/zones/us-central1-a/instances/test-vm"
    asset.resource.parent = "//cloudresourcemanager.googleapis.com/projects/test-project"
    asset.resource.data = {
        "name": "test-vm",
        "zone": "us-central1-a",
        "machineType": "n1-standard-1",
        "status": "RUNNING"
    }
    asset.iam_policy = None
    asset.org_policy = []
    asset.access_policy = None
    asset.os_inventory = None
    asset.related_assets = None
    return asset

@pytest.fixture
def mock_vulnerability():
    """Mock vulnerability assessment result."""
    vuln = Mock()
    vuln.name = "CVE-2024-1234"
    vuln.description = "Critical security vulnerability in system component"
    vuln.severity = "CRITICAL"
    vuln.cvss_score = 9.8
    vuln.cvss_vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    vuln.affected_package = "vulnerable-package"
    vuln.fixed_version = "1.2.3"
    vuln.installed_version = "1.0.0"
    vuln.fix_available = True
    return vuln

@pytest.fixture
def mock_credentials():
    """Mock GCP credentials."""
    with patch('backend.api.security._get_credentials') as mock_creds:
        mock_creds.return_value = Mock()
        yield mock_creds

# ============================================================================
# SECURITY ANALYSIS ENDPOINT TESTS
# ============================================================================

class TestSecurityAnalysisEndpoints:
    """Test security analysis endpoints."""

    def test_analyze_security_success(self, mock_credentials, mock_security_client, mock_finding):
        """Test successful security analysis."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_security_client.return_value = mock_client_instance
        
        # Mock findings listing
        mock_client_instance.list_findings.return_value = [mock_finding]
        
        response = client.get("/api/v1/security/analyze/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["project_id"] == "test-project"
        assert "findings" in data
        assert "summary" in data
        assert "risk_score" in data

    def test_analyze_security_with_filters(self, mock_credentials, mock_security_client):
        """Test security analysis with filters."""
        response = client.get("/api/v1/security/analyze/test-project?severity=HIGH&category=MALWARE&state=ACTIVE")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "filtered_results" in data

    def test_analyze_security_comprehensive(self, mock_credentials):
        """Test comprehensive security analysis."""
        response = client.get("/api/v1/security/analyze/test-project?comprehensive=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "vulnerability_scan" in data
        assert "compliance_check" in data
        assert "threat_analysis" in data
        assert "recommendations" in data

    def test_analyze_security_no_credentials(self, mock_credentials):
        """Test security analysis without credentials."""
        mock_credentials.return_value = None
        
        response = client.get("/api/v1/security/analyze/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data_source"] == "api_failed"  # Falls back to mock data

# ============================================================================
# VULNERABILITY SCANNING TESTS
# ============================================================================

class TestVulnerabilityScanning:
    """Test vulnerability scanning functionality."""

    def test_vulnerability_scan_success(self, mock_credentials, mock_vulnerability):
        """Test successful vulnerability scan."""
        with patch('backend.api.security._run_vulnerability_scan') as mock_scan:
            mock_scan.return_value = {
                "success": True,
                "vulnerabilities": [mock_vulnerability],
                "critical_count": 1,
                "high_count": 2,
                "medium_count": 5,
                "low_count": 10
            }
            
            response = client.post("/api/v1/security/vulnerability-scan/test-project", json={
                "scan_type": "comprehensive",
                "include_packages": True,
                "include_os": True
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "vulnerabilities" in data
            assert "summary" in data

    def test_vulnerability_scan_with_filters(self, mock_credentials):
        """Test vulnerability scan with severity filters."""
        response = client.post("/api/v1/security/vulnerability-scan/test-project", json={
            "scan_type": "targeted",
            "severity_filter": ["CRITICAL", "HIGH"],
            "package_filter": ["vulnerable-package"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_vulnerability_scan_cvss_scoring(self, mock_credentials, mock_vulnerability):
        """Test CVSS scoring integration."""
        with patch('backend.api.security._calculate_cvss_score') as mock_cvss:
            mock_cvss.return_value = {
                "base_score": 9.8,
                "temporal_score": 9.1,
                "environmental_score": 8.7,
                "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
            }
            
            response = client.post("/api/v1/security/vulnerability-scan/test-project", json={
                "scan_type": "cvss_analysis",
                "include_cvss_details": True
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "cvss_analysis" in data

# ============================================================================
# THREAT DETECTION TESTS
# ============================================================================

class TestThreatDetection:
    """Test threat detection functionality."""

    def test_threat_detection_scan(self, mock_credentials):
        """Test threat detection scan."""
        response = client.post("/api/v1/security/threat-detection/test-project", json={
            "detection_rules": ["malware", "suspicious_activity", "data_exfiltration"],
            "time_range": "24h",
            "include_details": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "threats" in data
        assert "detection_summary" in data
        assert "recommended_actions" in data

    def test_behavioral_analysis(self, mock_credentials):
        """Test behavioral analysis for threat detection."""
        response = client.post("/api/v1/security/behavioral-analysis/test-project", json={
            "analysis_type": "anomaly_detection",
            "baseline_period": "7d",
            "sensitivity": "medium"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "anomalies" in data
        assert "risk_indicators" in data

# ============================================================================
# COMPLIANCE CHECKING TESTS
# ============================================================================

class TestComplianceChecking:
    """Test security compliance checking."""

    def test_compliance_check_multiple_frameworks(self, mock_credentials):
        """Test compliance checking against multiple frameworks."""
        response = client.post("/api/v1/security/compliance-check/test-project", json={
            "frameworks": ["SOC2", "HIPAA", "PCI-DSS", "ISO27001"],
            "include_remediation": True,
            "detailed_report": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "compliance_results" in data
        assert "overall_score" in data
        assert "gaps" in data
        assert "remediation_plan" in data

    def test_compliance_check_single_framework(self, mock_credentials):
        """Test compliance checking for single framework."""
        response = client.post("/api/v1/security/compliance-check/test-project", json={
            "frameworks": ["SOC2"],
            "control_categories": ["access_control", "monitoring", "incident_response"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "soc2_results" in data

    def test_custom_compliance_rules(self, mock_credentials):
        """Test custom compliance rule checking."""
        custom_rules = [
            {
                "name": "encryption_at_rest",
                "description": "All storage must be encrypted at rest",
                "rule_type": "storage_encryption",
                "severity": "HIGH"
            },
            {
                "name": "mfa_required",
                "description": "MFA required for all admin accounts",
                "rule_type": "iam_policy",
                "severity": "CRITICAL"
            }
        ]
        
        response = client.post("/api/v1/security/compliance-check/test-project", json={
            "custom_rules": custom_rules,
            "include_standard_frameworks": False
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "custom_rule_results" in data

# ============================================================================
# SECURITY RECOMMENDATIONS TESTS
# ============================================================================

class TestSecurityRecommendations:
    """Test security recommendations functionality."""

    def test_generate_security_recommendations(self, mock_credentials):
        """Test generation of security recommendations."""
        response = client.get("/api/v1/security/recommendations/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "recommendations" in data
        assert "priority_actions" in data
        assert "automation_scripts" in data

    def test_recommendations_with_context(self, mock_credentials):
        """Test recommendations with specific context."""
        response = client.post("/api/v1/security/recommendations/test-project", json={
            "context": {
                "industry": "healthcare",
                "data_classification": "PII",
                "compliance_requirements": ["HIPAA", "SOC2"]
            },
            "focus_areas": ["data_protection", "access_control", "monitoring"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "contextual_recommendations" in data

    def test_prioritized_recommendations(self, mock_credentials):
        """Test prioritized security recommendations."""
        response = client.get("/api/v1/security/recommendations/test-project?prioritize=true&max_recommendations=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "priority_score" in str(data)
        assert len(data["recommendations"]) <= 10

# ============================================================================
# SECURITY METRICS TESTS
# ============================================================================

class TestSecurityMetrics:
    """Test security metrics and scoring."""

    def test_security_score_calculation(self, mock_credentials):
        """Test security score calculation."""
        response = client.get("/api/v1/security/score/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "overall_score" in data
        assert "category_scores" in data
        assert "improvement_areas" in data
        assert 0 <= data["overall_score"] <= 100

    def test_security_metrics_dashboard(self, mock_credentials):
        """Test security metrics dashboard."""
        response = client.get("/api/v1/security/metrics/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "metrics" in data
        assert "trends" in data
        assert "kpis" in data

    def test_security_posture_assessment(self, mock_credentials):
        """Test comprehensive security posture assessment."""
        response = client.post("/api/v1/security/posture-assessment/test-project", json={
            "assessment_scope": "full",
            "include_historical": True,
            "benchmark_comparison": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "posture_score" in data
        assert "risk_categories" in data
        assert "benchmark_comparison" in data

# ============================================================================
# INCIDENT RESPONSE TESTS
# ============================================================================

class TestIncidentResponse:
    """Test incident response functionality."""

    def test_security_incident_detection(self, mock_credentials, mock_finding):
        """Test security incident detection."""
        with patch('backend.api.security._detect_security_incidents') as mock_detect:
            mock_detect.return_value = [
                {
                    "incident_id": "INC-001",
                    "severity": "HIGH",
                    "type": "data_breach",
                    "description": "Potential data exfiltration detected",
                    "affected_resources": ["test-vm"],
                    "recommended_actions": ["Isolate resources", "Investigate logs"]
                }
            ]
            
            response = client.get("/api/v1/security/incidents/test-project")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "incidents" in data
            assert "summary" in data

    def test_incident_response_playbook(self, mock_credentials):
        """Test incident response playbook generation."""
        response = client.post("/api/v1/security/incident-response/test-project", json={
            "incident_type": "malware",
            "severity": "HIGH",
            "affected_resources": ["test-vm", "test-storage"],
            "generate_playbook": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "playbook" in data
        assert "immediate_steps" in data
        assert "investigation_steps" in data
        assert "recovery_steps" in data

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestSecurityErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_project_id(self):
        """Test handling of invalid project ID."""
        response = client.get("/api/v1/security/analyze/invalid-project-id!")
        
        # Should handle gracefully or return appropriate error
        assert response.status_code in [400, 200]

    def test_api_permission_denied(self, mock_credentials):
        """Test handling of permission denied errors."""
        with patch('backend.api.security.securitycenter.SecurityCenterClient') as mock_client:
            mock_client.side_effect = Exception("Permission denied")
            
            response = client.get("/api/v1/security/analyze/test-project")
            
            # Should handle gracefully
            assert response.status_code == 200
            data = response.json()
            # Should fallback to mock data
            assert data["data_source"] == "api_failed"

    def test_security_scan_timeout(self, mock_credentials):
        """Test handling of scan timeout."""
        with patch('backend.api.security._run_security_scan') as mock_scan:
            mock_scan.side_effect = TimeoutError("Scan timeout")
            
            response = client.get("/api/v1/security/analyze/test-project")
            
            assert response.status_code == 200
            data = response.json()
            # Should handle timeout gracefully
            assert "timeout" in str(data).lower() or data["data_source"] == "api_failed"

    def test_malformed_scan_request(self):
        """Test handling of malformed scan request."""
        response = client.post("/api/v1/security/vulnerability-scan/test-project", json={
            "invalid_field": "invalid_value",
            "scan_type": "invalid_type"
        })
        
        # Should validate input
        assert response.status_code in [400, 422]

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSecurityIntegration:
    """Test security integration scenarios."""

    def test_full_security_assessment(self, mock_credentials):
        """Test complete security assessment workflow."""
        response = client.post("/api/v1/security/full-assessment/test-project", json={
            "assessment_type": "comprehensive",
            "include_vulnerability_scan": True,
            "include_compliance_check": True,
            "include_threat_detection": True,
            "frameworks": ["SOC2", "ISO27001"],
            "generate_report": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Should have comprehensive results
        assert "vulnerability_results" in data
        assert "compliance_results" in data
        assert "threat_analysis" in data
        assert "overall_risk_score" in data
        assert "executive_summary" in data

    def test_security_monitoring_setup(self, mock_credentials):
        """Test security monitoring setup."""
        response = client.post("/api/v1/security/monitoring/test-project", json={
            "monitoring_config": {
                "enable_real_time_alerts": True,
                "alert_thresholds": {
                    "critical": "immediate",
                    "high": "15m",
                    "medium": "1h"
                },
                "notification_channels": ["email", "slack"]
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "monitoring_setup" in data

    def test_security_automation_scripts(self, mock_credentials):
        """Test security automation script generation."""
        response = client.post("/api/v1/security/automation/test-project", json={
            "automation_type": "remediation",
            "security_issues": ["weak_passwords", "unencrypted_storage", "public_buckets"],
            "approval_required": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "automation_scripts" in data
        assert "approval_workflow" in data

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestSecurityPerformance:
    """Test security API performance."""

    def test_large_scan_performance(self, mock_credentials):
        """Test performance with large-scale security scan."""
        import time
        
        start_time = time.time()
        response = client.post("/api/v1/security/vulnerability-scan/large-project", json={
            "scan_type": "comprehensive",
            "parallel_scanning": True
        })
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 10.0  # 10 seconds max
        assert response.status_code == 200

    def test_concurrent_security_requests(self, mock_credentials):
        """Test handling of concurrent security analysis requests."""
        import threading
        
        results = []
        
        def make_request():
            response = client.get("/api/v1/security/analyze/test-project")
            results.append(response.status_code)
        
        # Start multiple concurrent requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])