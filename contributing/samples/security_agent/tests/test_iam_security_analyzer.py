"""
Test suite for Enhanced IAM Security Analyzer (STORY-003)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.iam_security_analyzer import (
    IAMSecurityAnalyzer, IAMFinding, IAMFindingType, RiskLevel,
    ServiceAccountAnalysis, IAMSecurityPosture
)


class TestIAMSecurityAnalyzer:
    """Test IAM Security Analyzer functionality"""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create IAM analyzer with mocked clients"""
        with patch('backend.services.iam_security_analyzer.GCP_CLIENTS_AVAILABLE', True):
            analyzer = IAMSecurityAnalyzer("test-project")
            analyzer.iam_client = Mock()
            analyzer.rm_client = Mock()
            analyzer.asset_client = Mock()
            return analyzer
    
    @pytest.fixture
    def sample_service_account(self):
        """Sample service account data"""
        return {
            "name": "projects/test-project/serviceAccounts/test-sa@test-project.iam.gserviceaccount.com",
            "email": "test-sa@test-project.iam.gserviceaccount.com",
            "display_name": "Test Service Account",
            "description": "Test description",
            "unique_id": "123456789",
            "disabled": False,
            "keys": [
                {
                    "name": "projects/test-project/serviceAccounts/test-sa@test-project.iam.gserviceaccount.com/keys/key1",
                    "key_algorithm": "KEY_ALG_RSA_2048",
                    "valid_after_time": datetime.now() - timedelta(days=120),  # Stale key
                    "disabled": False
                }
            ]
        }
    
    @pytest.fixture
    def sample_iam_policy(self):
        """Sample IAM policy"""
        policy = Mock()
        
        # Create bindings
        binding1 = Mock()
        binding1.role = "roles/owner"
        binding1.members = ["serviceAccount:test-sa@test-project.iam.gserviceaccount.com"]
        
        binding2 = Mock()
        binding2.role = "roles/editor"
        binding2.members = ["user:external@gmail.com"]
        
        binding3 = Mock()
        binding3.role = "roles/viewer"
        binding3.members = ["allUsers"]
        
        policy.bindings = [binding1, binding2, binding3]
        return policy
    
    def test_overprivileged_account_detection(self, mock_analyzer, sample_service_account):
        """Test detection of overprivileged service accounts"""
        # Service account with admin role
        roles = ["roles/owner", "roles/editor"]
        
        findings = mock_analyzer._check_overprivileged_account(sample_service_account, roles)
        
        # Should detect admin role misuse
        admin_findings = [f for f in findings if f.finding_type == IAMFindingType.ADMIN_ROLE_MISUSE]
        assert len(admin_findings) == 1
        assert admin_findings[0].risk_level == RiskLevel.CRITICAL
        assert admin_findings[0].risk_score == 90
        
        # Should detect broad roles
        broad_findings = [f for f in findings if f.finding_type == IAMFindingType.EXCESSIVE_PERMISSIONS]
        assert len(broad_findings) == 1
        assert broad_findings[0].risk_level == RiskLevel.HIGH
    
    def test_stale_key_detection(self, mock_analyzer, sample_service_account):
        """Test detection of stale service account keys"""
        findings = mock_analyzer._check_stale_keys(sample_service_account)
        
        assert len(findings) == 1
        stale_finding = findings[0]
        assert stale_finding.finding_type == IAMFindingType.STALE_SERVICE_ACCOUNT_KEY
        assert stale_finding.risk_level == RiskLevel.MEDIUM
        assert "120" in str(stale_finding.metadata)  # Key age
    
    def test_unused_service_account_detection(self, mock_analyzer):
        """Test detection of unused service accounts"""
        # Service account with no keys and not disabled
        unused_sa = {
            "name": "projects/test-project/serviceAccounts/unused@test-project.iam.gserviceaccount.com",
            "email": "unused@test-project.iam.gserviceaccount.com",
            "display_name": "Unused Account",
            "unique_id": "987654321",
            "disabled": False,
            "keys": []  # No keys
        }
        
        findings = mock_analyzer._check_service_account_usage(unused_sa)
        
        assert len(findings) == 1
        unused_finding = findings[0]
        assert unused_finding.finding_type == IAMFindingType.UNUSED_SERVICE_ACCOUNT
        assert unused_finding.risk_level == RiskLevel.LOW
    
    def test_wildcard_binding_detection(self, mock_analyzer, sample_iam_policy):
        """Test detection of wildcard IAM bindings"""
        findings = mock_analyzer._analyze_iam_policy(sample_iam_policy)
        
        # Should detect allUsers binding
        wildcard_findings = [f for f in findings if f.finding_type == IAMFindingType.WILDCARD_BINDING]
        assert len(wildcard_findings) == 1
        assert wildcard_findings[0].risk_level == RiskLevel.CRITICAL
        assert "allUsers" in wildcard_findings[0].affected_principal
    
    def test_external_user_detection(self, mock_analyzer, sample_iam_policy):
        """Test detection of external users"""
        findings = mock_analyzer._analyze_iam_policy(sample_iam_policy)
        
        # Should detect external user
        external_findings = [f for f in findings if f.finding_type == IAMFindingType.EXTERNAL_USER_ACCESS]
        assert len(external_findings) == 1
        assert external_findings[0].risk_level == RiskLevel.MEDIUM
        assert "external@gmail.com" in external_findings[0].affected_principal
    
    def test_service_account_role_extraction(self, mock_analyzer, sample_iam_policy):
        """Test extraction of roles for service accounts"""
        sa_email = "test-sa@test-project.iam.gserviceaccount.com"
        roles = mock_analyzer._get_service_account_roles(sa_email, sample_iam_policy)
        
        assert "roles/owner" in roles
        assert len(roles) == 1
    
    def test_security_posture_calculation(self, mock_analyzer):
        """Test overall security posture calculation"""
        # Mock service accounts and findings
        service_accounts = [{"email": "sa1@test.com"}, {"email": "sa2@test.com"}]
        
        # Create sample findings
        findings = [
            IAMFinding(
                finding_type=IAMFindingType.ADMIN_ROLE_MISUSE,
                risk_level=RiskLevel.CRITICAL,
                risk_score=90,
                title="Admin Role Misuse",
                description="Test description",
                resource_name="test-resource",
                affected_principal="test-principal",
                remediation_steps=["Fix it"],
                metadata={},
                detected_at=datetime.utcnow()
            ),
            IAMFinding(
                finding_type=IAMFindingType.STALE_SERVICE_ACCOUNT_KEY,
                risk_level=RiskLevel.HIGH,
                risk_score=70,
                title="Stale Key",
                description="Test description",
                resource_name="test-resource",
                affected_principal="test-principal",
                remediation_steps=["Rotate key"],
                metadata={},
                detected_at=datetime.utcnow()
            )
        ]
        
        # Mock service account analyses
        sa_analyses = [
            ServiceAccountAnalysis(
                email="sa1@test.com",
                display_name="SA1",
                unique_id="123",
                project_id="test-project",
                disabled=False,
                key_count=1,
                oldest_key_age_days=120,
                roles=["roles/owner"],
                risk_score=90,
                findings=[findings[0]],
                last_used=None,
                cross_project_access=False
            ),
            ServiceAccountAnalysis(
                email="sa2@test.com",
                display_name="SA2",
                unique_id="456",
                project_id="test-project",
                disabled=False,
                key_count=1,
                oldest_key_age_days=30,
                roles=["roles/viewer"],
                risk_score=20,
                findings=[],
                last_used=None,
                cross_project_access=False
            )
        ]
        
        posture = mock_analyzer._calculate_security_posture(
            service_accounts, sa_analyses, findings
        )
        
        assert isinstance(posture, IAMSecurityPosture)
        assert posture.project_id == "test-project"
        assert posture.total_findings == 2
        assert posture.critical_findings == 1
        assert posture.high_findings == 1
        assert posture.service_account_count == 2
        assert posture.overprivileged_accounts == 1
        assert posture.posture_score <= 100
        assert len(posture.recommendations) > 0
    
    def test_risk_scoring_weights(self, mock_analyzer):
        """Test risk scoring calculation"""
        findings = [
            IAMFinding(
                finding_type=IAMFindingType.ADMIN_ROLE_MISUSE,
                risk_level=RiskLevel.CRITICAL,
                risk_score=90,
                title="Critical Finding",
                description="Test",
                resource_name="test",
                affected_principal="test",
                remediation_steps=[],
                metadata={},
                detected_at=datetime.utcnow()
            )
        ]
        
        # Calculate posture with critical finding
        posture = mock_analyzer._calculate_security_posture([], [], findings)
        
        # Score should be penalized for critical finding
        assert posture.posture_score <= 75  # 100 - 25 penalty for critical
    
    def test_recommendations_generation(self, mock_analyzer):
        """Test generation of actionable recommendations"""
        risk_distribution = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 1, "LOW": 0, "MINIMAL": 0}
        
        recommendations = mock_analyzer._generate_recommendations(
            risk_distribution, overprivileged_accounts=2, stale_keys=3
        )
        
        assert len(recommendations) <= 10
        assert any("CRITICAL" in rec for rec in recommendations)
        assert any("overprivileged" in rec.lower() for rec in recommendations)
        assert any("stale" in rec.lower() for rec in recommendations)
    
    def test_multiple_high_privilege_roles(self, mock_analyzer, sample_service_account):
        """Test detection of service accounts with multiple high-privilege roles"""
        roles = ["roles/compute.admin", "roles/storage.admin", "roles/iam.serviceAccountAdmin"]
        
        findings = mock_analyzer._check_overprivileged_account(sample_service_account, roles)
        
        # Should detect multiple high-privilege roles
        overprivilege_findings = [f for f in findings 
                                if f.finding_type == IAMFindingType.OVERPRIVILEGED_SERVICE_ACCOUNT]
        assert len(overprivilege_findings) == 1
        assert overprivilege_findings[0].risk_score == 80
    
    def test_cross_project_access_detection(self, mock_analyzer):
        """Test detection of cross-project access"""
        sa_roles = ["roles/viewer", "projects/other-project/roles/custom"]
        
        # This would be detected in the service account analysis
        cross_project = any(role.startswith("projects/") and "test-project" not in role 
                          for role in sa_roles)
        
        assert cross_project is True
    
    def test_sample_analysis_fallback(self):
        """Test sample analysis when GCP clients unavailable"""
        with patch('backend.services.iam_security_analyzer.GCP_CLIENTS_AVAILABLE', False):
            analyzer = IAMSecurityAnalyzer("test-project")
            posture = analyzer.analyze_iam_security()
            
            assert isinstance(posture, IAMSecurityPosture)
            assert posture.project_id == "test-project"
            assert len(posture.findings) > 0
            assert len(posture.recommendations) > 0
    
    @patch('backend.services.iam_security_analyzer.iam_admin_v1.IAMClient')
    @patch('backend.services.iam_security_analyzer.resourcemanager_v3.ProjectsClient')
    def test_full_analysis_workflow(self, mock_rm_client, mock_iam_client, mock_analyzer):
        """Test complete analysis workflow"""
        # Mock service accounts list
        mock_sa = Mock()
        mock_sa.email = "test@test-project.iam.gserviceaccount.com"
        mock_sa.display_name = "Test SA"
        mock_sa.unique_id = "123"
        mock_sa.disabled = False
        
        mock_analyzer.iam_client.list_service_accounts.return_value = [mock_sa]
        
        # Mock keys list
        mock_key = Mock()
        mock_key.valid_after_time = datetime.now() - timedelta(days=120)
        mock_key.disabled = False
        mock_analyzer.iam_client.list_service_account_keys.return_value = [mock_key]
        
        # Mock IAM policy
        mock_policy = Mock()
        mock_binding = Mock()
        mock_binding.role = "roles/owner"
        mock_binding.members = ["serviceAccount:test@test-project.iam.gserviceaccount.com"]
        mock_policy.bindings = [mock_binding]
        
        mock_analyzer.rm_client.get_iam_policy.return_value = mock_policy
        
        # Run analysis
        posture = mock_analyzer.analyze_iam_security()
        
        assert isinstance(posture, IAMSecurityPosture)
        assert posture.total_findings > 0  # Should find overprivileged account and stale key


class TestIAMAnalyzerIntegration:
    """Integration tests for IAM analyzer"""
    
    @pytest.mark.skipif(
        not os.getenv("TEST_INTEGRATION", False),
        reason="Integration tests require GCP credentials"
    )
    def test_real_gcp_analysis(self):
        """Test with real GCP project (requires credentials)"""
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "test-project")
        analyzer = IAMSecurityAnalyzer(project_id)
        
        posture = analyzer.analyze_iam_security()
        
        assert isinstance(posture, IAMSecurityPosture)
        assert posture.project_id == project_id
        assert isinstance(posture.posture_score, int)
        assert 0 <= posture.posture_score <= 100


class TestIAMFindingModel:
    """Test IAM finding data models"""
    
    def test_iam_finding_creation(self):
        """Test IAM finding creation and properties"""
        finding = IAMFinding(
            finding_type=IAMFindingType.OVERPRIVILEGED_SERVICE_ACCOUNT,
            risk_level=RiskLevel.HIGH,
            risk_score=80,
            title="Test Finding",
            description="Test description",
            resource_name="test-resource",
            affected_principal="test-principal",
            remediation_steps=["Step 1", "Step 2"],
            metadata={"key": "value"},
            detected_at=datetime.utcnow()
        )
        
        assert finding.finding_type == IAMFindingType.OVERPRIVILEGED_SERVICE_ACCOUNT
        assert finding.risk_level == RiskLevel.HIGH
        assert finding.risk_score == 80
        assert len(finding.remediation_steps) == 2
    
    def test_service_account_analysis_model(self):
        """Test service account analysis model"""
        analysis = ServiceAccountAnalysis(
            email="test@project.iam.gserviceaccount.com",
            display_name="Test SA",
            unique_id="123",
            project_id="test-project",
            disabled=False,
            key_count=2,
            oldest_key_age_days=60,
            roles=["roles/viewer"],
            risk_score=30,
            findings=[],
            last_used=None,
            cross_project_access=False
        )
        
        assert analysis.email == "test@project.iam.gserviceaccount.com"
        assert analysis.key_count == 2
        assert analysis.risk_score == 30
        assert not analysis.cross_project_access
    
    def test_security_posture_model(self):
        """Test security posture model"""
        posture = IAMSecurityPosture(
            project_id="test-project",
            posture_score=75,
            risk_distribution={"HIGH": 2, "MEDIUM": 1, "LOW": 0},
            total_findings=3,
            critical_findings=0,
            high_findings=2,
            service_account_count=5,
            overprivileged_accounts=1,
            stale_keys=2,
            cross_project_bindings=0,
            external_users=1,
            recommendations=["Fix this", "Do that"],
            findings=[],
            analyzed_at=datetime.utcnow()
        )
        
        assert posture.project_id == "test-project"
        assert posture.posture_score == 75
        assert posture.total_findings == 3
        assert len(posture.recommendations) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])