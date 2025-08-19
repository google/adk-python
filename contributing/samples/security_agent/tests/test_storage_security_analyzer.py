"""
Tests for Enhanced Storage Security Analyzer (STORY-004)

Comprehensive test suite for storage security analysis including
public bucket detection, encryption validation, and compliance scoring.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from backend.services.storage_security_analyzer import (
    StorageSecurityAnalyzer,
    StorageFinding,
    StorageFindingType,
    StorageRiskLevel,
    BucketSecurityAnalysis,
    StorageSecurityPosture
)


class TestStorageSecurityAnalyzer:
    """Test suite for StorageSecurityAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        return StorageSecurityAnalyzer("test-project")
    
    @pytest.fixture
    def mock_bucket(self):
        """Create mock bucket for testing"""
        bucket = Mock()
        bucket.name = "test-bucket"
        bucket.location = "US-CENTRAL1"
        bucket.storage_class = "STANDARD"
        bucket.versioning_enabled = False
        bucket.lifecycle_rules = []
        bucket.logging = None
        bucket.cors = []
        bucket.default_kms_key_name = None
        bucket.retention_policy = None
        
        # Mock IAM configuration
        iam_config = Mock()
        iam_config.uniform_bucket_level_access_enabled = False
        iam_config.public_access_prevention = "inherited"
        bucket.iam_configuration = iam_config
        
        return bucket
    
    @pytest.fixture
    def mock_public_bucket(self, mock_bucket):
        """Create mock public bucket"""
        mock_bucket.name = "public-test-bucket"
        
        # Mock IAM policy with public access
        policy = Mock()
        policy.bindings = [
            {
                'role': 'roles/storage.objectViewer',
                'members': ['allUsers']
            }
        ]
        mock_bucket.get_iam_policy.return_value = policy
        
        return mock_bucket
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.project_id == "test-project"
        assert analyzer.COMPLIANCE_FRAMEWORKS is not None
        assert "SOC2" in analyzer.COMPLIANCE_FRAMEWORKS
        assert "HIPAA" in analyzer.COMPLIANCE_FRAMEWORKS
    
    def test_check_public_access_no_public_access(self, analyzer, mock_bucket):
        """Test public access check with no public access"""
        # Mock IAM policy with no public access
        policy = Mock()
        policy.bindings = [
            {
                'role': 'roles/storage.objectViewer',
                'members': ['user:test@example.com']
            }
        ]
        mock_bucket.get_iam_policy.return_value = policy
        
        findings = analyzer._check_public_access(mock_bucket)
        
        # Should have finding for public access prevention disabled
        assert len(findings) == 1
        assert findings[0].finding_type == StorageFindingType.PUBLIC_ACCESS_PREVENTION_DISABLED
    
    def test_check_public_access_with_public_read(self, analyzer, mock_public_bucket):
        """Test public access check with public read access"""
        findings = analyzer._check_public_access(mock_public_bucket)
        
        # Should have findings for public read access and prevention disabled
        assert len(findings) >= 1
        
        public_read_findings = [f for f in findings if f.finding_type == StorageFindingType.PUBLIC_BUCKET_READ]
        assert len(public_read_findings) == 1
        assert public_read_findings[0].risk_level == StorageRiskLevel.HIGH
        assert public_read_findings[0].risk_score == 80
    
    def test_check_public_access_with_public_write(self, analyzer, mock_bucket):
        """Test public access check with public write access"""
        # Mock IAM policy with public write access
        policy = Mock()
        policy.bindings = [
            {
                'role': 'roles/storage.objectAdmin',
                'members': ['allUsers']
            }
        ]
        mock_bucket.get_iam_policy.return_value = policy
        
        findings = analyzer._check_public_access(mock_bucket)
        
        public_write_findings = [f for f in findings if f.finding_type == StorageFindingType.PUBLIC_BUCKET_WRITE]
        assert len(public_write_findings) == 1
        assert public_write_findings[0].risk_level == StorageRiskLevel.CRITICAL
        assert public_write_findings[0].risk_score == 95
    
    def test_check_public_access_all_authenticated_users(self, analyzer, mock_bucket):
        """Test public access check with allAuthenticatedUsers"""
        # Mock IAM policy with allAuthenticatedUsers
        policy = Mock()
        policy.bindings = [
            {
                'role': 'roles/storage.objectViewer',
                'members': ['allAuthenticatedUsers']
            }
        ]
        mock_bucket.get_iam_policy.return_value = policy
        
        findings = analyzer._check_public_access(mock_bucket)
        
        auth_users_findings = [f for f in findings if f.finding_type == StorageFindingType.PUBLIC_BUCKET_NO_AUTH]
        assert len(auth_users_findings) == 1
        assert auth_users_findings[0].risk_level == StorageRiskLevel.HIGH
        assert auth_users_findings[0].risk_score == 75
    
    def test_check_encryption_no_cmek(self, analyzer, mock_bucket):
        """Test encryption check with no customer-managed keys"""
        findings = analyzer._check_encryption(mock_bucket)
        
        assert len(findings) == 1
        assert findings[0].finding_type == StorageFindingType.MISSING_ENCRYPTION
        assert findings[0].risk_level == StorageRiskLevel.HIGH
        assert findings[0].risk_score == 70
    
    def test_check_encryption_with_cmek(self, analyzer, mock_bucket):
        """Test encryption check with customer-managed keys"""
        mock_bucket.default_kms_key_name = "projects/test/locations/us/keyRings/ring/cryptoKeys/key"
        
        findings = analyzer._check_encryption(mock_bucket)
        
        assert len(findings) == 0
    
    def test_check_lifecycle_policies_missing(self, analyzer, mock_bucket):
        """Test lifecycle policy check with missing policies"""
        findings = analyzer._check_lifecycle_policies(mock_bucket)
        
        assert len(findings) == 1
        assert findings[0].finding_type == StorageFindingType.NO_LIFECYCLE_POLICY
        assert findings[0].risk_level == StorageRiskLevel.MEDIUM
        assert findings[0].risk_score == 40
    
    def test_check_lifecycle_policies_present(self, analyzer, mock_bucket):
        """Test lifecycle policy check with policies present"""
        mock_bucket.lifecycle_rules = [{"action": {"type": "Delete"}, "condition": {"age": 30}}]
        
        findings = analyzer._check_lifecycle_policies(mock_bucket)
        
        assert len(findings) == 0
    
    def test_check_access_controls_uniform_disabled(self, analyzer, mock_bucket):
        """Test access controls check with uniform access disabled"""
        findings = analyzer._check_access_controls(mock_bucket)
        
        assert len(findings) == 1
        assert findings[0].finding_type == StorageFindingType.UNIFORM_BUCKET_ACCESS_DISABLED
        assert findings[0].risk_level == StorageRiskLevel.MEDIUM
        assert findings[0].risk_score == 45
    
    def test_check_access_controls_uniform_enabled(self, analyzer, mock_bucket):
        """Test access controls check with uniform access enabled"""
        mock_bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        
        findings = analyzer._check_access_controls(mock_bucket)
        
        assert len(findings) == 0
    
    def test_check_compliance_configurations_versioning_disabled(self, analyzer, mock_bucket):
        """Test compliance check with versioning disabled"""
        findings = analyzer._check_compliance_configurations(mock_bucket)
        
        versioning_findings = [f for f in findings if f.finding_type == StorageFindingType.MISSING_VERSIONING]
        assert len(versioning_findings) == 1
        assert versioning_findings[0].risk_level == StorageRiskLevel.MEDIUM
        assert versioning_findings[0].risk_score == 35
    
    def test_check_compliance_configurations_logging_disabled(self, analyzer, mock_bucket):
        """Test compliance check with logging disabled"""
        findings = analyzer._check_compliance_configurations(mock_bucket)
        
        logging_findings = [f for f in findings if f.finding_type == StorageFindingType.LOGGING_DISABLED]
        assert len(logging_findings) == 1
        assert logging_findings[0].risk_level == StorageRiskLevel.MEDIUM
        assert logging_findings[0].risk_score == 40
    
    def test_check_sensitive_data_patterns(self, analyzer, mock_bucket):
        """Test sensitive data pattern detection"""
        mock_bucket.name = "secret-data-bucket"
        
        findings = analyzer._check_sensitive_data(mock_bucket)
        
        assert len(findings) == 1
        assert findings[0].finding_type == StorageFindingType.SENSITIVE_DATA_EXPOSURE
        assert findings[0].risk_level == StorageRiskLevel.HIGH
        assert findings[0].risk_score == 65
    
    def test_check_bucket_naming_generic(self, analyzer, mock_bucket):
        """Test bucket naming check with generic name"""
        mock_bucket.name = "test-bucket-123"
        
        findings = analyzer._check_bucket_naming(mock_bucket)
        
        assert len(findings) == 1
        assert findings[0].finding_type == StorageFindingType.BUCKET_NAMING_VIOLATION
        assert findings[0].risk_level == StorageRiskLevel.LOW
        assert findings[0].risk_score == 15
    
    def test_classify_data_high_sensitivity(self, analyzer, mock_bucket):
        """Test data classification for high sensitivity"""
        mock_bucket.name = "personal-data-bucket"
        
        classification = analyzer._classify_data(mock_bucket)
        
        assert classification == "HIGH"
    
    def test_classify_data_medium_sensitivity(self, analyzer, mock_bucket):
        """Test data classification for medium sensitivity"""
        mock_bucket.name = "internal-business-bucket"
        
        classification = analyzer._classify_data(mock_bucket)
        
        assert classification == "MEDIUM"
    
    def test_classify_data_low_sensitivity(self, analyzer, mock_bucket):
        """Test data classification for low sensitivity"""
        mock_bucket.name = "public-website-assets"
        
        classification = analyzer._classify_data(mock_bucket)
        
        assert classification == "LOW"
    
    def test_likely_needs_retention_backup_bucket(self, analyzer, mock_bucket):
        """Test retention policy requirement for backup bucket"""
        mock_bucket.name = "backup-data-bucket"
        
        needs_retention = analyzer._likely_needs_retention(mock_bucket)
        
        assert needs_retention is True
    
    def test_likely_needs_retention_regular_bucket(self, analyzer, mock_bucket):
        """Test retention policy requirement for regular bucket"""
        mock_bucket.name = "app-data-bucket"
        
        needs_retention = analyzer._likely_needs_retention(mock_bucket)
        
        assert needs_retention is False
    
    def test_calculate_compliance_scores(self, analyzer):
        """Test compliance score calculation"""
        # Create mock bucket analyses
        bucket_analyses = [
            Mock(
                encryption_type="CUSTOMER_MANAGED",
                uniform_bucket_access=True,
                public_access_prevention=True,
                logging_enabled=True,
                retention_policy={"retention_period": 30}
            ),
            Mock(
                encryption_type="GOOGLE_MANAGED",
                uniform_bucket_access=False,
                public_access_prevention=False,
                logging_enabled=False,
                retention_policy=None
            )
        ]
        
        compliance_scores = analyzer._calculate_compliance_scores(bucket_analyses)
        
        assert "SOC2" in compliance_scores
        assert "HIPAA" in compliance_scores
        assert "PCI_DSS" in compliance_scores
        assert "GDPR" in compliance_scores
        assert "ISO27001" in compliance_scores
        
        # First bucket should score higher than second
        assert all(0 <= score <= 100 for score in compliance_scores.values())
    
    def test_generate_storage_recommendations(self, analyzer):
        """Test storage recommendation generation"""
        risk_distribution = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 1, "LOW": 0, "MINIMAL": 0}
        
        recommendations = analyzer._generate_storage_recommendations(
            risk_distribution, public_buckets=1, encrypted_buckets=2, total_buckets=5
        )
        
        assert len(recommendations) > 0
        assert any("CRITICAL" in rec for rec in recommendations)
        assert any("public" in rec.lower() for rec in recommendations)
        assert any("encryption" in rec.lower() for rec in recommendations)
    
    @patch('backend.services.storage_security_analyzer.storage.Client')
    def test_analyze_storage_security_with_real_client(self, mock_storage_client, analyzer):
        """Test full storage security analysis with mocked GCP client"""
        # Mock storage client and buckets
        mock_client = Mock()
        mock_storage_client.return_value = mock_client
        
        mock_bucket1 = Mock()
        mock_bucket1.name = "test-bucket-1"
        mock_bucket1.location = "US"
        mock_bucket1.storage_class = "STANDARD"
        mock_bucket1.versioning_enabled = False
        mock_bucket1.lifecycle_rules = []
        mock_bucket1.logging = None
        mock_bucket1.cors = []
        mock_bucket1.default_kms_key_name = None
        mock_bucket1.retention_policy = None
        
        # Mock IAM configuration
        iam_config = Mock()
        iam_config.uniform_bucket_level_access_enabled = False
        iam_config.public_access_prevention = "inherited"
        mock_bucket1.iam_configuration = iam_config
        
        # Mock IAM policy
        policy = Mock()
        policy.bindings = []
        mock_bucket1.get_iam_policy.return_value = policy
        
        # Mock blob listing
        mock_bucket1.list_blobs.return_value = []
        
        mock_client.list_buckets.return_value = [mock_bucket1]
        
        # Update analyzer to use mocked client
        analyzer.storage_client = mock_client
        
        posture = analyzer.analyze_storage_security()
        
        assert isinstance(posture, StorageSecurityPosture)
        assert posture.project_id == "test-project"
        assert posture.total_buckets >= 1
        assert isinstance(posture.posture_score, int)
        assert 0 <= posture.posture_score <= 100
        assert len(posture.findings) >= 0
        assert len(posture.recommendations) > 0
    
    def test_generate_sample_analysis(self, analyzer):
        """Test sample analysis generation when GCP unavailable"""
        posture = analyzer._generate_sample_analysis()
        
        assert isinstance(posture, StorageSecurityPosture)
        assert posture.project_id == "test-project"
        assert posture.posture_score == 65
        assert posture.total_buckets == 5
        assert posture.public_buckets == 1
        assert posture.encrypted_buckets == 2
        assert len(posture.findings) == 2
        assert len(posture.recommendations) >= 4
        assert "SOC2" in posture.compliance_status
    
    def test_storage_finding_dataclass(self):
        """Test StorageFinding dataclass"""
        finding = StorageFinding(
            finding_type=StorageFindingType.PUBLIC_BUCKET_READ,
            risk_level=StorageRiskLevel.HIGH,
            risk_score=80,
            title="Test Finding",
            description="Test description",
            bucket_name="test-bucket",
            object_name=None,
            remediation_steps=["Step 1", "Step 2"],
            compliance_frameworks=["SOC2"],
            metadata={"test": "data"},
            detected_at=datetime.utcnow()
        )
        
        assert finding.finding_type == StorageFindingType.PUBLIC_BUCKET_READ
        assert finding.risk_level == StorageRiskLevel.HIGH
        assert finding.risk_score == 80
        assert finding.bucket_name == "test-bucket"
        assert len(finding.remediation_steps) == 2
    
    def test_bucket_security_analysis_dataclass(self):
        """Test BucketSecurityAnalysis dataclass"""
        analysis = BucketSecurityAnalysis(
            bucket_name="test-bucket",
            project_id="test-project",
            location="US",
            storage_class="STANDARD",
            is_public=False,
            public_access_type=None,
            encryption_type="GOOGLE_MANAGED",
            kms_key_name=None,
            versioning_enabled=False,
            lifecycle_configured=False,
            logging_enabled=False,
            cors_configured=False,
            uniform_bucket_access=False,
            public_access_prevention=False,
            retention_policy=None,
            risk_score=45,
            findings=[],
            object_count=0,
            total_size_bytes=0,
            last_modified=None,
            data_classification="UNKNOWN"
        )
        
        assert analysis.bucket_name == "test-bucket"
        assert analysis.is_public is False
        assert analysis.encryption_type == "GOOGLE_MANAGED"
        assert analysis.risk_score == 45
    
    def test_storage_security_posture_dataclass(self):
        """Test StorageSecurityPosture dataclass"""
        posture = StorageSecurityPosture(
            project_id="test-project",
            posture_score=75,
            risk_distribution={"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 1, "MINIMAL": 0},
            total_buckets=5,
            public_buckets=0,
            encrypted_buckets=3,
            compliant_buckets=4,
            findings=[],
            recommendations=["Enable encryption", "Review access"],
            compliance_status={"SOC2": 80.0, "HIPAA": 75.0},
            analyzed_at=datetime.utcnow()
        )
        
        assert posture.project_id == "test-project"
        assert posture.posture_score == 75
        assert posture.total_buckets == 5
        assert len(posture.recommendations) == 2


class TestStorageRiskLevels:
    """Test risk level enum"""
    
    def test_risk_level_values(self):
        """Test risk level enum values"""
        assert StorageRiskLevel.MINIMAL.value == "MINIMAL"
        assert StorageRiskLevel.LOW.value == "LOW"
        assert StorageRiskLevel.MEDIUM.value == "MEDIUM"
        assert StorageRiskLevel.HIGH.value == "HIGH"
        assert StorageRiskLevel.CRITICAL.value == "CRITICAL"


class TestStorageFindingTypes:
    """Test finding type enum"""
    
    def test_finding_type_values(self):
        """Test finding type enum values"""
        assert StorageFindingType.PUBLIC_BUCKET_NO_AUTH.value == "PUBLIC_BUCKET_NO_AUTH"
        assert StorageFindingType.PUBLIC_BUCKET_READ.value == "PUBLIC_BUCKET_READ"
        assert StorageFindingType.PUBLIC_BUCKET_WRITE.value == "PUBLIC_BUCKET_WRITE"
        assert StorageFindingType.MISSING_ENCRYPTION.value == "MISSING_ENCRYPTION"
        assert StorageFindingType.NO_LIFECYCLE_POLICY.value == "NO_LIFECYCLE_POLICY"


if __name__ == "__main__":
    pytest.main([__file__])