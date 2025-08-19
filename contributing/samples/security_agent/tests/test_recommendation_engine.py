"""
Tests for Enhanced Recommendation Engine (STORY-007)

Comprehensive test suite for recommendation engine including CVSS-based prioritization,
business impact scoring, and automation script generation.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from backend.services.recommendation_engine import (
    RecommendationEngine,
    Recommendation,
    RecommendationSummary,
    RecommendationCategory,
    BusinessImpact,
    Priority
)


class TestRecommendationEngine:
    """Test suite for RecommendationEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create recommendation engine instance for testing"""
        return RecommendationEngine("test-project")
    
    @pytest.fixture
    def mock_security_findings(self):
        """Mock security findings"""
        return [
            {
                "id": "sec-001",
                "severity": "CRITICAL",
                "cvss_score": 9.5,
                "description": "Critical vulnerability detected",
                "resource": "compute-instance-1",
                "remediation": ["Patch immediately", "Restart service"],
                "compliance_frameworks": ["SOC2", "ISO27001"]
            },
            {
                "id": "sec-002", 
                "severity": "HIGH",
                "cvss_score": 7.8,
                "description": "High severity issue found",
                "resource": "database-instance-1",
                "remediation": ["Update configuration", "Apply patches"],
                "compliance_frameworks": ["HIPAA"]
            }
        ]
    
    @pytest.fixture
    def mock_iam_findings(self):
        """Mock IAM findings"""
        return [
            {
                "type": "ADMIN_ROLE_MISUSE",
                "risk_level": "CRITICAL",
                "risk_score": 90,
                "title": "Service Account with Owner Role",
                "description": "Service account has admin roles",
                "resource_name": "projects/test/serviceAccounts/admin-sa@test.iam.gserviceaccount.com",
                "affected_principal": "admin-sa@test.iam.gserviceaccount.com",
                "remediation_steps": ["Review admin role necessity", "Implement least privilege"]
            },
            {
                "type": "STALE_SERVICE_ACCOUNT_KEY",
                "risk_level": "MEDIUM",
                "risk_score": 60,
                "title": "Stale Service Account Key",
                "description": "Service account key is older than 90 days",
                "resource_name": "projects/test/serviceAccounts/app-sa@test.iam.gserviceaccount.com",
                "affected_principal": "app-sa@test.iam.gserviceaccount.com",
                "remediation_steps": ["Rotate service account key", "Implement automated rotation"]
            }
        ]
    
    @pytest.fixture
    def mock_storage_findings(self):
        """Mock storage findings"""
        return [
            {
                "type": "PUBLIC_BUCKET_READ",
                "risk_level": "HIGH",
                "risk_score": 80,
                "title": "Public Read Access",
                "description": "Bucket allows public read access",
                "bucket_name": "public-test-bucket",
                "remediation_steps": ["Remove allUsers from IAM policy", "Enable public access prevention"],
                "compliance_frameworks": ["SOC2", "GDPR"]
            }
        ]
    
    def test_engine_initialization(self, engine):
        """Test recommendation engine initialization"""
        assert engine.project_id == "test-project"
        assert engine.backend_base_url == "http://localhost:8000"
        assert hasattr(engine, 'BUSINESS_IMPACT_WEIGHTS')
        assert hasattr(engine, 'CVSS_PRIORITY_MAP')
    
    def test_cvss_priority_mapping(self, engine):
        """Test CVSS score to priority mapping"""
        assert engine._calculate_priority(9.5, 95) == Priority.P0
        assert engine._calculate_priority(8.0, 80) == Priority.P1
        assert engine._calculate_priority(6.0, 60) == Priority.P2
        assert engine._calculate_priority(3.0, 30) == Priority.P3
        assert engine._calculate_priority(0.5, 5) == Priority.P4
    
    def test_due_date_calculation(self, engine):
        """Test due date calculation based on priority"""
        now = datetime.utcnow()
        
        p0_due = engine._calculate_due_date(Priority.P0)
        assert p0_due <= now + timedelta(hours=4.1)
        
        p1_due = engine._calculate_due_date(Priority.P1)
        assert p1_due <= now + timedelta(days=1.1)
        
        p2_due = engine._calculate_due_date(Priority.P2)
        assert p2_due <= now + timedelta(days=7.1)
    
    def test_business_impact_score_calculation(self, engine):
        """Test business impact score calculation"""
        finding_critical = {"severity": "CRITICAL", "public_exposure": True}
        score_critical = engine._calculate_business_impact_score(finding_critical)
        assert score_critical >= 65  # 40 + 25
        
        finding_low = {"severity": "LOW"}
        score_low = engine._calculate_business_impact_score(finding_low)
        assert score_low == 10
    
    def test_severity_to_business_impact(self, engine):
        """Test severity to business impact conversion"""
        assert engine._severity_to_business_impact("CRITICAL") == BusinessImpact.CRITICAL
        assert engine._severity_to_business_impact("HIGH") == BusinessImpact.HIGH
        assert engine._severity_to_business_impact("MEDIUM") == BusinessImpact.MEDIUM
        assert engine._severity_to_business_impact("LOW") == BusinessImpact.LOW
        assert engine._severity_to_business_impact("UNKNOWN") == BusinessImpact.MINIMAL
    
    def test_risk_score_to_cvss_conversion(self, engine):
        """Test risk score to CVSS conversion"""
        assert engine._risk_score_to_cvss(100) == 10.0
        assert engine._risk_score_to_cvss(75) == 7.5
        assert engine._risk_score_to_cvss(50) == 5.0
        assert engine._risk_score_to_cvss(0) == 0.0
    
    def test_effort_estimation(self, engine):
        """Test effort estimation for different finding types"""
        # Security effort estimation
        assert engine._estimate_security_effort("CRITICAL") == 8.0
        assert engine._estimate_security_effort("HIGH") == 4.0
        assert engine._estimate_security_effort("MEDIUM") == 2.0
        assert engine._estimate_security_effort("LOW") == 1.0
        
        # IAM effort estimation
        assert engine._estimate_iam_effort("ADMIN_ROLE_MISUSE") == 3.0
        assert engine._estimate_iam_effort("OVERPRIVILEGED_ACCOUNT") == 2.0
        assert engine._estimate_iam_effort("STALE_KEY_ROTATION") == 0.5
        
        # Storage effort estimation
        assert engine._estimate_storage_effort("PUBLIC_BUCKET_WRITE") == 1.0
        assert engine._estimate_storage_effort("MISSING_ENCRYPTION") == 2.0
        assert engine._estimate_storage_effort("NO_LIFECYCLE_POLICY") == 1.5
    
    def test_cost_impact_estimation(self, engine):
        """Test cost impact estimation"""
        critical_cost = engine._estimate_security_cost_impact("CRITICAL")
        assert "$100K+" in critical_cost
        
        high_cost = engine._estimate_security_cost_impact("HIGH")
        assert "$50K+" in high_cost
        
        storage_lifecycle_cost = engine._estimate_storage_cost_impact("NO_LIFECYCLE_POLICY")
        assert "savings" in storage_lifecycle_cost.lower()
    
    def test_generate_security_recommendations(self, engine, mock_security_findings):
        """Test security recommendation generation"""
        recommendations = engine._generate_security_recommendations(mock_security_findings)
        
        assert len(recommendations) == 2
        
        # Check critical finding
        critical_rec = recommendations[0]
        assert critical_rec.cvss_score == 9.5
        assert critical_rec.category == RecommendationCategory.SECURITY
        assert critical_rec.business_impact == BusinessImpact.CRITICAL
        assert "sec-001" in critical_rec.id
        
        # Check high finding
        high_rec = recommendations[1]
        assert high_rec.cvss_score == 7.8
        assert high_rec.business_impact == BusinessImpact.HIGH
        assert len(high_rec.remediation_steps) > 0
    
    def test_generate_iam_recommendations(self, engine, mock_iam_findings):
        """Test IAM recommendation generation"""
        recommendations = engine._generate_iam_recommendations(mock_iam_findings)
        
        assert len(recommendations) == 2
        
        # Check admin role misuse
        admin_rec = recommendations[0]
        assert admin_rec.category == RecommendationCategory.IAM
        assert admin_rec.cvss_score == 9.0  # 90/10
        assert admin_rec.business_impact == BusinessImpact.CRITICAL
        assert "admin-sa" in admin_rec.id
        
        # Check stale key
        stale_rec = recommendations[1]
        assert stale_rec.cvss_score == 6.0  # 60/10
        assert stale_rec.business_impact == BusinessImpact.MEDIUM
        assert len(stale_rec.remediation_steps) > 0
    
    def test_generate_storage_recommendations(self, engine, mock_storage_findings):
        """Test storage recommendation generation"""
        recommendations = engine._generate_storage_recommendations(mock_storage_findings)
        
        assert len(recommendations) == 1
        
        storage_rec = recommendations[0]
        assert storage_rec.category == RecommendationCategory.STORAGE
        assert storage_rec.cvss_score == 8.0  # 80/10
        assert storage_rec.business_impact == BusinessImpact.HIGH
        assert "public-test-bucket" in storage_rec.id
        assert "gs://public-test-bucket" in storage_rec.affected_resources
    
    def test_generate_general_recommendations(self, engine):
        """Test general recommendation generation"""
        recommendations = engine._generate_general_recommendations()
        
        assert len(recommendations) >= 3
        
        # Check for Security Command Center recommendation
        scc_rec = next((r for r in recommendations if "Security Command Center" in r.title), None)
        assert scc_rec is not None
        assert scc_rec.category == RecommendationCategory.SECURITY
        assert scc_rec.cvss_score == 6.5
        
        # Check for organization policies recommendation
        org_rec = next((r for r in recommendations if "Organization Policies" in r.title), None)
        assert org_rec is not None
        assert org_rec.category == RecommendationCategory.COMPLIANCE
    
    def test_automation_script_generation(self, engine):
        """Test automation script generation"""
        # Test IAM automation
        iam_finding = {
            "type": "STALE_SERVICE_ACCOUNT_KEY",
            "affected_principal": "test-sa@project.iam.gserviceaccount.com",
            "key_id": "12345"
        }
        iam_script = engine._generate_iam_automation(iam_finding)
        assert iam_script is not None
        assert "gcloud iam service-accounts keys create" in iam_script
        assert "gcloud iam service-accounts keys delete" in iam_script
        
        # Test storage automation
        storage_finding = {
            "type": "PUBLIC_BUCKET_READ",
            "bucket_name": "test-bucket"
        }
        storage_script = engine._generate_storage_automation(storage_finding)
        assert storage_script is not None
        assert "gsutil iam ch -d allUsers" in storage_script
        assert "gcloud storage buckets update" in storage_script
    
    @patch('httpx.AsyncClient')
    async def test_collect_findings_success(self, mock_httpx, engine):
        """Test successful finding collection from APIs"""
        # Mock HTTP client
        mock_client = AsyncMock()
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        # Mock API responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "analysis": {
                "findings": [{"type": "TEST_FINDING", "risk_score": 75}]
            }
        }
        mock_client.get.return_value = mock_response
        
        # Test IAM findings collection
        iam_findings = await engine._collect_iam_findings()
        assert len(iam_findings) == 1
        assert iam_findings[0]["type"] == "TEST_FINDING"
    
    @patch('httpx.AsyncClient')
    async def test_collect_findings_failure(self, mock_httpx, engine):
        """Test finding collection failure handling"""
        # Mock HTTP client with failure
        mock_client = AsyncMock()
        mock_httpx.return_value.__aenter__.return_value = mock_client
        mock_client.get.side_effect = Exception("API Error")
        
        # Test that failures are handled gracefully
        security_findings = await engine._collect_security_findings()
        assert security_findings == []
        
        iam_findings = await engine._collect_iam_findings()
        assert iam_findings == []
        
        storage_findings = await engine._collect_storage_findings()
        assert storage_findings == []
    
    def test_generate_summary(self, engine):
        """Test recommendation summary generation"""
        # Create test recommendations
        recommendations = [
            Recommendation(
                id="test-1",
                title="Test Critical",
                description="Critical issue",
                category=RecommendationCategory.SECURITY,
                priority=Priority.P0,
                cvss_score=9.0,
                business_impact=BusinessImpact.CRITICAL,
                business_impact_score=90,
                affected_resources=["resource-1"],
                remediation_steps=["Fix immediately"],
                automation_script=None,
                estimated_effort_hours=4.0,
                cost_impact="$0",
                compliance_frameworks=["SOC2"],
                related_findings=["finding-1"],
                created_at=datetime.utcnow(),
                due_date=datetime.utcnow() + timedelta(hours=4),
                metadata={}
            ),
            Recommendation(
                id="test-2",
                title="Test High",
                description="High priority issue",
                category=RecommendationCategory.IAM,
                priority=Priority.P1,
                cvss_score=7.5,
                business_impact=BusinessImpact.HIGH,
                business_impact_score=75,
                affected_resources=["resource-2"],
                remediation_steps=["Fix soon"],
                automation_script=None,
                estimated_effort_hours=2.0,
                cost_impact="$0",
                compliance_frameworks=["HIPAA"],
                related_findings=["finding-2"],
                created_at=datetime.utcnow(),
                due_date=datetime.utcnow() + timedelta(days=1),
                metadata={}
            )
        ]
        
        summary = engine._generate_summary(recommendations)
        
        assert isinstance(summary, RecommendationSummary)
        assert summary.total_recommendations == 2
        assert summary.critical_count == 2  # P0 and P1 are critical
        assert summary.total_estimated_effort == 6.0  # 4.0 + 2.0
        assert summary.estimated_risk_reduction == 16.5  # 9.0 + 7.5
        
        # Check priority distribution
        assert summary.by_priority["P0"] == 1
        assert summary.by_priority["P1"] == 1
        
        # Check category distribution
        assert summary.by_category["SECURITY"] == 1
        assert summary.by_category["IAM"] == 1
        
        # Check business impact distribution
        assert summary.by_business_impact["CRITICAL"] == 1
        assert summary.by_business_impact["HIGH"] == 1
    
    @patch.object(RecommendationEngine, '_collect_security_findings')
    @patch.object(RecommendationEngine, '_collect_iam_findings')
    @patch.object(RecommendationEngine, '_collect_storage_findings')
    async def test_generate_comprehensive_recommendations(self, mock_storage, mock_iam, mock_security, engine):
        """Test comprehensive recommendation generation"""
        # Mock finding collection
        mock_security.return_value = [{"id": "sec-1", "severity": "HIGH", "cvss_score": 7.0}]
        mock_iam.return_value = [{"type": "ADMIN_ROLE", "risk_score": 85}]
        mock_storage.return_value = [{"type": "PUBLIC_BUCKET", "risk_score": 75}]
        
        summary = await engine.generate_comprehensive_recommendations()
        
        assert isinstance(summary, RecommendationSummary)
        assert summary.total_recommendations > 0
        assert len(summary.recommendations) > 0
        
        # Check that recommendations are sorted by priority
        for i in range(len(summary.recommendations) - 1):
            current_priority = summary.recommendations[i].priority.value
            next_priority = summary.recommendations[i + 1].priority.value
            # P0 < P1 < P2 < P3 < P4 in string comparison
            assert current_priority <= next_priority


class TestRecommendationDataClasses:
    """Test recommendation data classes"""
    
    def test_recommendation_creation(self):
        """Test Recommendation dataclass creation"""
        rec = Recommendation(
            id="test-rec",
            title="Test Recommendation",
            description="Test description",
            category=RecommendationCategory.SECURITY,
            priority=Priority.P1,
            cvss_score=7.5,
            business_impact=BusinessImpact.HIGH,
            business_impact_score=75,
            affected_resources=["resource-1", "resource-2"],
            remediation_steps=["Step 1", "Step 2"],
            automation_script="# automation script",
            estimated_effort_hours=3.5,
            cost_impact="$100/month",
            compliance_frameworks=["SOC2", "HIPAA"],
            related_findings=["finding-1"],
            created_at=datetime.utcnow(),
            due_date=datetime.utcnow() + timedelta(days=1),
            metadata={"source": "test"}
        )
        
        assert rec.id == "test-rec"
        assert rec.category == RecommendationCategory.SECURITY
        assert rec.priority == Priority.P1
        assert rec.cvss_score == 7.5
        assert len(rec.affected_resources) == 2
        assert len(rec.remediation_steps) == 2
    
    def test_recommendation_summary_creation(self):
        """Test RecommendationSummary dataclass creation"""
        summary = RecommendationSummary(
            total_recommendations=10,
            by_priority={"P0": 2, "P1": 3, "P2": 5},
            by_category={"SECURITY": 6, "IAM": 4},
            by_business_impact={"CRITICAL": 2, "HIGH": 4, "MEDIUM": 4},
            total_estimated_effort=25.5,
            critical_count=5,
            overdue_count=2,
            estimated_risk_reduction=45.0,
            recommendations=[]
        )
        
        assert summary.total_recommendations == 10
        assert summary.critical_count == 5
        assert summary.total_estimated_effort == 25.5
        assert summary.by_priority["P0"] == 2


class TestRecommendationEnums:
    """Test recommendation enums"""
    
    def test_recommendation_category_enum(self):
        """Test RecommendationCategory enum"""
        assert RecommendationCategory.SECURITY.value == "SECURITY"
        assert RecommendationCategory.IAM.value == "IAM"
        assert RecommendationCategory.STORAGE.value == "STORAGE"
        assert RecommendationCategory.NETWORK.value == "NETWORK"
        assert RecommendationCategory.COST.value == "COST"
        assert RecommendationCategory.COMPLIANCE.value == "COMPLIANCE"
        assert RecommendationCategory.PERFORMANCE.value == "PERFORMANCE"
    
    def test_business_impact_enum(self):
        """Test BusinessImpact enum"""
        assert BusinessImpact.CRITICAL.value == "CRITICAL"
        assert BusinessImpact.HIGH.value == "HIGH"
        assert BusinessImpact.MEDIUM.value == "MEDIUM"
        assert BusinessImpact.LOW.value == "LOW"
        assert BusinessImpact.MINIMAL.value == "MINIMAL"
    
    def test_priority_enum(self):
        """Test Priority enum"""
        assert Priority.P0.value == "P0"
        assert Priority.P1.value == "P1"
        assert Priority.P2.value == "P2"
        assert Priority.P3.value == "P3"
        assert Priority.P4.value == "P4"


if __name__ == "__main__":
    pytest.main([__file__])