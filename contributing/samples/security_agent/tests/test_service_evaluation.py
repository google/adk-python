"""
Test-Driven Development (London School) for Service Evaluation
==============================================================

Following TDD principles:
1. Write failing tests first
2. Use test doubles (mocks, stubs, spies)
3. Test behavior, not implementation
4. Keep tests isolated and fast
"""

import sys
import unittest
from unittest.mock import Mock, MagicMock, patch, call
import json
from pathlib import Path
from datetime import datetime
import sqlite3
import tempfile
import os

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test doubles for external dependencies
class TestDoubles:
    """Test doubles following London School TDD"""
    
    @staticmethod
    def create_mock_credentials():
        """Create mock Google credentials"""
        mock_creds = Mock()
        mock_creds.token = "mock-token"
        mock_creds.expiry = None
        return mock_creds
    
    @staticmethod
    def create_stub_service_profile():
        """Create stub service profile data"""
        return {
            "service_name": "test-service",
            "description": "Test service description",
            "use_cases": ["Test use case 1", "Test use case 2"],
            "release_stage": "GA",
            "is_enabled": True,
            "security_assessment": {
                "iam_permissions": ["test.permission.read", "test.permission.write"],
                "network_exposure": "VPC-native",
                "data_encryption": "AES-256",
                "compliance_certifications": ["SOC2", "ISO27001"],
                "risk_score": 5,
                "risk_profile": {
                    "data_exposure": 4,
                    "misconfiguration": 5,
                    "attack_surface": 6,
                    "compliance_violation": 3
                },
                "threat_model_summary": "Test threat model",
                "data_residency": "us-central1"
            }
        }
    
    @staticmethod
    def create_spy_database():
        """Create spy database to track interactions"""
        spy = Mock()
        spy.execute_calls = []
        spy.commit_calls = []
        
        def track_execute(query, params=None):
            spy.execute_calls.append((query, params))
            return Mock()
        
        def track_commit():
            spy.commit_calls.append(datetime.now())
        
        spy.execute = track_execute
        spy.commit = track_commit
        return spy


class TestServiceEvaluationUnit(unittest.TestCase):
    """Unit tests for Service Evaluation - London School TDD"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.test_doubles = TestDoubles()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_service_analyzer_initializes_with_database(self):
        """Test that GoogleServiceAnalyzer creates database on init"""
        # Arrange
        from backend.services.google_service_analyzer import GoogleServiceAnalyzer
        
        # Act
        with patch('google.auth.default') as mock_auth:
            mock_auth.return_value = (self.test_doubles.create_mock_credentials(), "test-project")
            analyzer = GoogleServiceAnalyzer(db_path=str(self.db_path))
        
        # Assert
        self.assertTrue(self.db_path.exists())
        
        # Verify schema
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        self.assertIn(('service_evaluations',), tables)
        conn.close()
    
    def test_analyze_new_service_returns_profile(self):
        """Test that analyze_new_service returns ServiceProfile"""
        # Arrange
        from backend.services.google_service_analyzer import GoogleServiceAnalyzer
        
        with patch('google.auth.default') as mock_auth:
            mock_auth.return_value = (self.test_doubles.create_mock_credentials(), "test-project")
            analyzer = GoogleServiceAnalyzer(db_path=str(self.db_path))
        
        # Act
        result = analyzer.analyze_new_service("vertex-ai-memory-store", "test-project")
        
        # Assert
        self.assertEqual(result.service_name, "vertex-ai-memory-store")
        self.assertIsNotNone(result.description)
        self.assertIsInstance(result.use_cases, list)
        self.assertIsNotNone(result.security_assessment)
        self.assertIsInstance(result.security_assessment.risk_score, int)
        self.assertBetween(result.security_assessment.risk_score, 0, 10)
    
    def test_service_evaluation_persists_to_database(self):
        """Test that evaluations are saved to database"""
        # Arrange
        from backend.services.google_service_analyzer import GoogleServiceAnalyzer
        
        with patch('google.auth.default') as mock_auth:
            mock_auth.return_value = (self.test_doubles.create_mock_credentials(), "test-project")
            analyzer = GoogleServiceAnalyzer(db_path=str(self.db_path))
        
        # Act
        analyzer.analyze_new_service("test-service", "test-project")
        
        # Assert - Verify data was saved
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM service_evaluations WHERE service_name = ?", ("test-service",))
        count = cursor.fetchone()[0]
        self.assertEqual(count, 1)
        conn.close()
    
    def test_list_evaluations_returns_all_profiles(self):
        """Test that list_all_evaluations returns saved profiles"""
        # Arrange
        from backend.services.google_service_analyzer import GoogleServiceAnalyzer
        
        with patch('google.auth.default') as mock_auth:
            mock_auth.return_value = (self.test_doubles.create_mock_credentials(), "test-project")
            analyzer = GoogleServiceAnalyzer(db_path=str(self.db_path))
        
        # Save multiple evaluations
        analyzer.analyze_new_service("service-1", "test-project")
        analyzer.analyze_new_service("service-2", "test-project")
        
        # Act
        profiles = analyzer.list_all_evaluations()
        
        # Assert
        self.assertEqual(len(profiles), 2)
        service_names = [p.service_name for p in profiles]
        self.assertIn("service-1", service_names)
        self.assertIn("service-2", service_names)
    
    def test_handles_missing_credentials_gracefully(self):
        """Test graceful handling of missing GCP credentials"""
        # Arrange
        from backend.services.google_service_analyzer import GoogleServiceAnalyzer
        
        with patch('google.auth.default') as mock_auth:
            mock_auth.side_effect = Exception("Credentials not found")
            analyzer = GoogleServiceAnalyzer(db_path=str(self.db_path))
        
        # Act & Assert - Should not raise, uses mock data
        result = analyzer.analyze_new_service("test-service", "test-project")
        self.assertIsNotNone(result)
        self.assertEqual(result.service_name, "test-service")
    
    def assertBetween(self, value, min_val, max_val):
        """Helper to assert value is between min and max"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)


class TestServiceEvaluationAPI(unittest.TestCase):
    """Integration tests for Service Evaluation API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        from fastapi.testclient import TestClient
        from backend.main import app
        
        self.client = TestClient(app)
        self.test_doubles = TestDoubles()
    
    def test_evaluate_endpoint_returns_service_profile(self):
        """Test POST /api/v1/google-services/evaluate"""
        # Arrange
        payload = {
            "service_name": "cloud-run",
            "project_id": "test-project"
        }
        
        # Act
        with patch('backend.services.google_service_analyzer.google.auth.default') as mock_auth:
            mock_auth.return_value = (self.test_doubles.create_mock_credentials(), "test-project")
            response = self.client.post("/api/v1/google-services/evaluate", json=payload)
        
        # Assert
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service_name"], "cloud-run")
        self.assertIn("security_assessment", data)
        self.assertIn("risk_score", data["security_assessment"])
    
    def test_list_evaluations_endpoint(self):
        """Test GET /api/v1/google-services/evaluations/list"""
        # Act
        with patch('backend.services.google_service_analyzer.google.auth.default') as mock_auth:
            mock_auth.return_value = (self.test_doubles.create_mock_credentials(), "test-project")
            
            # First evaluate some services
            self.client.post("/api/v1/google-services/evaluate", 
                           json={"service_name": "service-1", "project_id": "test"})
            
            # Then list them
            response = self.client.get("/api/v1/google-services/evaluations/list")
        
        # Assert
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
    
    def test_evaluate_endpoint_validates_input(self):
        """Test input validation for evaluate endpoint"""
        # Arrange - Missing required field
        payload = {"project_id": "test-project"}
        
        # Act
        response = self.client.post("/api/v1/google-services/evaluate", json=payload)
        
        # Assert
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_evaluate_endpoint_handles_errors(self):
        """Test error handling in evaluate endpoint"""
        # Arrange
        payload = {
            "service_name": "error-service",
            "project_id": "test-project"
        }
        
        # Act
        with patch('backend.api.google_services.analyzer.analyze_new_service') as mock_analyze:
            mock_analyze.side_effect = Exception("Test error")
            response = self.client.post("/api/v1/google-services/evaluate", json=payload)
        
        # Assert
        self.assertEqual(response.status_code, 500)
        self.assertIn("Failed to evaluate service", response.json()["detail"])


class TestServiceEvaluationContract(unittest.TestCase):
    """Contract tests for Service Evaluation - verify interfaces"""
    
    def test_service_profile_contract(self):
        """Test ServiceProfile data contract"""
        from backend.services.google_service_analyzer import ServiceProfile, SecurityAssessment, RiskProfile
        
        # Arrange
        risk_profile = RiskProfile(
            data_exposure=5,
            misconfiguration=4,
            attack_surface=6,
            compliance_violation=3
        )
        
        security_assessment = SecurityAssessment(
            iam_permissions=["test.permission"],
            network_exposure="public",
            data_encryption="AES-256",
            compliance_certifications=["SOC2"],
            risk_score=5,
            risk_profile=risk_profile,
            threat_model_summary="Test threats",
            data_residency="us-central1"
        )
        
        # Act
        profile = ServiceProfile(
            service_name="test-service",
            description="Test description",
            use_cases=["Test use case"],
            security_assessment=security_assessment,
            release_stage="GA"
        )
        
        # Assert - Verify all required fields
        self.assertEqual(profile.service_name, "test-service")
        self.assertEqual(profile.security_assessment.risk_score, 5)
        self.assertEqual(profile.security_assessment.risk_profile.data_exposure, 5)
        
        # Verify JSON serialization
        json_data = profile.model_dump_json()
        self.assertIsInstance(json_data, str)
        
        # Verify deserialization
        restored = ServiceProfile.model_validate_json(json_data)
        self.assertEqual(restored.service_name, profile.service_name)
    
    def test_api_request_contract(self):
        """Test API request/response contract"""
        from backend.api.google_services import ServiceEvaluationRequest
        
        # Arrange
        request_data = {
            "service_name": "test-service",
            "project_id": "test-project"
        }
        
        # Act
        request = ServiceEvaluationRequest(**request_data)
        
        # Assert
        self.assertEqual(request.service_name, "test-service")
        self.assertEqual(request.project_id, "test-project")


class TestServiceEvaluationBehavior(unittest.TestCase):
    """Behavior tests - verify system behavior not implementation"""
    
    def test_caches_evaluation_results(self):
        """Test that repeated evaluations use cache"""
        from backend.services.google_service_analyzer import GoogleServiceAnalyzer
        
        # Arrange
        with patch('google.auth.default') as mock_auth:
            mock_auth.return_value = (TestDoubles.create_mock_credentials(), "test-project")
            analyzer = GoogleServiceAnalyzer(db_path=":memory:")
        
        # Act - Evaluate same service twice
        with patch.object(analyzer, '_fetch_real_service_data') as mock_fetch:
            mock_fetch.return_value = {"is_enabled": True, "iam_permissions": []}
            
            result1 = analyzer.analyze_new_service("cached-service", "test-project")
            result2 = analyzer.analyze_new_service("cached-service", "test-project")
        
        # Assert - Should only fetch once due to caching
        self.assertEqual(mock_fetch.call_count, 1)
        self.assertEqual(result1.service_name, result2.service_name)
    
    def test_provides_specialized_profiles_for_known_services(self):
        """Test that known services get specialized profiles"""
        from backend.services.google_service_analyzer import GoogleServiceAnalyzer
        
        # Arrange
        with patch('google.auth.default') as mock_auth:
            mock_auth.return_value = (TestDoubles.create_mock_credentials(), "test-project")
            analyzer = GoogleServiceAnalyzer(db_path=":memory:")
        
        # Act
        vertex_profile = analyzer.analyze_new_service("vertex-ai-memory-store", "test-project")
        alloydb_profile = analyzer.analyze_new_service("alloydb", "test-project")
        generic_profile = analyzer.analyze_new_service("unknown-service", "test-project")
        
        # Assert - Different services get different profiles
        self.assertIn("vector", vertex_profile.description.lower())
        self.assertIn("postgresql", alloydb_profile.description.lower())
        self.assertNotIn("vector", generic_profile.description.lower())
        self.assertNotIn("postgresql", generic_profile.description.lower())
    
    def test_risk_scores_reflect_service_characteristics(self):
        """Test that risk scores match service characteristics"""
        from backend.services.google_service_analyzer import GoogleServiceAnalyzer
        
        # Arrange
        with patch('google.auth.default') as mock_auth:
            mock_auth.return_value = (TestDoubles.create_mock_credentials(), "test-project")
            analyzer = GoogleServiceAnalyzer(db_path=":memory:")
        
        # Act
        vertex_profile = analyzer.analyze_new_service("vertex-ai-memory-store", "test-project")
        alloydb_profile = analyzer.analyze_new_service("alloydb", "test-project")
        
        # Assert - AI services have higher data exposure risk
        vertex_risk = vertex_profile.security_assessment.risk_profile.data_exposure
        alloydb_risk = alloydb_profile.security_assessment.risk_profile.data_exposure
        self.assertGreater(vertex_risk, alloydb_risk)


# Test runner
if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)