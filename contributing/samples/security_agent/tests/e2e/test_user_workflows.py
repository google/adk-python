"""
End-to-End User Workflow Tests
==============================

Tests complete user workflows from start to finish:
- New user onboarding flow
- Security analysis workflow
- Asset discovery workflow  
- Multi-turn conversation flows
- Error recovery workflows
- Session management workflows
"""

import pytest
import time
import uuid
from unittest.mock import patch
from fastapi.testclient import TestClient

# Add backend to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from backend.main import app


class TestSecurityAnalysisWorkflow:
    """Test complete security analysis workflow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.session_id = f"security-workflow-{uuid.uuid4().hex[:8]}"
        self.user_id = "security-analyst-user"
    
    def test_complete_security_analysis_workflow(self):
        """Test complete security analysis from start to finish."""
        # Step 1: Initial greeting and project setup
        response1 = self.client.post("/api/v1/chat/message", json={
            "query": "Hello, I need help with security analysis of my GCP project",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["success"] is True
        assert "response" in data1
        assert len(data1["response"]) > 0
        
        # Step 2: Ask about resources
        response2 = self.client.post("/api/v1/chat/message", json={
            "query": "What resources do I have in my project?",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["success"] is True
        assert data2["session_id"] == self.session_id
        
        # Step 3: Security scan request
        response3 = self.client.post("/api/v1/chat/message", json={
            "query": "Run a security scan on my infrastructure",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response3.status_code == 200
        data3 = response3.json()
        assert data3["success"] is True
        
        # Step 4: Ask for recommendations
        response4 = self.client.post("/api/v1/chat/message", json={
            "query": "What security recommendations do you have?",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response4.status_code == 200
        data4 = response4.json()
        assert data4["success"] is True
        
        # All responses should maintain session continuity
        assert all(data["session_id"] == self.session_id for data in [data1, data2, data3, data4])
        assert all(data["user_id"] == self.user_id for data in [data1, data2, data3, data4])


class TestAssetDiscoveryWorkflow:
    """Test asset discovery and inventory workflow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.session_id = f"asset-workflow-{uuid.uuid4().hex[:8]}"
        self.user_id = "asset-manager-user"
    
    def test_asset_discovery_workflow(self):
        """Test complete asset discovery workflow."""
        # Step 1: Request asset inventory
        response1 = self.client.post("/api/v1/chat/message", json={
            "query": "Show me all my GCP assets",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["success"] is True
        
        # Step 2: Ask for specific resource types
        response2 = self.client.post("/api/v1/chat/message", json={
            "query": "What compute instances do I have?",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["success"] is True
        
        # Step 3: Ask about storage
        response3 = self.client.post("/api/v1/chat/message", json={
            "query": "Show me my storage buckets",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response3.status_code == 200
        data3 = response3.json()
        assert data3["success"] is True
        
        # Step 4: Request summary report
        response4 = self.client.post("/api/v1/chat/message", json={
            "query": "Give me a summary of all my resources",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response4.status_code == 200
        data4 = response4.json()
        assert data4["success"] is True
        
        # Verify session consistency
        assert all(data["session_id"] == self.session_id for data in [data1, data2, data3, data4])


class TestMultiTurnConversationWorkflow:
    """Test complex multi-turn conversation workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.session_id = f"conversation-{uuid.uuid4().hex[:8]}"
        self.user_id = "conversation-user"
    
    def test_context_aware_conversation(self):
        """Test conversation maintains context across turns."""
        # Turn 1: Establish context
        response1 = self.client.post("/api/v1/chat/message", json={
            "query": "I'm concerned about my project's security posture",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["success"] is True
        
        # Turn 2: Follow-up question (should understand context)
        response2 = self.client.post("/api/v1/chat/message", json={
            "query": "What are the most critical issues I should address first?",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["success"] is True
        
        # Turn 3: Reference to previous conversation
        response3 = self.client.post("/api/v1/chat/message", json={
            "query": "How do I fix those issues you mentioned?",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response3.status_code == 200
        data3 = response3.json()
        assert data3["success"] is True
        
        # Turn 4: Change topic but maintain session
        response4 = self.client.post("/api/v1/chat/message", json={
            "query": "Actually, let me also check my IAM policies",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response4.status_code == 200
        data4 = response4.json()
        assert data4["success"] is True
        
        # Turn 5: Reference back to earlier topic
        response5 = self.client.post("/api/v1/chat/message", json={
            "query": "Going back to those security issues, can you prioritize them?",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response5.status_code == 200
        data5 = response5.json()
        assert data5["success"] is True
        
        # All responses should maintain session
        responses = [data1, data2, data3, data4, data5]
        assert all(data["session_id"] == self.session_id for data in responses)
        assert all(data["user_id"] == self.user_id for data in responses)
    
    def test_complex_analytical_workflow(self):
        """Test complex analytical workflow with multiple topics."""
        queries = [
            "I need a comprehensive security assessment",
            "Start with analyzing my IAM permissions",
            "Are there any overprivileged service accounts?",
            "What about my network security?",
            "Check for any public storage buckets",
            "Do I have any compliance issues?",
            "What's my overall security score?",
            "Give me an action plan to improve security"
        ]
        
        responses = []
        for i, query in enumerate(queries):
            response = self.client.post("/api/v1/chat/message", json={
                "query": query,
                "session_id": self.session_id,
                "user_id": self.user_id
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == self.session_id
            responses.append(data)
            
            # Small delay between requests to simulate real user interaction
            time.sleep(0.1)
        
        # All responses should have content
        assert all(len(data["response"]) > 0 for data in responses)


class TestErrorRecoveryWorkflow:
    """Test error recovery and resilience workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.session_id = f"error-recovery-{uuid.uuid4().hex[:8]}"
        self.user_id = "error-test-user"
    
    def test_invalid_query_recovery(self):
        """Test recovery from invalid queries."""
        # Send invalid/problematic queries and ensure system recovers
        problematic_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "a" * 10000,  # Very long query
            "SELECT * FROM users; DROP TABLE sessions;",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "What is my project ID?" + "\x00" * 10,  # Null bytes
        ]
        
        for query in problematic_queries:
            response = self.client.post("/api/v1/chat/message", json={
                "query": query,
                "session_id": self.session_id,
                "user_id": self.user_id
            })
            
            # Should handle gracefully (not crash)
            assert response.status_code in [200, 422, 413]
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                assert "response" in data
        
        # After problematic queries, normal query should still work
        recovery_response = self.client.post("/api/v1/chat/message", json={
            "query": "Hello, are you working properly?",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert recovery_response.status_code == 200
        data = recovery_response.json()
        assert data["success"] is True
        assert len(data["response"]) > 0
    
    def test_session_recovery(self):
        """Test recovery with session issues."""
        # Test with problematic session IDs
        problematic_sessions = [
            "",  # Empty session
            "../../etc/passwd",  # Path traversal
            "session" + "x" * 1000,  # Very long session
            "session\x00\x01\x02",  # Binary data
        ]
        
        for session_id in problematic_sessions:
            response = self.client.post("/api/v1/chat/message", json={
                "query": "Test session recovery",
                "session_id": session_id,
                "user_id": self.user_id
            })
            
            # Should handle gracefully
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
        
        # Normal session should work after problematic ones
        normal_response = self.client.post("/api/v1/chat/message", json={
            "query": "Normal session test",
            "session_id": "normal-session-test",
            "user_id": self.user_id
        })
        
        assert normal_response.status_code == 200
        data = normal_response.json()
        assert data["success"] is True


class TestNewUserOnboardingWorkflow:
    """Test new user onboarding experience."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.session_id = f"onboarding-{uuid.uuid4().hex[:8]}"
        self.user_id = f"new-user-{uuid.uuid4().hex[:8]}"
    
    def test_first_time_user_experience(self):
        """Test first-time user onboarding flow."""
        # Step 1: Initial greeting
        response1 = self.client.post("/api/v1/chat/message", json={
            "query": "Hello",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["success"] is True
        assert len(data1["response"]) > 0
        
        # Step 2: Ask for help
        response2 = self.client.post("/api/v1/chat/message", json={
            "query": "I'm new here, what can you help me with?",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["success"] is True
        
        # Step 3: Ask about capabilities
        response3 = self.client.post("/api/v1/chat/message", json={
            "query": "What security features do you provide?",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response3.status_code == 200
        data3 = response3.json()
        assert data3["success"] is True
        
        # Step 4: Request getting started guidance
        response4 = self.client.post("/api/v1/chat/message", json={
            "query": "How do I get started with security analysis?",
            "session_id": self.session_id,
            "user_id": self.user_id
        })
        
        assert response4.status_code == 200
        data4 = response4.json()
        assert data4["success"] is True
        
        # All responses should be helpful and maintain session
        responses = [data1, data2, data3, data4]
        assert all(data["success"] is True for data in responses)
        assert all(data["session_id"] == self.session_id for data in responses)
        assert all(len(data["response"]) > 10 for data in responses)  # Should have substantial responses


class TestHealthCheckIntegrationWorkflow:
    """Test health check integration in user workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
    
    def test_system_health_during_user_workflow(self):
        """Test system remains healthy during active user workflows."""
        # Check initial system health
        health_response = self.client.get("/health")
        assert health_response.status_code == 200
        initial_health = health_response.json()
        assert initial_health["status"] in ["healthy", "degraded"]
        
        # Simulate active user workflow
        session_id = f"health-test-{uuid.uuid4().hex[:8]}"
        for i in range(10):
            chat_response = self.client.post("/api/v1/chat/message", json={
                "query": f"Health test query {i}",
                "session_id": session_id,
                "user_id": "health-test-user"
            })
            assert chat_response.status_code == 200
            
            # Check health periodically during workflow
            if i % 3 == 0:
                health_check = self.client.get("/health")
                assert health_check.status_code == 200
                health_data = health_check.json()
                assert health_data["status"] in ["healthy", "degraded"]
        
        # Check final system health
        final_health_response = self.client.get("/health")
        assert final_health_response.status_code == 200
        final_health = final_health_response.json()
        assert final_health["status"] in ["healthy", "degraded"]
        
        # System should still be operational
        assert final_health["status"] != "unhealthy"


class TestLongRunningSessionWorkflow:
    """Test long-running session workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.session_id = f"long-session-{uuid.uuid4().hex[:8]}"
        self.user_id = "long-session-user"
    
    def test_extended_conversation_session(self):
        """Test extended conversation maintains consistency."""
        # Simulate a long conversation over time
        conversation_topics = [
            "Hello, I need help with GCP security",
            "What are the most common security issues?",
            "How do I check my current security posture?",
            "What about IAM best practices?", 
            "Can you analyze my service accounts?",
            "What network security controls should I implement?",
            "How do I monitor security events?",
            "What compliance frameworks do you support?",
            "Can you help with incident response planning?",
            "What about backup and disaster recovery security?",
            "How do I secure my CI/CD pipelines?",
            "What are the latest security threats to watch for?",
            "Can you create a security roadmap for me?",
            "How do I train my team on security best practices?",
            "What security tools do you recommend?",
            "Thank you for all the help!"
        ]
        
        responses = []
        for i, query in enumerate(conversation_topics):
            response = self.client.post("/api/v1/chat/message", json={
                "query": query,
                "session_id": self.session_id,
                "user_id": self.user_id
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == self.session_id
            assert data["user_id"] == self.user_id
            responses.append(data)
            
            # Add small delay to simulate realistic conversation timing
            time.sleep(0.05)
        
        # All responses should be successful and maintain session continuity
        assert len(responses) == len(conversation_topics)
        assert all(data["success"] is True for data in responses)
        assert all(len(data["response"]) > 0 for data in responses)
        
        # Session should be consistent throughout
        unique_sessions = set(data["session_id"] for data in responses)
        assert len(unique_sessions) == 1
        assert list(unique_sessions)[0] == self.session_id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])