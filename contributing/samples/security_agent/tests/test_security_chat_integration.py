"""
Test GCP Security Chat Integration
Verifies the ChatGPT-like experience with Asset Inventory and Recommendations
"""

import asyncio
import requests
import json
from typing import Dict, List, Any
from datetime import datetime

class SecurityChatTester:
    """Test the complete security chat experience"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        self.project_id = "mgm-digitalconcierge"
        
    def test_asset_inventory_integration(self):
        """Test asset inventory queries through chat"""
        print("\n🔍 Testing Asset Inventory Integration...")
        
        queries = [
            "Show me all my GCP resources",
            "What storage buckets do I have?",
            "List my compute instances",
            "Show me resources with public access",
            "Which assets have security issues?"
        ]
        
        for query in queries:
            print(f"\n📝 Query: {query}")
            response = self._send_chat_query(query)
            
            if response.get("success"):
                print(f"✅ Response from {response.get('agent_used', 'Unknown')}:")
                print(f"   {response.get('response', '')[:200]}...")
                
                if response.get("suggestions"):
                    print(f"💡 Suggestions: {response['suggestions'][:2]}")
            else:
                print(f"❌ Failed: {response.get('error', 'Unknown error')}")
    
    def test_security_recommendations(self):
        """Test security recommendation queries"""
        print("\n💡 Testing Security Recommendations...")
        
        queries = [
            "What are my top security recommendations?",
            "How can I improve my security posture?",
            "Show me critical security issues",
            "What should I fix first?",
            "Give me quick security wins"
        ]
        
        for query in queries:
            print(f"\n📝 Query: {query}")
            response = self._send_chat_query(query)
            
            if response.get("success"):
                print(f"✅ Response from {response.get('agent_used', 'Unknown')}:")
                print(f"   {response.get('response', '')[:200]}...")
                
                # Check for recommendation-specific content
                if "recommendation" in response.get("response", "").lower():
                    print("   ✓ Contains recommendations")
                if "priority" in response.get("response", "").lower():
                    print("   ✓ Has priority information")
            else:
                print(f"❌ Failed: {response.get('error', 'Unknown error')}")
    
    def test_conversational_flow(self):
        """Test multi-turn conversational flow"""
        print("\n💬 Testing Conversational Flow...")
        
        conversation = [
            ("Tell me about my buckets", "storage"),
            ("Which ones are public?", "storage_security"),
            ("How do I fix the public access?", "remediation"),
            ("What other storage issues should I look at?", "recommendations"),
            ("Show me the compliance status", "compliance")
        ]
        
        for query, expected_context in conversation:
            print(f"\n👤 User: {query}")
            response = self._send_chat_query(query)
            
            if response.get("success"):
                print(f"🤖 Assistant ({response.get('agent_used', 'Unknown')}):")
                print(f"   {response.get('response', '')[:150]}...")
                
                # Verify context-aware response
                if expected_context in ["storage", "bucket"] and "bucket" in response.get("response", "").lower():
                    print("   ✓ Context maintained")
                
                # Check suggestions are relevant
                if response.get("suggestions"):
                    print(f"   💡 Follow-ups: {response['suggestions'][0]}")
            else:
                print(f"❌ Failed: {response.get('error', 'Unknown error')}")
    
    def test_asset_inventory_summary(self):
        """Test asset inventory summary endpoint"""
        print("\n📊 Testing Asset Inventory Summary...")
        
        response = requests.get(
            f"{self.base_url}/api/v1/asset-inventory/summary",
            params={"project_id": self.project_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                summary = data.get("data", {})
                print("✅ Asset Summary Retrieved:")
                print(f"   Total Assets: {summary.get('total_assets', 0)}")
                print(f"   Security Findings: {summary.get('security_findings', 0)}")
                print(f"   High Risk Assets: {summary.get('high_risk_assets', 0)}")
                print(f"   Active Recommendations: {summary.get('active_recommendations', 0)}")
                
                if summary.get("asset_types"):
                    print("   Asset Breakdown:")
                    for asset_type, count in summary["asset_types"].items():
                        print(f"     • {asset_type}: {count}")
            else:
                print(f"❌ Failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP {response.status_code}: Failed to get summary")
    
    def test_performance_metrics(self):
        """Test response times and caching"""
        print("\n⚡ Testing Performance...")
        
        # First query (cache miss)
        start = datetime.now()
        response1 = self._send_chat_query("Show me all my resources")
        time1 = (datetime.now() - start).total_seconds()
        
        # Second identical query (cache hit)
        start = datetime.now()
        response2 = self._send_chat_query("Show me all my resources")
        time2 = (datetime.now() - start).total_seconds()
        
        print(f"First query: {time1:.2f}s")
        print(f"Second query: {time2:.2f}s")
        
        if time2 < time1:
            print("✅ Caching appears to be working")
        else:
            print("⚠️ Caching may not be optimized")
        
        # Check response metrics
        if response1.get("performance_metrics"):
            metrics = response1["performance_metrics"]
            print(f"Response time: {metrics.get('response_time_ms', 0)}ms")
            print(f"Query length: {metrics.get('query_length', 0)} chars")
            print(f"Response length: {metrics.get('response_length', 0)} chars")
    
    def _send_chat_query(self, query: str) -> Dict[str, Any]:
        """Send a chat query to the backend"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/agent/chat",
                json={
                    "query": query,
                    "user_id": "test_user",
                    "project_id": self.project_id,
                    "session_id": self.session_id
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                # Store session for conversation continuity
                if data.get("session_id"):
                    self.session_id = data["session_id"]
                return data
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("=" * 60)
        print("🚀 GCP Security Chat Integration Test Suite")
        print("=" * 60)
        
        # Check if backend is running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            if response.status_code != 200:
                print("⚠️ Backend health check failed")
        except:
            print("❌ Backend is not running on port 8000")
            print("Please start the backend with: python run_backend.py")
            return
        
        # Run test suites
        self.test_asset_inventory_summary()
        self.test_asset_inventory_integration()
        self.test_security_recommendations()
        self.test_conversational_flow()
        self.test_performance_metrics()
        
        print("\n" + "=" * 60)
        print("✅ Integration tests completed")
        print("=" * 60)

if __name__ == "__main__":
    tester = SecurityChatTester()
    tester.run_all_tests()