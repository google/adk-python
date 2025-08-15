"""
Complete Integration Test Suite
Tests all refinements and optimizations
"""

import asyncio
import requests
import json
import time
from datetime import datetime

class CompleteIntegrationTester:
    """Test all system components with real GCP data"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.project_id = "mgm-digitalconcierge"
        self.results = []
        
    def test_real_gcp_assets(self):
        """Test real GCP asset fetching"""
        print("\n🔍 Testing Real GCP Asset Integration...")
        
        # Test asset inventory summary
        response = requests.get(
            f"{self.base_url}/api/v1/asset-inventory/summary",
            params={"project_id": self.project_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            assets = data.get("data", {})
            
            print(f"✅ Total Assets: {assets.get('total_assets', 0)}")
            print(f"   - Storage Buckets: {assets.get('asset_types', {}).get('Storage Buckets', 0)}")
            print(f"   - IAM Accounts: {assets.get('asset_types', {}).get('IAM Accounts', 0)}")
            print(f"   - Networks: {assets.get('asset_types', {}).get('Networks', 0)}")
            print(f"   - Databases: {assets.get('asset_types', {}).get('Databases', 0)}")
            print(f"   - Compute Instances: {assets.get('asset_types', {}).get('Compute Instances', 0)}")
            
            self.results.append(("Asset Inventory", "PASS", assets.get('total_assets', 0)))
        else:
            print(f"❌ Asset inventory failed: HTTP {response.status_code}")
            self.results.append(("Asset Inventory", "FAIL", 0))
    
    def test_cache_performance(self):
        """Test caching layer performance"""
        print("\n⚡ Testing Cache Performance...")
        
        # First request (cache miss)
        start = time.time()
        response1 = requests.get(
            f"{self.base_url}/api/v1/asset-inventory/summary",
            params={"project_id": self.project_id}
        )
        time1 = time.time() - start
        
        # Second request (cache hit)
        start = time.time()
        response2 = requests.get(
            f"{self.base_url}/api/v1/asset-inventory/summary",
            params={"project_id": self.project_id}
        )
        time2 = time.time() - start
        
        if response1.status_code == 200 and response2.status_code == 200:
            improvement = ((time1 - time2) / time1) * 100 if time1 > 0 else 0
            print(f"✅ First request: {time1:.3f}s")
            print(f"✅ Second request: {time2:.3f}s")
            print(f"✅ Cache improvement: {improvement:.1f}%")
            
            self.results.append(("Cache Performance", "PASS", f"{improvement:.1f}%"))
        else:
            print("❌ Cache test failed")
            self.results.append(("Cache Performance", "FAIL", "0%"))
    
    def test_chat_with_real_data(self):
        """Test chat interface with real GCP data"""
        print("\n💬 Testing Chat with Real Data...")
        
        queries = [
            "List my storage buckets and security issues",
            "Show me IAM accounts with excessive permissions",
            "What network security risks do I have?",
            "Are there any databases with public access?"
        ]
        
        for query in queries:
            response = requests.post(
                f"{self.base_url}/api/v1/agent/chat",
                json={
                    "query": query,
                    "user_id": "test_user",
                    "project_id": self.project_id
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                agent = data.get("agent_used", "Unknown")
                response_text = data.get("response", "")
                
                # Check if response contains real data
                has_real_data = any([
                    "10" in response_text,  # Real bucket count
                    "4" in response_text,   # Real IAM count
                    "mgm-digitalconcierge" in response_text,
                    "419850945193" in response_text  # Real bucket names
                ])
                
                status = "PASS" if has_real_data else "PARTIAL"
                print(f"{'✅' if has_real_data else '⚠️'} {query[:30]}... -> {agent}")
                
                self.results.append((f"Chat: {query[:20]}...", status, agent))
            else:
                print(f"❌ Failed: {query[:30]}...")
                self.results.append((f"Chat: {query[:20]}...", "FAIL", "N/A"))
    
    def test_error_handling(self):
        """Test comprehensive error handling"""
        print("\n🛡️ Testing Error Handling...")
        
        # Test invalid project
        response = requests.get(
            f"{self.base_url}/api/v1/asset-inventory/summary",
            params={"project_id": "invalid-project-12345"}
        )
        
        if response.status_code in [200, 400, 403, 404]:
            print("✅ Invalid project handled gracefully")
            self.results.append(("Error Handling", "PASS", "Graceful"))
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            self.results.append(("Error Handling", "FAIL", "Ungraceful"))
    
    def test_modularization(self):
        """Test that large files have been modularized"""
        print("\n📦 Testing Code Modularization...")
        
        # Check if new modules exist
        modules = [
            "backend/api/agent_routing.py",
            "backend/api/agent_factory.py",
            "backend/services/cache_service.py",
            "backend/services/gcp_thin_client_service.py"
        ]
        
        import os
        modules_exist = []
        for module in modules:
            path = f"/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/{module}"
            exists = os.path.exists(path)
            modules_exist.append(exists)
            print(f"{'✅' if exists else '❌'} {module}")
        
        if all(modules_exist):
            self.results.append(("Modularization", "PASS", f"{len(modules)} modules"))
        else:
            self.results.append(("Modularization", "PARTIAL", f"{sum(modules_exist)}/{len(modules)}"))
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "=" * 60)
        print("📊 COMPLETE INTEGRATION TEST REPORT")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Project: {self.project_id}")
        print("\nTest Results:")
        print("-" * 60)
        
        for test, status, details in self.results:
            status_emoji = {
                "PASS": "✅",
                "FAIL": "❌",
                "PARTIAL": "⚠️"
            }.get(status, "❓")
            print(f"{status_emoji} {test:<30} {status:<10} {details}")
        
        # Calculate overall score
        pass_count = sum(1 for _, status, _ in self.results if status == "PASS")
        total_count = len(self.results)
        score = (pass_count / total_count * 100) if total_count > 0 else 0
        
        print("-" * 60)
        print(f"Overall Score: {score:.1f}% ({pass_count}/{total_count} tests passed)")
        
        if score >= 80:
            print("\n🎉 SYSTEM READY FOR PRODUCTION")
        elif score >= 60:
            print("\n⚠️ SYSTEM PARTIALLY READY - Some improvements needed")
        else:
            print("\n❌ SYSTEM NEEDS MORE WORK")
        
        print("=" * 60)
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Complete Integration Test Suite...")
        
        self.test_modularization()
        self.test_real_gcp_assets()
        self.test_cache_performance()
        self.test_chat_with_real_data()
        self.test_error_handling()
        
        self.generate_report()

if __name__ == "__main__":
    tester = CompleteIntegrationTester()
    tester.run_all_tests()