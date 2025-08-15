"""
Full Integration Test for GCP Security Agent
Tests all real GCP API integrations without mock data
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.gcp_thin_client_service import GCPThinClientService
from backend.services.gcp_extended_assets import ExtendedAssetDiscovery


class FullIntegrationTester:
    """Comprehensive integration testing for all GCP capabilities"""
    
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "mgm-digitalconcierge")
        self.results = []
        self.service = GCPThinClientService(self.project_id)
        self.extended = ExtendedAssetDiscovery(self.project_id)
        
    async def test_asset_inventory_snapshot(self):
        """Test comprehensive asset inventory snapshot"""
        print("\n📊 Testing Asset Inventory Snapshot...")
        
        try:
            snapshot = await self.service.get_asset_inventory_snapshot()
            
            print(f"✅ Total assets found: {snapshot.total_assets}")
            print("\nAsset Breakdown:")
            for asset_type, count in snapshot.asset_breakdown.items():
                if count > 0:
                    print(f"  • {asset_type}: {count}")
            
            if snapshot.security_findings:
                print(f"\n🔍 Security findings: {len(snapshot.security_findings)}")
            
            if snapshot.high_risk_assets:
                print(f"⚠️  High-risk assets: {len(snapshot.high_risk_assets)}")
            
            if snapshot.recommendations:
                print(f"💡 Recommendations: {len(snapshot.recommendations)}")
                for rec in snapshot.recommendations[:3]:
                    print(f"   - [{rec.severity}] {rec.title}")
            
            print(f"\n⏱️  Scan duration: {snapshot.scan_duration_ms:.2f}ms")
            
            self.results.append(("Asset Inventory Snapshot", "PASS", snapshot.total_assets))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("Asset Inventory Snapshot", "FAIL", str(e)))
            return False
    
    async def test_storage_assets(self):
        """Test storage bucket discovery"""
        print("\n🪣 Testing Storage Assets...")
        
        try:
            buckets = await self.service._fetch_storage_assets()
            
            if buckets:
                print(f"✅ Found {len(buckets)} storage buckets")
                for bucket in buckets[:3]:
                    print(f"  • {bucket['name']}")
                    print(f"    - Location: {bucket.get('location', 'N/A')}")
                    print(f"    - Public: {bucket.get('public_access', False)}")
                    print(f"    - Encrypted: {bucket.get('encryption_enabled', False)}")
            else:
                print("ℹ️  No storage buckets found")
            
            self.results.append(("Storage Assets", "PASS", len(buckets)))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("Storage Assets", "FAIL", str(e)))
            return False
    
    async def test_iam_assets(self):
        """Test IAM and service account discovery"""
        print("\n👤 Testing IAM Assets...")
        
        try:
            iam_assets = await self.service._fetch_iam_assets()
            
            if iam_assets:
                print(f"✅ Found {len(iam_assets)} IAM assets")
                for asset in iam_assets[:3]:
                    if asset.get('email'):
                        print(f"  • Service Account: {asset['email']}")
                    elif asset.get('total_bindings'):
                        print(f"  • IAM Policy: {asset['total_bindings']} bindings, {asset['total_members']} members")
            else:
                print("ℹ️  No IAM assets found")
            
            self.results.append(("IAM Assets", "PASS", len(iam_assets)))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("IAM Assets", "FAIL", str(e)))
            return False
    
    async def test_network_assets(self):
        """Test network and firewall discovery"""
        print("\n🌐 Testing Network Assets...")
        
        try:
            networks = await self.service._fetch_network_assets()
            
            if networks:
                print(f"✅ Found {len(networks)} network resources")
                vpc_count = sum(1 for n in networks if "Network" in n.get('asset_type', ''))
                firewall_count = sum(1 for n in networks if "Firewall" in n.get('asset_type', ''))
                
                print(f"  • VPC Networks: {vpc_count}")
                print(f"  • Firewall Rules: {firewall_count}")
                
                # Check for risky firewall rules
                risky_rules = [n for n in networks if n.get('public_access')]
                if risky_rules:
                    print(f"  ⚠️  Risky firewall rules (0.0.0.0/0): {len(risky_rules)}")
            else:
                print("ℹ️  No network assets found")
            
            self.results.append(("Network Assets", "PASS", len(networks)))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("Network Assets", "FAIL", str(e)))
            return False
    
    async def test_cloud_functions(self):
        """Test Cloud Functions discovery"""
        print("\n⚡ Testing Cloud Functions...")
        
        try:
            functions = await self.extended.fetch_cloud_functions()
            
            if functions:
                print(f"✅ Found {len(functions)} Cloud Functions")
                for func in functions[:3]:
                    print(f"  • {func['name'].split('/')[-1]}")
                    print(f"    - Runtime: {func.get('runtime', 'N/A')}")
                    print(f"    - Trigger: {func.get('trigger', 'N/A')}")
            else:
                print("ℹ️  No Cloud Functions found")
            
            self.results.append(("Cloud Functions", "PASS", len(functions)))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("Cloud Functions", "FAIL", str(e)))
            return False
    
    async def test_bigquery_datasets(self):
        """Test BigQuery dataset discovery"""
        print("\n📊 Testing BigQuery Datasets...")
        
        try:
            datasets = await self.extended.fetch_bigquery_datasets()
            
            if datasets:
                print(f"✅ Found {len(datasets)} BigQuery datasets")
                for dataset in datasets[:3]:
                    print(f"  • {dataset['name']}")
                    print(f"    - Location: {dataset.get('location', 'N/A')}")
                    print(f"    - Tables: {dataset.get('table_count', 0)}")
            else:
                print("ℹ️  No BigQuery datasets found")
            
            self.results.append(("BigQuery Datasets", "PASS", len(datasets)))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("BigQuery Datasets", "FAIL", str(e)))
            return False
    
    async def test_pubsub_topics(self):
        """Test Pub/Sub topic discovery"""
        print("\n📨 Testing Pub/Sub Topics...")
        
        try:
            topics = await self.extended.fetch_pubsub_topics()
            
            if topics:
                print(f"✅ Found {len(topics)} Pub/Sub topics")
                for topic in topics[:3]:
                    print(f"  • {topic['name']}")
                    print(f"    - KMS Key: {'Yes' if topic.get('kms_key') else 'No'}")
            else:
                print("ℹ️  No Pub/Sub topics found")
            
            self.results.append(("Pub/Sub Topics", "PASS", len(topics)))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("Pub/Sub Topics", "FAIL", str(e)))
            return False
    
    async def test_gke_clusters(self):
        """Test GKE cluster discovery"""
        print("\n☸️  Testing GKE Clusters...")
        
        try:
            clusters = await self.extended.fetch_gke_clusters()
            
            if clusters:
                print(f"✅ Found {len(clusters)} GKE clusters")
                for cluster in clusters[:3]:
                    print(f"  • {cluster['name']}")
                    print(f"    - Location: {cluster.get('location', 'N/A')}")
                    print(f"    - Nodes: {cluster.get('node_count', 0)}")
                    if cluster.get('security_issues'):
                        print(f"    ⚠️  Security issues: {', '.join(cluster['security_issues'])}")
            else:
                print("ℹ️  No GKE clusters found")
            
            self.results.append(("GKE Clusters", "PASS", len(clusters)))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("GKE Clusters", "FAIL", str(e)))
            return False
    
    async def test_cloud_run_services(self):
        """Test Cloud Run service discovery"""
        print("\n🏃 Testing Cloud Run Services...")
        
        try:
            services = await self.extended.fetch_cloud_run_services()
            
            if services:
                print(f"✅ Found {len(services)} Cloud Run services")
                for service in services[:3]:
                    print(f"  • {service['name']}")
                    print(f"    - URI: {service.get('uri', 'N/A')}")
                    print(f"    - Public: {service.get('public_access', False)}")
            else:
                print("ℹ️  No Cloud Run services found")
            
            self.results.append(("Cloud Run Services", "PASS", len(services)))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("Cloud Run Services", "FAIL", str(e)))
            return False
    
    async def test_security_analysis(self):
        """Test security analysis capabilities"""
        print("\n🔒 Testing Security Analysis...")
        
        try:
            # Test different query types
            queries = [
                "Check storage security",
                "Review IAM permissions",
                "Analyze network firewall rules"
            ]
            
            for query in queries:
                result = await self.service.analyze_asset_security(query)
                print(f"  • {query}")
                print(f"    - Focus: {result.get('focus', 'N/A')}")
                print(f"    - Risk: {result.get('risk_level', 'N/A')}")
            
            self.results.append(("Security Analysis", "PASS", len(queries)))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("Security Analysis", "FAIL", str(e)))
            return False
    
    async def test_recommendations(self):
        """Test recommendation generation"""
        print("\n💡 Testing Recommendations...")
        
        try:
            context = {
                "recent_topics": ["storage", "iam", "network", "compliance"]
            }
            
            recommendations = await self.service.get_contextual_recommendations(context)
            
            if recommendations:
                print(f"✅ Generated {len(recommendations)} recommendations")
                for rec in recommendations[:3]:
                    print(f"  • [{rec.severity}] {rec.title}")
                    print(f"    - {rec.description}")
            else:
                print("ℹ️  No recommendations generated")
            
            self.results.append(("Recommendations", "PASS", len(recommendations)))
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("Recommendations", "FAIL", str(e)))
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("📊 FULL INTEGRATION TEST REPORT")
        print("=" * 70)
        print(f"Project: {self.project_id}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("-" * 70)
        
        for test_name, status, details in self.results:
            emoji = "✅" if status == "PASS" else "❌"
            print(f"{emoji} {test_name:<30} {status:<6} {details}")
        
        pass_count = sum(1 for _, status, _ in self.results if status == "PASS")
        total = len(self.results)
        score = (pass_count / total * 100) if total > 0 else 0
        
        print("-" * 70)
        print(f"Score: {score:.1f}% ({pass_count}/{total} tests passed)")
        
        if score == 100:
            print("\n🎉 ALL TESTS PASSED - SYSTEM FULLY OPERATIONAL")
        elif score >= 80:
            print("\n✅ SYSTEM OPERATIONAL WITH MINOR ISSUES")
        elif score >= 50:
            print("\n⚠️  SYSTEM PARTIALLY OPERATIONAL")
        else:
            print("\n❌ SYSTEM HAS CRITICAL ISSUES")
        
        print("=" * 70)
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Full Integration Tests...")
        print(f"Testing project: {self.project_id}")
        
        # Core tests
        await self.test_asset_inventory_snapshot()
        
        # Individual asset type tests
        await self.test_storage_assets()
        await self.test_iam_assets()
        await self.test_network_assets()
        
        # Extended asset tests
        await self.test_cloud_functions()
        await self.test_bigquery_datasets()
        await self.test_pubsub_topics()
        await self.test_gke_clusters()
        await self.test_cloud_run_services()
        
        # Analysis tests
        await self.test_security_analysis()
        await self.test_recommendations()
        
        # Generate report
        self.generate_report()


async def main():
    """Main entry point"""
    tester = FullIntegrationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run tests
    asyncio.run(main())