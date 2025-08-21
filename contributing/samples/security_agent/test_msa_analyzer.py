#!/usr/bin/env python3
"""
Test MSA Analyzer functionality
"""

import httpx
import json
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_msa_analyzer():
    """Test the MSA analyzer API endpoint."""
    
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    # First, get the sample MSA
    print("📋 Fetching sample MSA...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{backend_url}/api/v1/msa/sample")
            if response.status_code == 200:
                sample_data = response.json()
                sample_msa = sample_data.get("sample_msa", "")
                print("✅ Sample MSA loaded successfully")
                print(f"   Length: {len(sample_msa)} characters")
            else:
                print(f"❌ Failed to get sample: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Error fetching sample: {e}")
            return
        
        # Now analyze the sample
        print("\n🤖 Analyzing MSA with Gemini...")
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        payload = {
            "email_content": sample_msa,
            "project_id": project_id if project_id and project_id != "your-project-id" else None
        }
        
        try:
            response = await client.post(
                f"{backend_url}/api/v1/msa/analyze",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                results = response.json()
                
                print("✅ Analysis complete!")
                print("\n📊 Summary:")
                summary = results.get("summary", {})
                print(f"   Total changes: {summary.get('total_changes', 0)}")
                print(f"   Critical changes: {summary.get('critical_changes', 0)}")
                print(f"   High impact changes: {summary.get('high_impact_changes', 0)}")
                print(f"   Resources affected: {summary.get('total_resources_affected', 0)}")
                print(f"   Services affected: {', '.join(summary.get('services_affected', []))}")
                
                print("\n🔄 Extracted Changes:")
                for i, change in enumerate(results.get("extracted_changes", [])[:5], 1):
                    print(f"\n   {i}. {change['service']} - {change['change_type']}")
                    print(f"      Description: {change['description'][:100]}...")
                    print(f"      Impact Level: {change['impact_level'].upper()}")
                    if change.get('effective_date'):
                        print(f"      Effective Date: {change['effective_date']}")
                    if change.get('required_action'):
                        print(f"      Required Action: {change['required_action'][:100]}...")
                
                if project_id and results.get("impact_assessments"):
                    print(f"\n🎯 Impact on Project {project_id}:")
                    for assessment in results["impact_assessments"]:
                        print(f"   - {assessment['resource_type']}: {assessment['resource_count']} resources")
                        print(f"     Impact: {assessment['impact_level'].upper()}")
                
                print("\n💡 Overall Recommendations:")
                for rec in results.get("recommendations", []):
                    print(f"   • {rec}")
                    
            else:
                print(f"❌ Analysis failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error analyzing MSA: {e}")

if __name__ == "__main__":
    print("🧪 Testing MSA Analyzer")
    print("=" * 50)
    asyncio.run(test_msa_analyzer())
    print("\n✅ Test complete!")