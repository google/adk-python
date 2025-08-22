#!/usr/bin/env python3
"""
Test script for MSA Analyzer workflow
=====================================

This script tests the complete MSA analysis workflow including:
1. Backend API availability
2. Sample MSA retrieval
3. MSA analysis with pattern detection
4. Results formatting and display

Usage:
    python test_msa_workflow.py
"""

import requests
import json
import sys

def test_msa_workflow():
    """Test the complete MSA analyzer workflow."""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing MSA Analyzer Workflow")
    print("=" * 40)
    
    # Test 1: Backend Health Check
    print("\n1. 🔍 Testing Backend Health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
        else:
            print(f"⚠️ Backend health check returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False
    
    # Test 2: Sample MSA Retrieval
    print("\n2. 📋 Testing Sample MSA Retrieval...")
    try:
        response = requests.get(f"{base_url}/api/v1/msa/sample", timeout=10)
        if response.status_code == 200:
            sample_data = response.json()
            sample_msa = sample_data.get("sample_msa", "")
            print(f"✅ Retrieved sample MSA ({len(sample_msa)} characters)")
            print(f"📄 Sample preview: {sample_msa[:100]}...")
        else:
            print(f"❌ Sample MSA retrieval failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Sample MSA retrieval error: {e}")
        return False
    
    # Test 3: MSA Analysis with BigQuery Pattern
    print("\n3. 🔍 Testing MSA Analysis...")
    test_msa_content = """
    Subject: BigQuery dataset Access Control Lists Permission Changes
    
    Dear Google Cloud Customer,
    
    We are updating BigQuery dataset Access Control Lists permissions.
    
    The permission bigquery.datasets.get will be modified on March 17, 2026.
    After this date, it will only allow viewing metadata.
    To view ACLs, you'll need the new permission bigquery.datasets.getIamPolicy.
    
    Action Required: Update your IAM policies before the effective date.
    
    Best regards,
    Google Cloud Team
    """
    
    try:
        analysis_payload = {
            "email_content": test_msa_content,
            "project_id": "test-project-123"
        }
        
        response = requests.post(
            f"{base_url}/api/v1/msa/analyze",
            json=analysis_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ MSA analysis completed successfully")
            print(f"📊 Results summary:")
            print(f"   - Success: {results.get('success')}")
            print(f"   - Changes detected: {len(results.get('extracted_changes', []))}")
            print(f"   - Impact assessments: {len(results.get('impact_assessments', []))}")
            print(f"   - Recommendations: {len(results.get('recommendations', []))}")
            
            # Display detected changes
            changes = results.get('extracted_changes', [])
            if changes:
                print(f"\n🔍 Detected Changes:")
                for i, change in enumerate(changes, 1):
                    print(f"   {i}. {change.get('service')} - {change.get('change_type')}")
                    print(f"      Impact: {change.get('impact_level')}")
                    print(f"      Date: {change.get('effective_date')}")
            else:
                print("⚠️ No changes detected - this may indicate pattern matching issues")
            
            # Display recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for rec in recommendations:
                    print(f"   - {rec}")
            
        else:
            print(f"❌ MSA analysis failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ MSA analysis error: {e}")
        return False
    
    # Test 4: Sample MSA Analysis
    print("\n4. 📋 Testing with Full Sample MSA...")
    try:
        analysis_payload = {
            "email_content": sample_msa,
            "project_id": "test-project-123"
        }
        
        response = requests.post(
            f"{base_url}/api/v1/msa/analyze",
            json=analysis_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            changes = results.get('extracted_changes', [])
            print(f"✅ Sample MSA analysis: {len(changes)} changes detected")
            
            if changes:
                print("🎯 Change types found:")
                for change in changes:
                    print(f"   - {change.get('service')}: {change.get('change_type')}")
        else:
            print(f"⚠️ Sample MSA analysis returned: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Sample MSA analysis error: {e}")
    
    print("\n" + "=" * 40)
    print("✅ MSA Analyzer workflow test completed!")
    print("\n📖 To test the frontend:")
    print("   1. Open http://localhost:8501")
    print("   2. Navigate to 'MSA Analyzer' tab")
    print("   3. Click 'Load Sample MSA'")
    print("   4. Click 'Analyze MSA Impact'")
    print("   5. Verify results display correctly")
    
    return True

if __name__ == "__main__":
    success = test_msa_workflow()
    sys.exit(0 if success else 1)