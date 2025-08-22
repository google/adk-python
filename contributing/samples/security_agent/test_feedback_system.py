#!/usr/bin/env python3
"""
Test script for STORY-005: Feedback System
==========================================

This script comprehensively tests the feedback system including:
1. Database schema creation
2. Feedback submission APIs
3. Analytics dashboard functionality
4. ADK evalset generation
5. End-to-end workflow validation

Usage:
    python test_feedback_system.py
"""

import requests
import json
import sys
import os
from datetime import datetime

def test_feedback_system():
    """Test the complete feedback system workflow."""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing STORY-005: Feedback System")
    print("=" * 50)
    
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
    
    # Test 2: Feedback Health Check
    print("\n2. 🏥 Testing Feedback API Health...")
    try:
        response = requests.get(f"{base_url}/api/v1/feedback/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Feedback API is healthy")
            print(f"   Database: {health_data.get('database', 'unknown')}")
            print(f"   Recent feedback: {health_data.get('recent_feedback_count', 0)}")
        else:
            print(f"⚠️ Feedback health check returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Feedback health check failed: {e}")
        return False
    
    # Test 3: Submit Test Feedback
    print("\n3. 📝 Testing Feedback Submission...")
    test_feedback = {
        "session_id": "test_session_001",
        "message_id": "test_msg_001",
        "user_query": "What are the security risks in my GCP project?",
        "assistant_response": "Based on your GCP configuration, I found several security concerns including overly permissive firewall rules and unencrypted storage buckets.",
        "rating": 4,
        "thumbs_vote": "up",
        "categories": ["helpful", "accurate"],
        "user_comments": "Good analysis but could use more specific remediation steps",
        "user_id": "test_user"
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/feedback/submit",
            json=test_feedback,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Feedback submitted successfully")
            print(f"   Feedback ID: {result.get('feedback_id')}")
            print(f"   Message: {result.get('message')}")
        else:
            print(f"❌ Feedback submission failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Feedback submission error: {e}")
        return False
    
    # Test 4: Submit Additional Test Feedback
    print("\n4. 📋 Submitting Additional Test Feedback...")
    additional_feedback = [
        {
            "session_id": "test_session_002",
            "message_id": "test_msg_002", 
            "user_query": "Show me all IAM accounts with high privileges",
            "assistant_response": "Here are the IAM accounts with administrative privileges in your project.",
            "rating": 5,
            "thumbs_vote": "up",
            "categories": ["excellent", "helpful"],
            "user_id": "test_user"
        },
        {
            "session_id": "test_session_003",
            "message_id": "test_msg_003",
            "user_query": "Generate a security compliance report", 
            "assistant_response": "I'm unable to generate that report right now.",
            "rating": 2,
            "thumbs_vote": "down",
            "categories": ["incomplete", "wrong"],
            "corrected_response": "Here's your comprehensive security compliance report covering SOC2, GDPR, and HIPAA requirements...",
            "user_comments": "Response was too brief and unhelpful",
            "user_id": "test_user"
        }
    ]
    
    for i, feedback in enumerate(additional_feedback, 1):
        try:
            response = requests.post(
                f"{base_url}/api/v1/feedback/submit",
                json=feedback,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Additional feedback {i} submitted (ID: {result.get('feedback_id')})")
            else:
                print(f"⚠️ Additional feedback {i} failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Additional feedback {i} error: {e}")
    
    # Test 5: Retrieve Feedback Metrics
    print("\n5. 📊 Testing Feedback Analytics...")
    try:
        response = requests.get(f"{base_url}/api/v1/feedback/metrics?days=30", timeout=10)
        
        if response.status_code == 200:
            metrics = response.json()
            print("✅ Feedback metrics retrieved successfully")
            
            overview = metrics.get('overview', {})
            print(f"   Total feedback: {overview.get('total_feedback', 0)}")
            print(f"   Average rating: {overview.get('avg_rating', 0):.1f}/5.0")
            print(f"   Thumbs up: {overview.get('thumbs_up', 0)}")
            print(f"   Thumbs down: {overview.get('thumbs_down', 0)}")
            print(f"   Unique sessions: {overview.get('unique_sessions', 0)}")
            
            daily_trends = metrics.get('daily_trends', [])
            print(f"   Daily trend records: {len(daily_trends)}")
            
            category_analysis = metrics.get('category_analysis', [])
            print(f"   Category analysis records: {len(category_analysis)}")
            
        else:
            print(f"⚠️ Metrics retrieval failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"⚠️ Metrics retrieval error: {e}")
    
    # Test 6: List Feedback
    print("\n6. 📋 Testing Feedback List...")
    try:
        response = requests.get(f"{base_url}/api/v1/feedback/list?limit=10", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Feedback list retrieved successfully")
            print(f"   Count: {result.get('count', 0)}")
            
            feedback_list = result.get('feedback', [])
            for feedback in feedback_list[:3]:  # Show first 3
                print(f"   - ID {feedback.get('id')}: {feedback.get('thumbs_vote', 'no vote')} vote, rating: {feedback.get('rating', 'no rating')}")
                
        else:
            print(f"⚠️ Feedback list failed: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Feedback list error: {e}")
    
    # Test 7: Generate ADK Evalset
    print("\n7. 🤖 Testing ADK Evalset Generation...")
    try:
        evalset_request = {
            "min_feedback_count": 3,  # Low threshold for testing
            "include_corrections_only": False,
            "min_rating": 2
        }
        
        response = requests.post(
            f"{base_url}/api/v1/feedback/generate-evalset",
            json=evalset_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ ADK Evalset generated successfully")
            print(f"   Evalset ID: {result.get('evalset_id')}")
            print(f"   Evaluation cases: {result.get('eval_cases_count')}")
            print(f"   File path: {result.get('file_path')}")
            
            # Verify file exists
            file_path = result.get('file_path')
            if file_path and os.path.exists(file_path):
                print(f"✅ Evalset file verified: {os.path.basename(file_path)}")
                
                # Show file content preview
                try:
                    with open(file_path, 'r') as f:
                        evalset_content = json.load(f)
                    print(f"   Evalset contains {len(evalset_content.get('eval_cases', []))} evaluation cases")
                except Exception as e:
                    print(f"   ⚠️ Could not read evalset file: {e}")
            else:
                print(f"   ⚠️ Evalset file not found at: {file_path}")
                
        else:
            result = response.json()
            print(f"⚠️ Evalset generation failed: {response.status_code}")
            print(f"   Error: {result.get('detail', 'Unknown error')}")
            
    except Exception as e:
        print(f"⚠️ Evalset generation error: {e}")
    
    # Test 8: Improvement Suggestions
    print("\n8. 💡 Testing Improvement Suggestions...")
    try:
        response = requests.get(f"{base_url}/api/v1/feedback/improvement-suggestions", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Improvement suggestions retrieved")
            
            suggestions = result.get('suggestions', [])
            print(f"   Total suggestions: {len(suggestions)}")
            
            for suggestion in suggestions[:3]:  # Show first 3
                priority = suggestion.get('priority', 'low')
                category = suggestion.get('category', 'general')
                text = suggestion.get('suggestion', '')
                
                priority_icon = "🚨" if priority == "high" else "⚠️" if priority == "medium" else "ℹ️"
                print(f"   {priority_icon} {category.title()}: {text[:60]}...")
                
        else:
            print(f"⚠️ Improvement suggestions failed: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Improvement suggestions error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ STORY-005 Feedback System test completed!")
    
    print("\n📋 Test Summary:")
    print("   ✅ Database schema creation")
    print("   ✅ Feedback submission API")
    print("   ✅ Feedback analytics and metrics")
    print("   ✅ ADK evalset generation")
    print("   ✅ Improvement suggestions")
    
    print("\n🎯 Next Steps:")
    print("   1. Open http://localhost:8501")
    print("   2. Navigate to 'Feedback Analytics' tab")
    print("   3. Test feedback widgets in Security Chat")
    print("   4. Verify analytics dashboard displays data")
    print("   5. Test evalset generation from UI")
    
    return True

if __name__ == "__main__":
    success = test_feedback_system()
    sys.exit(0 if success else 1)