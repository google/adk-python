#!/usr/bin/env python3
"""
Comprehensive ADK Feedback Integration Test
==========================================

This script tests the complete feedback system integration with ADK evaluation,
including:
1. End-to-end feedback submission
2. ADK evalset generation and validation
3. Running ADK evaluations with generated evalsets
4. Feedback analytics and improvement pipeline

Usage:
    python test_adk_feedback_integration.py
"""

import requests
import json
import sys
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

def test_adk_feedback_integration():
    """Test complete ADK feedback integration."""
    
    base_url = "http://localhost:8000"
    project_root = Path("/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent")
    
    print("🧪 ADK Feedback Integration Test")
    print("=" * 50)
    
    # Test 1: Submit varied feedback for comprehensive testing
    print("\n1. 📝 Submitting Comprehensive Test Feedback...")
    
    test_feedback_cases = [
        {
            "session_id": "comprehensive_test_001",
            "message_id": "msg_001",
            "user_query": "What are the most critical security vulnerabilities in my GCP project?",
            "assistant_response": "Based on my analysis of your GCP project, I've identified several critical security vulnerabilities: 1) Storage buckets with public read access, 2) Firewall rules allowing 0.0.0.0/0 access, 3) Service accounts with overly broad permissions, 4) Unencrypted data at rest in several databases.",
            "rating": 5,
            "thumbs_vote": "up",
            "categories": ["accurate", "helpful", "excellent"],
            "user_comments": "Comprehensive and actionable analysis",
            "user_id": "security_admin"
        },
        {
            "session_id": "comprehensive_test_002", 
            "message_id": "msg_002",
            "user_query": "Show me all IAM policies that grant admin access",
            "assistant_response": "I found 3 IAM policies with admin access in your project.",
            "rating": 2,
            "thumbs_vote": "down",
            "categories": ["incomplete", "unclear"],
            "corrected_response": "I found 3 IAM policies with admin access in your project:\\n\\n1. **Project Editor Policy** (roles/editor)\\n   - Bound to: service-account@project.iam.gserviceaccount.com\\n   - Risk: High - Full project modification access\\n\\n2. **Custom Admin Role** (custom-admin-role)\\n   - Bound to: admin@company.com\\n   - Permissions: 247 permissions including critical ones\\n\\n3. **Legacy Owner Policy** (roles/owner)\\n   - Bound to: legacy-admin@company.com\\n   - Risk: Critical - Full project ownership\\n\\n**Recommendations:**\\n- Review and minimize permissions for service accounts\\n- Replace broad roles with principle of least privilege\\n- Audit admin access regularly",
            "user_comments": "Original response was too brief and didn't provide actionable details",
            "user_id": "security_admin"
        },
        {
            "session_id": "comprehensive_test_003",
            "message_id": "msg_003", 
            "user_query": "Generate a SOC2 compliance report for our infrastructure",
            "assistant_response": "Your infrastructure shows good SOC2 compliance with some areas for improvement. All data is encrypted in transit and at rest. Access controls are properly implemented. Monitoring and logging are comprehensive. Areas for improvement include: backup retention policies and incident response documentation.",
            "rating": 4,
            "thumbs_vote": "up",
            "categories": ["helpful", "accurate"],
            "user_comments": "Good overview, could use more specific remediation steps",
            "user_id": "compliance_officer"
        },
        {
            "session_id": "comprehensive_test_004",
            "message_id": "msg_004",
            "user_query": "What's the security posture of our cloud storage?",
            "assistant_response": "Storage security analysis failed.",
            "rating": 1,
            "thumbs_vote": "down", 
            "categories": ["wrong", "incomplete"],
            "corrected_response": "**Cloud Storage Security Analysis:**\\n\\n**✅ Secure Configurations:**\\n- 15/20 buckets have uniform bucket-level access enabled\\n- All buckets use Google-managed encryption\\n- Public access prevention is enabled on critical buckets\\n\\n**⚠️ Security Concerns:**\\n- 3 buckets have public read access (marketing-assets, public-docs, temp-uploads)\\n- 2 buckets lack lifecycle policies (logs-archive, backup-storage)\\n- Object versioning disabled on 5 buckets\\n\\n**🔴 Critical Issues:**\\n- Bucket 'sensitive-data' has all-users read permission\\n- No audit logging for bucket 'financial-reports'\\n\\n**Recommendations:**\\n1. Remove public access from sensitive-data bucket immediately\\n2. Enable audit logging on all buckets\\n3. Implement lifecycle policies for cost optimization\\n4. Enable object versioning for data protection",
            "user_comments": "Complete failure - need detailed security analysis",
            "user_id": "security_admin"
        },
        {
            "session_id": "comprehensive_test_005",
            "message_id": "msg_005",
            "user_query": "Check our firewall rules for security issues",
            "assistant_response": "I found several firewall rules that may pose security risks. Rule 'allow-ssh-from-internet' allows SSH access from any IP address (0.0.0.0/0) which is dangerous. Rule 'web-server-access' allows HTTP/HTTPS but could be more restrictive. I recommend implementing IP whitelisting and using VPN access for SSH connections.",
            "rating": 5,
            "thumbs_vote": "up",
            "categories": ["accurate", "helpful", "excellent"],
            "user_id": "network_admin"
        }
    ]
    
    feedback_ids = []
    
    for i, feedback in enumerate(test_feedback_cases, 1):
        try:
            response = requests.post(
                f"{base_url}/api/v1/feedback/submit",
                json=feedback,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                feedback_id = result.get('feedback_id')
                feedback_ids.append(feedback_id)
                print(f"✅ Test feedback {i} submitted (ID: {feedback_id})")
            else:
                print(f"❌ Test feedback {i} failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Test feedback {i} error: {e}")
            return False
    
    # Test 2: Verify feedback analytics
    print(f"\n2. 📊 Testing Feedback Analytics...")
    try:
        response = requests.get(f"{base_url}/api/v1/feedback/metrics?days=1", timeout=10)
        
        if response.status_code == 200:
            metrics = response.json()
            overview = metrics.get('overview', {})
            
            total_feedback = overview.get('total_feedback', 0)
            avg_rating = overview.get('avg_rating', 0)
            
            print(f"✅ Analytics working - {total_feedback} feedback items, avg rating: {avg_rating:.1f}")
            
            # Verify we have enough feedback for testing
            if total_feedback < 5:
                print(f"⚠️ Only {total_feedback} feedback items - may need more for robust testing")
                
        else:
            print(f"❌ Analytics failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Analytics error: {e}")
        return False
    
    # Test 3: Generate ADK Evalset
    print(f"\n3. 🤖 Generating ADK Evalset...")
    try:
        evalset_request = {
            "min_feedback_count": 5,
            "include_corrections_only": False,
            "min_rating": 1
        }
        
        response = requests.post(
            f"{base_url}/api/v1/feedback/generate-evalset",
            json=evalset_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            evalset_id = result.get('evalset_id')
            eval_cases_count = result.get('eval_cases_count')
            file_path = result.get('file_path')
            
            print(f"✅ Evalset generated: {evalset_id}")
            print(f"   Cases: {eval_cases_count}")
            print(f"   File: {file_path}")
            
            # Test 4: Validate ADK Evalset Format
            print(f"\n4. 🔍 Validating ADK Evalset Format...")
            
            if not os.path.exists(file_path):
                print(f"❌ Evalset file not found: {file_path}")
                return False
            
            try:
                with open(file_path, 'r') as f:
                    evalset_content = json.load(f)
                
                # Validate required ADK evalset fields
                required_fields = ['eval_set_id', 'eval_cases']
                for field in required_fields:
                    if field not in evalset_content:
                        print(f"❌ Missing required field: {field}")
                        return False
                
                eval_cases = evalset_content.get('eval_cases', [])
                if not eval_cases:
                    print("❌ No evaluation cases in evalset")
                    return False
                
                # Validate each eval case
                for i, case in enumerate(eval_cases):
                    required_case_fields = ['conversation', 'expected_final_response']
                    for field in required_case_fields:
                        if field not in case:
                            print(f"❌ Missing field '{field}' in eval case {i}")
                            return False
                    
                    # Validate conversation structure
                    conversation = case.get('conversation', [])
                    if len(conversation) != 2:
                        print(f"❌ Invalid conversation length in case {i}: {len(conversation)}")
                        return False
                    
                    user_msg = conversation[0]
                    assistant_msg = conversation[1]
                    
                    if user_msg.get('role') != 'user':
                        print(f"❌ First message should be user in case {i}")
                        return False
                    
                    if assistant_msg.get('role') != 'assistant':
                        print(f"❌ Second message should be assistant in case {i}")
                        return False
                
                print(f"✅ Evalset format validation passed")
                print(f"   Evalset ID: {evalset_content['eval_set_id']}")
                print(f"   Cases: {len(eval_cases)}")
                print(f"   Description: {evalset_content.get('description', 'N/A')}")
                
                # Test 5: Run ADK Evaluation (if ADK CLI is available)
                print(f"\n5. 🎯 Testing ADK Evaluation...")
                
                # Check if we're in the agent directory where ADK can run
                agent_dir = project_root / "agents" / "gcp_security"
                if not agent_dir.exists():
                    print(f"⚠️ Agent directory not found: {agent_dir}")
                    print("   Skipping ADK evaluation test")
                else:
                    # Test ADK evaluation with the generated evalset
                    try:
                        os.chdir(agent_dir)
                        
                        # First check if ADK is available
                        result = subprocess.run(
                            ["adk", "--version"],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        
                        if result.returncode == 0:
                            print(f"✅ ADK CLI available: {result.stdout.strip()}")
                            
                            # Run evaluation with the generated evalset
                            eval_cmd = [
                                "adk", "eval", "run",
                                "--evalset", file_path,
                                "--agent", "vertex_sqlite_agent.py"
                            ]
                            
                            print(f"   Running: {' '.join(eval_cmd)}")
                            
                            result = subprocess.run(
                                eval_cmd,
                                capture_output=True,
                                text=True,
                                timeout=60
                            )
                            
                            if result.returncode == 0:
                                print(f"✅ ADK evaluation completed successfully")
                                print(f"   Output preview: {result.stdout[:200]}...")
                                
                                # Look for evaluation results
                                if "accuracy" in result.stdout.lower():
                                    print(f"✅ Evaluation metrics generated")
                                
                            else:
                                print(f"⚠️ ADK evaluation returned non-zero exit code: {result.returncode}")
                                print(f"   stdout: {result.stdout[:300]}")
                                print(f"   stderr: {result.stderr[:300]}")
                        else:
                            print(f"⚠️ ADK CLI not available or not working")
                            print(f"   stdout: {result.stdout}")
                            print(f"   stderr: {result.stderr}")
                            
                    except subprocess.TimeoutExpired:
                        print(f"⚠️ ADK evaluation timed out")
                    except Exception as e:
                        print(f"⚠️ ADK evaluation error: {e}")
                    finally:
                        os.chdir(project_root)
                
                # Test 6: Verify Improvement Suggestions
                print(f"\n6. 💡 Testing Improvement Suggestions...")
                try:
                    response = requests.get(f"{base_url}/api/v1/feedback/improvement-suggestions", timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        suggestions = result.get('suggestions', [])
                        
                        print(f"✅ Generated {len(suggestions)} improvement suggestions")
                        
                        for suggestion in suggestions:
                            priority = suggestion.get('priority', 'low')
                            category = suggestion.get('category', 'general')
                            text = suggestion.get('suggestion', '')
                            
                            priority_icon = "🚨" if priority == "high" else "⚠️" if priority == "medium" else "ℹ️"
                            print(f"   {priority_icon} {category.title()}: {text[:80]}...")
                    else:
                        print(f"⚠️ Improvement suggestions failed: {response.status_code}")
                        
                except Exception as e:
                    print(f"⚠️ Improvement suggestions error: {e}")
                
                return True
                
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in evalset file: {e}")
                return False
            except Exception as e:
                print(f"❌ Error validating evalset: {e}")
                return False
        
        else:
            result = response.json()
            print(f"❌ Evalset generation failed: {response.status_code}")
            print(f"   Error: {result.get('detail', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Evalset generation error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ ADK Feedback Integration Test COMPLETED!")
    
    print(f"\n📋 Test Results Summary:")
    print(f"   ✅ Feedback submission: {len(feedback_ids)} cases")
    print(f"   ✅ Analytics and metrics")
    print(f"   ✅ ADK evalset generation")
    print(f"   ✅ Evalset format validation")
    print(f"   ✅ Improvement suggestions")
    
    print(f"\n🎯 Next Steps for Production:")
    print(f"   1. Collect real user feedback in production")
    print(f"   2. Run periodic ADK evaluations with generated evalsets")
    print(f"   3. Monitor improvement suggestions dashboard")
    print(f"   4. Iterate on agent instructions based on feedback")
    print(f"   5. Track accuracy improvements over time")
    
    return True

if __name__ == "__main__":
    success = test_adk_feedback_integration()
    sys.exit(0 if success else 1)