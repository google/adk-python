#!/usr/bin/env python3
"""
End-to-End Chat Integration Test
================================

This script simulates actual chat conversations to verify that the knowledge base
is fully integrated and works as expected in real-world scenarios.
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "agents" / "gcp_security"))

# Import the SQLite tool
from sqlite_tool import query_security_data

def simulate_chat_conversation():
    """Simulate a realistic chat conversation about coding standards and policies"""
    
    print("=" * 70)
    print("🤖 END-TO-END CHAT INTEGRATION TEST")
    print("=" * 70)
    print("\nSimulating a conversation between a developer and the security agent...\n")
    
    conversations = [
        {
            "user": "Hi! I'm a new developer. What coding standards should I follow?",
            "agent_query": ("coding_standards", None),
            "expected_topics": ["No Hardcoded Secrets", "Test Coverage", "Resource Tagging"]
        },
        {
            "user": "What specific test requirements do we have?",
            "agent_query": ("coding_standards", '{"search": "test"}'),
            "expected_topics": ["Test Coverage Requirement", "Test Naming Convention", "Mock External Services"]
        },
        {
            "user": "Are there any critical security policies I need to know about?",
            "agent_query": ("enterprise_policies", '{"severity": "CRITICAL"}'),
            "expected_topics": ["Least Privilege Access", "CRITICAL"]
        },
        {
            "user": "What's the policy on encryption?",
            "agent_query": ("enterprise_policies", '{"search": "encryption"}'),
            "expected_topics": ["Encryption at Rest", "CMEK"]
        },
        {
            "user": "Show me best practices for Cloud Storage",
            "agent_query": ("best_practices", '{"service": "Cloud Storage"}'),
            "expected_topics": ["Enable Versioning", "Data Protection"]
        },
        {
            "user": "What compliance frameworks do we need to follow?",
            "agent_query": ("compliance", None),
            "expected_topics": ["SOC2", "PCI-DSS"]
        },
        {
            "user": "How much test coverage is required?",
            "agent_query": ("coding_standards", '{"search": "coverage"}'),
            "expected_topics": ["80%", "coverage"]
        },
        {
            "user": "What should I know about mocking in tests?",
            "agent_query": ("coding_standards", '{"search": "mock"}'),
            "expected_topics": ["Mock External Services", "External API"]
        }
    ]
    
    total_conversations = len(conversations)
    successful_conversations = 0
    
    for i, conv in enumerate(conversations, 1):
        print(f"💬 Conversation {i}/{total_conversations}")
        print("-" * 50)
        print(f"👤 Developer: {conv['user']}")
        
        # Agent processes the query
        result = query_security_data(conv['agent_query'][0], conv['agent_query'][1])
        
        # Check if expected topics are covered
        topics_found = []
        topics_missing = []
        
        for topic in conv['expected_topics']:
            if topic.lower() in result.lower():
                topics_found.append(topic)
            else:
                topics_missing.append(topic)
        
        # Display agent response (truncated)
        print(f"🤖 Agent: {result[:200]}...")
        
        # Evaluate response quality
        if len(topics_missing) == 0:
            print(f"✅ All expected topics covered: {', '.join(topics_found)}")
            successful_conversations += 1
        else:
            print(f"⚠️ Missing topics: {', '.join(topics_missing)}")
            print(f"✅ Found topics: {', '.join(topics_found)}")
        
        print(f"📊 Response length: {len(result)} characters")
        print()
    
    # Overall assessment
    success_rate = (successful_conversations / total_conversations) * 100
    
    print("=" * 70)
    print("📊 CONVERSATION ANALYSIS")
    print("=" * 70)
    print(f"\n🎯 Results:")
    print(f"  • Total Conversations: {total_conversations}")
    print(f"  • Successful Responses: {successful_conversations}")
    print(f"  • Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print(f"\n🎉 PERFECT! All conversations handled successfully!")
        print(f"The knowledge base integration is working flawlessly in real-world scenarios.")
    elif success_rate >= 80:
        print(f"\n🌟 EXCELLENT! Most conversations handled well.")
        print(f"The knowledge base integration is highly effective.")
    else:
        print(f"\n⚠️ Some conversations need improvement.")
    
    return success_rate


def test_specific_developer_workflows():
    """Test specific workflows developers might follow"""
    
    print("\n" + "=" * 70)
    print("🔧 DEVELOPER WORKFLOW TESTS")
    print("=" * 70)
    
    workflows = [
        {
            "name": "New Project Setup",
            "steps": [
                ("What coding standards apply to Python?", "coding_standards", '{"language": "Python"}'),
                ("What security policies must I follow?", "enterprise_policies", None),
                ("Any compliance requirements?", "compliance", None)
            ]
        },
        {
            "name": "Code Review Preparation", 
            "steps": [
                ("What test standards do I need to meet?", "coding_standards", '{"search": "test"}'),
                ("How should I handle secrets in code?", "coding_standards", '{"search": "secret"}'),
                ("What are the critical policies?", "enterprise_policies", '{"severity": "CRITICAL"}')
            ]
        },
        {
            "name": "Security Audit Preparation",
            "steps": [
                ("Show all enterprise policies", "enterprise_policies", None),
                ("What's our SOC2 compliance status?", "compliance", '{"framework": "SOC2"}'),
                ("Any encryption requirements?", "enterprise_policies", '{"search": "encryption"}')
            ]
        }
    ]
    
    workflow_success = []
    
    for workflow in workflows:
        print(f"\n🔄 Testing Workflow: {workflow['name']}")
        print("-" * 40)
        
        workflow_steps_passed = 0
        total_steps = len(workflow['steps'])
        
        for step_num, (question, query_type, params) in enumerate(workflow['steps'], 1):
            print(f"  Step {step_num}: {question}")
            
            result = query_security_data(query_type, params)
            
            # Basic validation - did we get a meaningful response?
            if len(result) > 200 and ("📝" in result or "🛡️" in result or "✨" in result or "📋" in result):
                print(f"    ✅ Got meaningful response ({len(result)} chars)")
                workflow_steps_passed += 1
            else:
                print(f"    ❌ Response too short or invalid format ({len(result)} chars)")
        
        workflow_success_rate = (workflow_steps_passed / total_steps) * 100
        workflow_success.append(workflow_success_rate)
        
        print(f"  📊 Workflow Success: {workflow_success_rate:.1f}% ({workflow_steps_passed}/{total_steps} steps)")
    
    overall_workflow_success = sum(workflow_success) / len(workflow_success)
    
    print(f"\n📈 Overall Workflow Success: {overall_workflow_success:.1f}%")
    
    return overall_workflow_success


def main():
    """Run all end-to-end tests"""
    
    # Test conversations
    conversation_success = simulate_chat_conversation()
    
    # Test workflows
    workflow_success = test_specific_developer_workflows()
    
    # Final assessment
    overall_success = (conversation_success + workflow_success) / 2
    
    print("\n" + "=" * 70)
    print("🏆 FINAL END-TO-END ASSESSMENT")
    print("=" * 70)
    
    print(f"\n📊 Summary:")
    print(f"  • Chat Conversations: {conversation_success:.1f}%")
    print(f"  • Developer Workflows: {workflow_success:.1f}%")
    print(f"  • Overall Success: {overall_success:.1f}%")
    
    if overall_success >= 95:
        print(f"\n🎉 OUTSTANDING! Knowledge base integration exceeds expectations!")
        print(f"✨ Ready for immediate production deployment!")
        return 0
    elif overall_success >= 85:
        print(f"\n🌟 EXCELLENT! Knowledge base integration is production-ready!")
        return 0
    elif overall_success >= 75:
        print(f"\n👍 GOOD! Knowledge base integration is functional with minor improvements needed.")
        return 0
    else:
        print(f"\n⚠️ Needs significant improvement before production use.")
        return 1


if __name__ == "__main__":
    exit(main())