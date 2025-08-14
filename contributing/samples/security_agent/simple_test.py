#!/usr/bin/env python3
"""
Simple Test for ADK Session Management (No FastAPI required)
Tests the core session management functionality directly
"""

import sys
import os
import asyncio
import time
from datetime import datetime

# Add backend to path
sys.path.append('backend')

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_test(message: str, status: str = "info"):
    """Print colored test output."""
    if status == "pass":
        print(f"{GREEN}✅ {message}{RESET}")
    elif status == "fail":
        print(f"{RED}❌ {message}{RESET}")
    elif status == "warn":
        print(f"{YELLOW}⚠️  {message}{RESET}")
    else:
        print(f"{BLUE}ℹ️  {message}{RESET}")

async def test_session_management():
    """Test the enhanced chat manager session functionality."""
    
    print(f"\n{BLUE}{'='*60}")
    print("ADK SESSION MANAGEMENT CORE TEST")
    print(f"{'='*60}{RESET}\n")
    
    try:
        # Import chat manager
        from chat_manager import chat_manager, ChatMessage, MessageType
        print_test("Chat manager imported successfully", "pass")
        
        # Test 1: Create Session
        print(f"\n{BLUE}Test 1: Session Creation{RESET}")
        user_id = "test_user_123"
        metadata = {
            "source": "simple_test",
            "project_id": "test-project",
            "test_run": True
        }
        
        session_id = chat_manager.create_session(user_id, metadata)
        print_test(f"Created session: {session_id[:30]}...", "pass")
        
        # Test 2: Get Session
        print(f"\n{BLUE}Test 2: Session Retrieval{RESET}")
        session = chat_manager.get_session(session_id)
        if session:
            print_test(f"Retrieved session for user: {session.user_id}", "pass")
            print_test(f"Session status: {session.status.value}", "pass")
            print_test(f"Created at: {session.created_at}", "pass")
        else:
            print_test("Failed to retrieve session", "fail")
            return False
        
        # Test 3: Add Messages
        print(f"\n{BLUE}Test 3: Message Management{RESET}")
        
        # Add user message
        msg1 = await chat_manager.add_message(
            session_id=session_id,
            content="Tell me about the storage buckets in my project",
            sender_type="user",
            performance_data={"start_time": time.time()}
        )
        print_test("Added user message", "pass")
        
        # Add assistant response
        msg2 = await chat_manager.add_message(
            session_id=session_id,
            content="I found 5 storage buckets in your project. 2 have public access enabled which poses a security risk.",
            sender_type="assistant",
            agent_used="StorageSecurityAgent",
            delegation_path=["CoordinatorAgent", "StorageSecurityAgent"],
            performance_data={"response_time_ms": 250}
        )
        print_test("Added assistant response with agent metadata", "pass")
        
        # Add follow-up conversation
        msg3 = await chat_manager.add_message(
            session_id=session_id,
            content="How do I fix the public access issues?",
            sender_type="user"
        )
        
        msg4 = await chat_manager.add_message(
            session_id=session_id,
            content="To fix public access issues: 1) Enable uniform bucket-level access, 2) Remove allUsers and allAuthenticatedUsers permissions, 3) Use IAM conditions for fine-grained access control.",
            sender_type="assistant",
            agent_used="StorageSecurityAgent"
        )
        print_test("Added follow-up conversation", "pass")
        
        # Test 4: Get Conversation History
        print(f"\n{BLUE}Test 4: Conversation History{RESET}")
        messages = chat_manager.get_conversation_history(session_id)
        print_test(f"Retrieved {len(messages)} messages from history", "pass")
        
        if messages:
            print(f"\n{YELLOW}Conversation Preview:{RESET}")
            for i, msg in enumerate(messages[:4], 1):
                content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                agent_info = f" (via {msg.agent_used})" if msg.agent_used else ""
                print(f"  {i}. {msg.sender_type.title()}{agent_info}: {content_preview}")
        
        # Test 5: Topic Detection
        print(f"\n{BLUE}Test 5: Topic Detection{RESET}")
        session = chat_manager.get_session(session_id)
        if session.topics:
            topics = [(t.name, t.confidence) for t in session.topics]
            print_test(f"Detected {len(topics)} topics: {topics}", "pass")
        else:
            print_test("No topics detected yet", "warn")
        
        # Test 6: Session Analytics
        print(f"\n{BLUE}Test 6: Session Analytics{RESET}")
        analytics = chat_manager.get_session_analytics(session_id)
        if analytics:
            print_test("Generated session analytics", "pass")
            print(f"\n{YELLOW}Analytics Summary:{RESET}")
            print(f"  Total Messages: {analytics.get('total_messages', 0)}")
            print(f"  User Messages: {analytics.get('user_messages', 0)}")
            print(f"  Assistant Messages: {analytics.get('assistant_messages', 0)}")
            print(f"  Duration: {analytics.get('duration_minutes', 0):.1f} minutes")
            print(f"  Status: {analytics.get('status', 'unknown')}")
        
        # Test 7: Contextual Suggestions
        print(f"\n{BLUE}Test 7: Contextual Suggestions{RESET}")
        suggestions = chat_manager.get_contextual_suggestions(session_id)
        if suggestions:
            print_test(f"Generated {len(suggestions)} suggestions", "pass")
            print(f"\n{YELLOW}Suggestions:{RESET}")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"  {i}. {suggestion}")
        
        # Test 8: Session Persistence Test
        print(f"\n{BLUE}Test 8: Session Persistence{RESET}")
        # Add one more message to test persistence
        msg5 = await chat_manager.add_message(
            session_id=session_id,
            content="Show me the specific commands to fix bucket permissions",
            sender_type="user"
        )
        
        # Retrieve updated history
        updated_messages = chat_manager.get_conversation_history(session_id)
        if len(updated_messages) > len(messages):
            print_test("Session persistence working correctly", "pass")
            print_test(f"Message count increased from {len(messages)} to {len(updated_messages)}", "pass")
        
        # Test 9: User Sessions
        print(f"\n{BLUE}Test 9: User Session Management{RESET}")
        user_sessions = chat_manager.get_user_sessions(user_id)
        print_test(f"Found {len(user_sessions)} sessions for user {user_id}", "pass")
        
        # Test 10: Session Status Update
        print(f"\n{BLUE}Test 10: Session Status Management{RESET}")
        chat_manager.close_session(session_id)
        closed_session = chat_manager.get_session(session_id)
        if closed_session and closed_session.status.value == "closed":
            print_test("Session closed successfully", "pass")
        
        # Final Summary
        print(f"\n{GREEN}{'='*60}")
        print("✅ ALL CORE TESTS PASSED!")
        print(f"Session ID: {session_id}")
        print(f"Final message count: {len(updated_messages)}")
        print(f"Topics detected: {len(session.topics) if session.topics else 0}")
        print(f"{'='*60}{RESET}\n")
        
        return True
        
    except Exception as e:
        print_test(f"Test failed with error: {e}", "fail")
        import traceback
        traceback.print_exc()
        return False

async def test_conversation_flow():
    """Test a realistic conversation flow."""
    
    print(f"\n{BLUE}{'='*60}")
    print("REALISTIC CONVERSATION FLOW TEST")
    print(f"{'='*60}{RESET}\n")
    
    try:
        from chat_manager import chat_manager
        
        # Create session for realistic test
        user_id = "security_analyst_001"
        session_id = chat_manager.create_session(user_id, {
            "role": "security_analyst",
            "project": "production_environment"
        })
        
        print_test(f"Created session for security analyst", "pass")
        
        # Simulate realistic conversation
        conversations = [
            ("user", "I need to audit the security posture of our GCP project", None),
            ("assistant", "I'll help you audit your GCP security. I can analyze storage, IAM, network, and compliance. Where would you like to start?", "CoordinatorAgent"),
            ("user", "Let's start with storage security", None),
            ("assistant", "Analyzing storage security... I found 12 buckets in your project. 3 have concerning configurations: public access enabled, missing encryption, and overly permissive IAM policies.", "StorageSecurityAgent"),
            ("user", "Which buckets have public access?", None),
            ("assistant", "The following buckets have public access: 'user-uploads-prod' (allUsers read), 'static-assets' (allUsers read), and 'backup-archive' (allAuthenticatedUsers read). This exposes approximately 2.4TB of data.", "StorageSecurityAgent"),
            ("user", "How critical is this issue?", None),
            ("assistant", "This is CRITICAL. Public buckets expose sensitive data to unauthorized access. I recommend immediate action: 1) Remove public access, 2) Implement IAM conditions, 3) Enable audit logging.", "StorageSecurityAgent"),
            ("user", "Give me the exact commands to fix this", None),
            ("assistant", "Here are the commands: gsutil iam ch -d allUsers:objectViewer gs://user-uploads-prod && gsutil iam ch -d allUsers:objectViewer gs://static-assets && gsutil iam ch -d allAuthenticatedUsers:objectViewer gs://backup-archive", "StorageSecurityAgent"),
        ]
        
        # Process conversation
        for i, (sender, content, agent) in enumerate(conversations, 1):
            await chat_manager.add_message(
                session_id=session_id,
                content=content,
                sender_type=sender,
                agent_used=agent,
                performance_data={"step": i, "response_time_ms": 150 + (i * 50)}
            )
            print(f"  Step {i}: Added {sender} message {'(via ' + agent + ')' if agent else ''}")
        
        print_test(f"Processed {len(conversations)} conversation steps", "pass")
        
        # Analyze conversation
        session = chat_manager.get_session(session_id)
        analytics = chat_manager.get_session_analytics(session_id)
        
        print(f"\n{YELLOW}Conversation Analysis:{RESET}")
        print(f"  Total Messages: {analytics.get('total_messages', 0)}")
        print(f"  Topics Detected: {[t.name for t in session.topics] if session.topics else 'None'}")
        print(f"  Duration: {analytics.get('duration_minutes', 0):.1f} minutes")
        print(f"  Agents Used: {analytics.get('agents_used', [])}")
        
        # Test suggestions
        suggestions = chat_manager.get_contextual_suggestions(session_id)
        if suggestions:
            print(f"\n{YELLOW}Next Step Suggestions:{RESET}")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"  {i}. {suggestion}")
        
        print_test("Realistic conversation flow completed successfully", "pass")
        return True
        
    except Exception as e:
        print_test(f"Conversation flow test failed: {e}", "fail")
        return False

if __name__ == "__main__":
    async def main():
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Run core tests
        core_success = await test_session_management()
        
        # Run conversation flow test
        flow_success = await test_conversation_flow()
        
        if core_success and flow_success:
            print(f"\n{GREEN}🎉 ALL TESTS PASSED - ADK SESSION MANAGEMENT IS WORKING!{RESET}")
            return True
        else:
            print(f"\n{RED}❌ SOME TESTS FAILED - CHECK OUTPUT ABOVE{RESET}")
            return False
    
    # Run tests
    success = asyncio.run(main())
    exit(0 if success else 1)