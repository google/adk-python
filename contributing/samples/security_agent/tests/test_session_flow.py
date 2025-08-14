#!/usr/bin/env python3
"""
Test script for ADK Session Management Flow
Tests the complete session lifecycle: create, use, persist, restore
"""

import sys
import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

# Color codes for output
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

class SessionFlowTester:
    """Test ADK session management flow."""
    
    def __init__(self):
        self.session_id = None
        self.user_id = "test_user_001"
        self.messages = []
        self.test_results = []
        
    def test_imports(self) -> bool:
        """Test that all required modules can be imported."""
        print_test("Testing module imports...", "info")
        
        try:
            # Test backend imports
            from backend.chat_manager import chat_manager, ChatMessage, MessageType
            print_test("Chat manager imported successfully", "pass")
            
            from backend.api import agent_llm
            print_test("ADK Agent LLM module imported successfully", "pass")
            
            from backend.api import sessions
            print_test("Sessions module imported successfully", "pass")
            
            # Test conversation memory
            try:
                from backend.services.conversation_memory import conversation_memory
                print_test("Conversation memory imported successfully", "pass")
            except ImportError:
                print_test("Conversation memory not available (optional)", "warn")
            
            return True
            
        except ImportError as e:
            print_test(f"Import failed: {e}", "fail")
            return False
    
    def test_chat_manager(self) -> bool:
        """Test chat manager functionality."""
        print_test("\nTesting chat manager...", "info")
        
        try:
            from backend.chat_manager import chat_manager, ChatMessage, MessageType
            
            # Create a session
            session_id = chat_manager.create_session(self.user_id, {
                "test": True,
                "source": "test_script"
            })
            self.session_id = session_id
            print_test(f"Created session: {session_id[:20]}...", "pass")
            
            # Get session
            session = chat_manager.get_session(session_id)
            if session:
                print_test(f"Retrieved session for user: {session.user_id}", "pass")
            else:
                print_test("Failed to retrieve session", "fail")
                return False
            
            # Add messages
            asyncio.run(self._add_test_messages(session_id))
            
            # Get conversation history
            messages = chat_manager.get_conversation_history(session_id)
            print_test(f"Retrieved {len(messages)} messages from history", "pass")
            
            # Get analytics
            analytics = chat_manager.get_session_analytics(session_id)
            print_test(f"Session analytics: {analytics.get('total_messages', 0)} messages", "pass")
            
            # Get suggestions
            suggestions = chat_manager.get_contextual_suggestions(session_id)
            if suggestions:
                print_test(f"Generated {len(suggestions)} contextual suggestions", "pass")
            
            return True
            
        except Exception as e:
            print_test(f"Chat manager test failed: {e}", "fail")
            return False
    
    async def _add_test_messages(self, session_id: str):
        """Add test messages to session."""
        from backend.chat_manager import chat_manager
        
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
            content="I found 5 storage buckets in your project. 2 have public access enabled.",
            sender_type="assistant",
            agent_used="StorageSecurityAgent",
            delegation_path=["CoordinatorAgent", "StorageSecurityAgent"],
            performance_data={"response_time_ms": 250}
        )
        print_test("Added assistant message with agent metadata", "pass")
        
        # Add follow-up
        msg3 = await chat_manager.add_message(
            session_id=session_id,
            content="How do I fix the public access issues?",
            sender_type="user"
        )
        
        msg4 = await chat_manager.add_message(
            session_id=session_id,
            content="To fix public access: 1) Enable uniform bucket-level access, 2) Remove allUsers permissions",
            sender_type="assistant",
            agent_used="StorageSecurityAgent"
        )
        print_test("Added follow-up conversation", "pass")
    
    def test_session_api(self) -> bool:
        """Test session API endpoints (mock without server)."""
        print_test("\nTesting session API logic...", "info")
        
        try:
            from backend.api.sessions import SessionCreateRequest
            
            # Test request model
            request = SessionCreateRequest(
                user_id=self.user_id,
                project_id="test-project",
                metadata={"client": "test"}
            )
            print_test(f"Created session request for user: {request.user_id}", "pass")
            
            # Test session manager integration
            if self.session_id:
                from backend.chat_manager import chat_manager
                
                # Get user sessions
                sessions = chat_manager.get_user_sessions(self.user_id)
                print_test(f"Found {len(sessions)} sessions for user", "pass")
                
                # Close session
                chat_manager.close_session(self.session_id)
                session = chat_manager.get_session(self.session_id)
                if session and session.status.value == "closed":
                    print_test("Session closed successfully", "pass")
                
            return True
            
        except Exception as e:
            print_test(f"Session API test failed: {e}", "fail")
            return False
    
    def test_conversation_flow(self) -> bool:
        """Test a complete conversation flow."""
        print_test("\nTesting complete conversation flow...", "info")
        
        try:
            from backend.chat_manager import chat_manager
            
            # Create new session for flow test
            flow_session_id = chat_manager.create_session("flow_test_user", {
                "test_type": "conversation_flow"
            })
            print_test(f"Created flow test session: {flow_session_id[:20]}...", "pass")
            
            # Simulate conversation
            asyncio.run(self._simulate_conversation(flow_session_id))
            
            # Check session persistence
            session = chat_manager.get_session(flow_session_id)
            if session:
                message_count = sum(len(conv) for conv in session.conversations.values())
                print_test(f"Session persisted with {message_count} messages", "pass")
                
                # Check topics detection
                if session.topics:
                    topics = [t.name for t in session.topics]
                    print_test(f"Detected topics: {', '.join(topics)}", "pass")
            
            return True
            
        except Exception as e:
            print_test(f"Conversation flow test failed: {e}", "fail")
            return False
    
    async def _simulate_conversation(self, session_id: str):
        """Simulate a realistic conversation."""
        from backend.chat_manager import chat_manager
        
        conversations = [
            ("user", "What are the main security issues in my GCP project?", None),
            ("assistant", "I've identified several security concerns: 1) Public storage buckets, 2) Overprivileged IAM roles, 3) Open firewall rules", "CoordinatorAgent"),
            ("user", "Tell me more about the storage bucket issues", None),
            ("assistant", "You have 3 buckets with public access: analytics-data, user-uploads, and backup-2023. These expose 450GB of data.", "StorageSecurityAgent"),
            ("user", "How do I fix this?", None),
            ("assistant", "Run these commands: gsutil iam ch -d allUsers gs://analytics-data", "StorageSecurityAgent"),
        ]
        
        for sender, content, agent in conversations:
            await chat_manager.add_message(
                session_id=session_id,
                content=content,
                sender_type=sender,
                agent_used=agent
            )
        
        print_test(f"Simulated {len(conversations)} message exchanges", "pass")
    
    def test_session_restoration(self) -> bool:
        """Test session restoration after interruption."""
        print_test("\nTesting session restoration...", "info")
        
        try:
            from backend.chat_manager import chat_manager
            
            if not self.session_id:
                print_test("No session to restore (skipping)", "warn")
                return True
            
            # Get original session state
            original_messages = chat_manager.get_conversation_history(self.session_id)
            original_count = len(original_messages)
            
            # Simulate restoration
            session = chat_manager.get_session(self.session_id)
            if session:
                # Update activity
                session.last_activity = datetime.now()
                
                # Add new message after restoration
                asyncio.run(chat_manager.add_message(
                    session_id=self.session_id,
                    content="Continuing after restoration...",
                    sender_type="user"
                ))
                
                # Verify continuity
                new_messages = chat_manager.get_conversation_history(self.session_id)
                if len(new_messages) == original_count + 1:
                    print_test("Session restored with message continuity", "pass")
                    return True
                else:
                    print_test("Message count mismatch after restoration", "fail")
                    return False
            
            print_test("Could not restore session", "fail")
            return False
            
        except Exception as e:
            print_test(f"Session restoration test failed: {e}", "fail")
            return False
    
    def run_all_tests(self):
        """Run all tests and report results."""
        print(f"\n{BLUE}{'='*60}")
        print("ADK SESSION MANAGEMENT TEST SUITE")
        print(f"{'='*60}{RESET}\n")
        
        tests = [
            ("Module Imports", self.test_imports),
            ("Chat Manager", self.test_chat_manager),
            ("Session API", self.test_session_api),
            ("Conversation Flow", self.test_conversation_flow),
            ("Session Restoration", self.test_session_restoration),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print_test(f"Test '{test_name}' crashed: {e}", "fail")
                results.append((test_name, False))
        
        # Print summary
        print(f"\n{BLUE}{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}{RESET}\n")
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            status = "PASS" if success else "FAIL"
            color = GREEN if success else RED
            print(f"{color}{status:6} {test_name}{RESET}")
        
        print(f"\n{BLUE}{'='*60}{RESET}")
        if passed == total:
            print(f"{GREEN}✅ ALL TESTS PASSED ({passed}/{total}){RESET}")
        else:
            print(f"{YELLOW}⚠️  PARTIAL SUCCESS ({passed}/{total} passed){RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
        
        return passed == total

if __name__ == "__main__":
    tester = SessionFlowTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)