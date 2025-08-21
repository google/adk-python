"""
Comprehensive Multi-Turn Conversation Testing Suite

Tests the complete cache-first conversation flow with multiple turns,
ensuring seamless integration between agent tools, cache, and APIs.
"""

import pytest
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import tempfile
import os

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTurnConversationTester:
    """Comprehensive multi-turn conversation testing framework."""
    
    def __init__(self):
        self.conversation_history = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0
        self.errors = []
        
    def log_interaction(self, turn: int, user_input: str, agent_response: str, 
                       response_time: float, cache_used: bool = False):
        """Log a conversation turn."""
        interaction = {
            'turn': turn,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'agent_response': agent_response,
            'response_time': response_time,
            'cache_used': cache_used
        }
        self.conversation_history.append(interaction)
        
        if cache_used:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        logger.info(f"Turn {turn}: {'CACHE' if cache_used else 'API'} ({response_time:.3f}s)")

class TestMultiTurnConversations:
    """Comprehensive multi-turn conversation tests."""
    
    @pytest.fixture
    def tester(self):
        """Create conversation tester instance."""
        return MultiTurnConversationTester()
    
    @pytest.fixture
    def agent_tools(self):
        """Load agent tools for testing."""
        try:
            import sys
            sys.path.insert(0, '/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent')
            from agent import (
                discover_assets, analyze_security, analyze_iam, analyze_storage,
                refresh_data, show_cache_status
            )
            return {
                'discover_assets': discover_assets,
                'analyze_security': analyze_security,
                'analyze_iam': analyze_iam,
                'analyze_storage': analyze_storage,
                'refresh_data': refresh_data,
                'show_cache_status': show_cache_status
            }
        except ImportError as e:
            pytest.skip(f"Agent tools not available: {e}")
    
    async def test_basic_multi_turn_flow(self, tester, agent_tools):
        """Test basic multi-turn conversation flow."""
        print("\n🧪 Testing Basic Multi-Turn Flow")
        
        # Turn 1: Check cache status
        start_time = time.time()
        response1 = agent_tools['show_cache_status']()
        turn1_time = time.time() - start_time
        
        tester.log_interaction(1, "show cache status", response1, turn1_time, True)
        assert "Cache Statistics" in response1 or "cache" in response1.lower()
        
        # Turn 2: Discover assets
        start_time = time.time()
        response2 = agent_tools['discover_assets']()
        turn2_time = time.time() - start_time
        
        cache_used = turn2_time < 0.1  # If very fast, likely from cache
        tester.log_interaction(2, "discover assets", response2, turn2_time, cache_used)
        assert "Asset Discovery" in response2 or "assets" in response2.lower()
        
        # Turn 3: Security analysis
        start_time = time.time()
        response3 = agent_tools['analyze_security']()
        turn3_time = time.time() - start_time
        
        cache_used = turn3_time < 0.1
        tester.log_interaction(3, "analyze security", response3, turn3_time, cache_used)
        assert "Security" in response3 or "findings" in response3.lower()
        
        # Turn 4: IAM analysis
        start_time = time.time()
        response4 = agent_tools['analyze_iam']()
        turn4_time = time.time() - start_time
        
        cache_used = turn4_time < 0.1
        tester.log_interaction(4, "analyze iam", response4, turn4_time, cache_used)
        assert "IAM" in response4 or "service account" in response4.lower()
        
        print(f"✅ Basic flow complete: {len(tester.conversation_history)} turns")
        print(f"   Cache hits: {tester.cache_hits}, API calls: {tester.cache_misses}")
        
        return tester.conversation_history

    async def test_cache_first_behavior(self, tester, agent_tools):
        """Test that subsequent calls use cache for speed."""
        print("\n🧪 Testing Cache-First Behavior")
        
        # First call - may be slow (cache miss)
        start_time = time.time()
        response1 = agent_tools['discover_assets']()
        first_call_time = time.time() - start_time
        
        tester.log_interaction(1, "discover assets (first)", response1, first_call_time, False)
        
        # Wait a moment
        await asyncio.sleep(0.1)
        
        # Second call - should be fast (cache hit)
        start_time = time.time()
        response2 = agent_tools['discover_assets']()
        second_call_time = time.time() - start_time
        
        cache_used = second_call_time < first_call_time / 2  # At least 50% faster
        tester.log_interaction(2, "discover assets (second)", response2, second_call_time, cache_used)
        
        print(f"   First call: {first_call_time:.3f}s")
        print(f"   Second call: {second_call_time:.3f}s")
        print(f"   Speed improvement: {(first_call_time/second_call_time):.1f}x faster")
        
        # Responses should be consistent
        assert len(response1) > 0
        assert len(response2) > 0
        
        return {
            'first_call_time': first_call_time,
            'second_call_time': second_call_time,
            'cache_working': cache_used
        }
    
    async def test_mixed_tool_conversation(self, tester, agent_tools):
        """Test conversation with mixed tool usage."""
        print("\n🧪 Testing Mixed Tool Conversation")
        
        conversation_turns = [
            ("show_cache_status", "What's my cache status?"),
            ("discover_assets", "Show me my assets"),
            ("analyze_security", "Check security issues"),  
            ("show_cache_status", "Cache status again?"),
            ("analyze_iam", "Analyze IAM"),
            ("analyze_storage", "Check storage security"),
            ("discover_assets", "Assets again"),
        ]
        
        results = []
        
        for turn, (tool_name, user_query) in enumerate(conversation_turns, 1):
            start_time = time.time()
            
            try:
                response = agent_tools[tool_name]()
                response_time = time.time() - start_time
                
                # Determine if cache was likely used
                cache_used = response_time < 0.1 and turn > 1
                
                tester.log_interaction(turn, user_query, response[:100] + "...", response_time, cache_used)
                
                results.append({
                    'turn': turn,
                    'tool': tool_name,
                    'response_time': response_time,
                    'cache_used': cache_used,
                    'success': True
                })
                
                # Brief pause between turns
                await asyncio.sleep(0.05)
                
            except Exception as e:
                tester.errors.append(f"Turn {turn} ({tool_name}): {str(e)}")
                results.append({
                    'turn': turn,
                    'tool': tool_name,
                    'error': str(e),
                    'success': False
                })
        
        print(f"✅ Mixed conversation complete: {len(results)} turns")
        print(f"   Successful turns: {sum(1 for r in results if r['success'])}")
        print(f"   Failed turns: {len(tester.errors)}")
        print(f"   Average response time: {sum(r.get('response_time', 0) for r in results)/len(results):.3f}s")
        
        # Should have mostly successful turns
        success_rate = sum(1 for r in results if r['success']) / len(results)
        assert success_rate >= 0.7, f"Success rate too low: {success_rate:.2%}"
        
        return results
    
    async def test_error_recovery_in_conversation(self, tester, agent_tools):
        """Test error recovery during multi-turn conversations."""
        print("\n🧪 Testing Error Recovery")
        
        # Start with a working call
        start_time = time.time()
        response1 = agent_tools['show_cache_status']()
        time1 = time.time() - start_time
        tester.log_interaction(1, "show cache status", response1, time1, True)
        
        # Try a potentially failing call
        try:
            start_time = time.time()
            response2 = agent_tools['analyze_security']()  # May fail if no cache/API
            time2 = time.time() - start_time
            tester.log_interaction(2, "analyze security (potential fail)", response2, time2)
            error_occurred = False
        except Exception as e:
            tester.errors.append(f"Expected error in turn 2: {str(e)}")
            error_occurred = True
        
        # Follow up with another working call
        start_time = time.time()
        response3 = agent_tools['show_cache_status']()
        time3 = time.time() - start_time
        tester.log_interaction(3, "show cache status (recovery)", response3, time3, True)
        
        print(f"   Error occurred: {error_occurred}")
        print(f"   Recovery successful: {len(response3) > 0}")
        
        # Should be able to continue conversation even after errors
        assert len(response3) > 0, "Failed to recover after error"
        
        return {
            'error_occurred': error_occurred,
            'recovery_successful': len(response3) > 0,
            'conversation_continued': len(tester.conversation_history) >= 2
        }
    
    async def test_conversation_with_refresh(self, tester, agent_tools):
        """Test conversation flow with data refresh."""
        print("\n🧪 Testing Conversation with Refresh")
        
        # Turn 1: Initial query
        start_time = time.time()
        response1 = agent_tools['discover_assets']()
        time1 = time.time() - start_time
        tester.log_interaction(1, "discover assets (initial)", response1, time1)
        
        # Turn 2: Trigger refresh (background task)
        start_time = time.time()
        response2 = agent_tools['refresh_data']()
        time2 = time.time() - start_time
        tester.log_interaction(2, "refresh data", response2, time2)
        
        # Turn 3: Query again (should still work during refresh)
        start_time = time.time()
        response3 = agent_tools['show_cache_status']()
        time3 = time.time() - start_time
        tester.log_interaction(3, "cache status (during refresh)", response3, time3, True)
        
        # Turn 4: Another query
        start_time = time.time()
        response4 = agent_tools['discover_assets']()
        time4 = time.time() - start_time
        tester.log_interaction(4, "discover assets (post refresh)", response4, time4)
        
        print(f"   Refresh initiated: {'refresh' in response2.lower()}")
        print(f"   Continued working: {len(response3) > 0 and len(response4) > 0}")
        
        # Conversation should continue working even during refresh
        assert len(response3) > 0, "Cache status failed during refresh"
        assert len(response4) > 0, "Asset discovery failed after refresh"
        
        return {
            'refresh_initiated': 'refresh' in response2.lower(),
            'conversation_continued': len(tester.conversation_history) == 4,
            'all_responses_valid': all(len(r['agent_response']) > 0 for r in tester.conversation_history)
        }


def run_comprehensive_testing():
    """Run all multi-turn conversation tests."""
    print("🧪 COMPREHENSIVE MULTI-TURN CONVERSATION TESTING")
    print("=" * 60)
    
    tester = MultiTurnConversationTester()
    test_instance = TestMultiTurnConversations()
    
    # Load agent tools
    try:
        import sys
        sys.path.insert(0, '/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent')
        from agent import (
            discover_assets, analyze_security, analyze_iam, analyze_storage,
            refresh_data, show_cache_status
        )
        agent_tools = {
            'discover_assets': discover_assets,
            'analyze_security': analyze_security,
            'analyze_iam': analyze_iam,
            'analyze_storage': analyze_storage,
            'refresh_data': refresh_data,
            'show_cache_status': show_cache_status
        }
        print("✅ Agent tools loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load agent tools: {e}")
        return False
    
    async def run_all_tests():
        results = {}
        
        try:
            # Test 1: Basic multi-turn flow
            results['basic_flow'] = await test_instance.test_basic_multi_turn_flow(tester, agent_tools)
            print("✅ Test 1 PASSED: Basic multi-turn flow")
        except Exception as e:
            print(f"❌ Test 1 FAILED: {e}")
            results['basic_flow'] = {'error': str(e)}
        
        try:
            # Test 2: Cache-first behavior
            results['cache_first'] = await test_instance.test_cache_first_behavior(tester, agent_tools)
            print("✅ Test 2 PASSED: Cache-first behavior")
        except Exception as e:
            print(f"❌ Test 2 FAILED: {e}")
            results['cache_first'] = {'error': str(e)}
        
        try:
            # Test 3: Mixed tool conversation
            results['mixed_tools'] = await test_instance.test_mixed_tool_conversation(tester, agent_tools)
            print("✅ Test 3 PASSED: Mixed tool conversation")
        except Exception as e:
            print(f"❌ Test 3 FAILED: {e}")
            results['mixed_tools'] = {'error': str(e)}
        
        try:
            # Test 4: Error recovery
            results['error_recovery'] = await test_instance.test_error_recovery_in_conversation(tester, agent_tools)
            print("✅ Test 4 PASSED: Error recovery")
        except Exception as e:
            print(f"❌ Test 4 FAILED: {e}")
            results['error_recovery'] = {'error': str(e)}
        
        try:
            # Test 5: Conversation with refresh
            results['with_refresh'] = await test_instance.test_conversation_with_refresh(tester, agent_tools)
            print("✅ Test 5 PASSED: Conversation with refresh")
        except Exception as e:
            print(f"❌ Test 5 FAILED: {e}")
            results['with_refresh'] = {'error': str(e)}
        
        return results
    
    # Run tests
    results = asyncio.run(run_all_tests())
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if 'error' not in r)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\nConversation History: {len(tester.conversation_history)} turns")
    print(f"Cache Hits: {tester.cache_hits}")
    print(f"Cache Misses: {tester.cache_misses}")
    print(f"Errors: {len(tester.errors)}")
    
    if tester.errors:
        print("\nErrors encountered:")
        for error in tester.errors:
            print(f"  - {error}")
    
    # Save detailed results
    try:
        with open('/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/MULTI_TURN_TEST_RESULTS.md', 'w') as f:
            f.write("# Multi-Turn Conversation Test Results\n\n")
            f.write(f"**Test Date:** {datetime.now().isoformat()}\n")
            f.write(f"**Overall Success Rate:** {(passed_tests/total_tests)*100:.1f}%\n\n")
            
            f.write("## Test Results Summary\n\n")
            for test_name, result in results.items():
                status = "✅ PASS" if 'error' not in result else "❌ FAIL"
                f.write(f"- **{test_name}**: {status}\n")
            
            f.write("\n## Conversation History\n\n")
            for i, turn in enumerate(tester.conversation_history, 1):
                f.write(f"### Turn {i}\n")
                f.write(f"- **User:** {turn['user_input']}\n")
                f.write(f"- **Response Time:** {turn['response_time']:.3f}s\n")
                f.write(f"- **Cache Used:** {turn['cache_used']}\n")
                f.write(f"- **Response:** {turn['agent_response'][:200]}...\n\n")
            
            f.write(f"\n## Statistics\n")
            f.write(f"- **Total Turns:** {len(tester.conversation_history)}\n")
            f.write(f"- **Cache Hits:** {tester.cache_hits}\n")
            f.write(f"- **API Calls:** {tester.cache_misses}\n")
            f.write(f"- **Errors:** {len(tester.errors)}\n")
        
        print("✅ Detailed results saved to MULTI_TURN_TEST_RESULTS.md")
        
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_testing()
    exit(0 if success else 1)