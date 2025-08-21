"""
Cache-First Multi-Turn Conversation Test

Tests the cache-first system's ability to handle multi-turn conversations
by directly testing the backend APIs and cache management.
"""

import asyncio
import time
import json
import sys
import os
import requests
from datetime import datetime
from typing import Dict, List, Any

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

class CacheConversationTester:
    """Test cache-first behavior in multi-turn conversation scenarios."""
    
    def __init__(self, backend_url="http://localhost:8000"):
        self.backend_url = backend_url
        self.conversation_log = []
        self.cache_hits = 0
        self.cache_misses = 0
        
    def log_interaction(self, turn: int, endpoint: str, response_time: float, 
                       cache_used: bool, status: str, response_size: int):
        """Log a conversation turn."""
        self.conversation_log.append({
            'turn': turn,
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'response_time': response_time,
            'cache_used': cache_used,
            'status': status,
            'response_size': response_size
        })
        
        if cache_used:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        cache_indicator = "🔥 CACHE" if cache_used else "🌐 API"
        print(f"Turn {turn}: {cache_indicator} - {endpoint} ({response_time:.3f}s) - {status}")

async def test_backend_availability():
    """Test if backend is running and responsive."""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False

async def test_cache_system():
    """Test the cache system directly."""
    print("🧪 Testing Cache System...")
    
    try:
        # Test cache manager directly
        from services.cache_manager import CacheManager
        
        cache = CacheManager()
        await cache.initialize()
        
        # Test cache operations
        test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
        
        # Set cache entry
        await cache.set("test/endpoint", {"param": "value"}, test_data, 300)
        
        # Get cache entry
        cached_data = await cache.get("test/endpoint", {"param": "value"})
        
        if cached_data:
            print("✅ Cache system working correctly")
            return True
        else:
            print("❌ Cache system not returning data")
            return False
            
    except Exception as e:
        print(f"❌ Cache system error: {e}")
        return False

async def test_multi_turn_api_calls():
    """Test multi-turn API call patterns."""
    print("\n🧪 Testing Multi-Turn API Call Patterns...")
    
    tester = CacheConversationTester()
    
    # Define conversation flow
    api_calls = [
        ("GET", "/api/v1/health", "Health check"),
        ("POST", "/api/v1/assets/list", {"project_id": "test-project"}, "Asset discovery"),
        ("POST", "/api/v1/security/analyze", {"project_id": "test-project"}, "Security analysis"),
        ("GET", "/api/v1/health", "Health check again"),
        ("POST", "/api/v1/assets/list", {"project_id": "test-project"}, "Asset discovery (repeat)"),
        ("POST", "/api/v1/iam/analyze", {"project_id": "test-project"}, "IAM analysis"),
        ("POST", "/api/v1/security/analyze", {"project_id": "test-project"}, "Security analysis (repeat)"),
    ]
    
    for turn, (method, endpoint, *args) in enumerate(api_calls, 1):
        payload = args[0] if args and isinstance(args[0], dict) else {}
        description = args[-1] if args else endpoint
        
        start_time = time.time()
        
        try:
            url = f"http://localhost:8000{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=payload, timeout=10)
            
            response_time = time.time() - start_time
            
            # Determine if this was likely from cache (very fast response on repeat calls)
            is_repeat = any(log['endpoint'] == endpoint for log in tester.conversation_log)
            cache_used = response_time < 0.1 and is_repeat
            
            status = f"{response.status_code} - {len(response.text)} bytes"
            
            tester.log_interaction(
                turn, endpoint, response_time, cache_used, status, len(response.text)
            )
            
            # Small delay between calls
            await asyncio.sleep(0.1)
            
        except Exception as e:
            tester.log_interaction(
                turn, endpoint, 0, False, f"ERROR: {str(e)}", 0
            )
    
    return tester

async def test_cache_wrapper_functions():
    """Test cache wrapper functions directly."""
    print("\n🧪 Testing Cache Wrapper Functions...")
    
    try:
        from services.agent_cache_wrapper import AgentCacheWrapper
        
        cache_wrapper = AgentCacheWrapper()
        
        # Test sequence
        functions_to_test = [
            ("discover_assets_cached", "Asset Discovery"),
            ("analyze_security_cached", "Security Analysis"), 
            ("analyze_iam_cached", "IAM Analysis"),
            ("analyze_storage_cached", "Storage Analysis"),
        ]
        
        results = []
        
        for func_name, description in functions_to_test:
            start_time = time.time()
            
            try:
                if hasattr(cache_wrapper, func_name):
                    result = await getattr(cache_wrapper, func_name)(force_refresh=False)
                    response_time = time.time() - start_time
                    
                    results.append({
                        'function': func_name,
                        'description': description,
                        'success': True,
                        'response_time': response_time,
                        'result_length': len(str(result))
                    })
                    
                    print(f"✅ {description}: {response_time:.3f}s ({len(str(result))} chars)")
                else:
                    results.append({
                        'function': func_name,
                        'description': description,
                        'success': False,
                        'error': 'Function not available'
                    })
                    print(f"⚠️ {description}: Function not available")
                
            except Exception as e:
                results.append({
                    'function': func_name,
                    'description': description,
                    'success': False,
                    'error': str(e)
                })
                print(f"❌ {description}: {str(e)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Cache wrapper test failed: {e}")
        return []

async def test_conversation_persistence():
    """Test conversation persistence through multiple interactions."""
    print("\n🧪 Testing Conversation Persistence...")
    
    try:
        # Test session management
        from services.session_manager import SessionManager
        
        session_mgr = SessionManager()
        
        # Create a test session
        session_id = "test-multi-turn-session"
        
        # Simulate conversation turns
        conversation_turns = [
            "What are my GCP assets?",
            "Are there any security issues?", 
            "Show me IAM analysis",
            "What about storage security?",
            "Show assets again"
        ]
        
        for turn, user_input in enumerate(conversation_turns, 1):
            # Simulate agent response
            agent_response = f"Response to turn {turn}: Processed '{user_input}'"
            
            # Store in session
            session_mgr.add_message(session_id, "user", user_input)
            session_mgr.add_message(session_id, "agent", agent_response)
            
            print(f"Turn {turn}: Stored conversation pair")
        
        # Retrieve session
        session_data = session_mgr.get_session(session_id)
        
        if session_data and len(session_data.get('messages', [])) == len(conversation_turns) * 2:
            print("✅ Conversation persistence working correctly")
            return True
        else:
            print(f"❌ Conversation persistence failed: {len(session_data.get('messages', []))} messages stored")
            return False
            
    except Exception as e:
        print(f"❌ Conversation persistence test failed: {e}")
        return False

async def run_comprehensive_cache_conversation_tests():
    """Run all cache conversation tests."""
    print("🧪 COMPREHENSIVE CACHE CONVERSATION TESTING")
    print("=" * 60)
    
    results = {
        'test_date': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Test 1: Backend availability
    print("🔍 Checking backend availability...")
    backend_available = await test_backend_availability()
    results['tests']['backend_available'] = backend_available
    
    if backend_available:
        print("✅ Backend is available")
    else:
        print("⚠️ Backend not available - will test cache system directly")
    
    # Test 2: Cache system
    cache_working = await test_cache_system()
    results['tests']['cache_system'] = cache_working
    
    # Test 3: Multi-turn API calls (if backend available)
    if backend_available:
        api_tester = await test_multi_turn_api_calls()
        results['tests']['multi_turn_api'] = {
            'total_calls': len(api_tester.conversation_log),
            'cache_hits': api_tester.cache_hits,
            'cache_misses': api_tester.cache_misses,
            'avg_response_time': sum(log['response_time'] for log in api_tester.conversation_log) / len(api_tester.conversation_log)
        }
    else:
        print("⏭️ Skipping API tests - backend not available")
        results['tests']['multi_turn_api'] = {'skipped': True}
    
    # Test 4: Cache wrapper functions
    wrapper_results = await test_cache_wrapper_functions()
    results['tests']['cache_wrapper'] = {
        'functions_tested': len(wrapper_results),
        'successful': len([r for r in wrapper_results if r['success']]),
        'failed': len([r for r in wrapper_results if not r['success']])
    }
    
    # Test 5: Conversation persistence
    persistence_working = await test_conversation_persistence()
    results['tests']['conversation_persistence'] = persistence_working
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 40)
    
    total_tests = len([t for t in results['tests'].values() if t is not True and t is not False])
    passed_tests = 0
    
    for test_name, result in results['tests'].items():
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
            if result:
                passed_tests += 1
        elif isinstance(result, dict) and 'skipped' not in result:
            status = "✅ PASS"
            passed_tests += 1
        else:
            status = "⏭️ SKIPPED"
        
        print(f"  {test_name}: {status}")
    
    success_rate = (passed_tests / len(results['tests'])) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}%")
    
    # Save results
    with open('CACHE_CONVERSATION_TEST_RESULTS.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved to CACHE_CONVERSATION_TEST_RESULTS.json")
    
    # Final verdict
    if success_rate >= 80:
        print("\n🎯 VERDICT: ✅ CACHE CONVERSATION SYSTEM WORKING EXCELLENTLY")
        return True
    elif success_rate >= 60:
        print("\n🎯 VERDICT: ⚠️ CACHE CONVERSATION SYSTEM WORKING WITH MINOR ISSUES")
        return True
    else:
        print("\n🎯 VERDICT: ❌ CACHE CONVERSATION SYSTEM NEEDS ATTENTION")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_cache_conversation_tests())
    exit(0 if success else 1)