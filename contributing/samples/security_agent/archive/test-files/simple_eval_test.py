#!/usr/bin/env python3
"""
Simple evaluation test for the security agent

This demonstrates how to manually test the agent's responses against expected behaviors.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add agents directory to path
sys.path.insert(0, str(Path(__file__).parent / "agents" / "gcp_security"))

from vertex_sqlite_agent import root_agent
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk import Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEvaluator:
    """Simple evaluator for testing agent responses"""
    
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            app_name="security_eval",
            agent=root_agent,
            session_service=self.session_service
        )
        
    async def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        
        test_id = test_case.get("test_id", "unknown")
        query = test_case.get("query", "")
        expected_tools = test_case.get("expected_tools", [])
        expected_keywords = test_case.get("expected_keywords", [])
        
        logger.info(f"Running test case: {test_id}")
        logger.info(f"Query: {query}")
        
        try:
            # Create session
            session_id = f"test_{test_id}_{int(time.time())}"
            session = self.session_service.create_session_sync(
                app_name="security_eval",
                user_id="evaluator",
                session_id=session_id,
                state={}
            )
            
            # Create message
            message = types.Content(
                role="user",
                parts=[types.Part(text=query)]
            )
            
            # Run agent
            response_parts = []
            tools_used = []
            
            for event in self.runner.run(
                user_id="evaluator", 
                session_id=session_id,
                new_message=message
            ):
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_parts.append(part.text)
                            elif hasattr(part, 'function_call') and part.function_call:
                                tools_used.append(part.function_call.name)
            
            full_response = " ".join(response_parts)
            
            # Evaluate response
            results = {
                "test_id": test_id,
                "query": query,
                "response": full_response,
                "tools_used": tools_used,
                "expected_tools": expected_tools,
                "expected_keywords": expected_keywords,
                "passed": False,
                "score": 0.0,
                "issues": []
            }
            
            # Check tools
            tools_score = 0.0
            if expected_tools:
                found_tools = set(tools_used)
                expected_tools_set = set(expected_tools)
                if expected_tools_set.issubset(found_tools):
                    tools_score = 1.0
                else:
                    missing_tools = expected_tools_set - found_tools
                    results["issues"].append(f"Missing expected tools: {list(missing_tools)}")
                    tools_score = len(expected_tools_set & found_tools) / len(expected_tools_set)
            
            # Check keywords
            keywords_score = 0.0
            if expected_keywords:
                response_lower = full_response.lower()
                found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
                keywords_score = len(found_keywords) / len(expected_keywords)
                if keywords_score < 1.0:
                    missing_keywords = set(expected_keywords) - set(found_keywords)
                    results["issues"].append(f"Missing expected keywords: {list(missing_keywords)}")
            
            # Calculate overall score
            overall_score = (tools_score + keywords_score) / 2 if expected_tools and expected_keywords else tools_score or keywords_score
            results["score"] = overall_score
            results["passed"] = overall_score >= 0.8
            
            logger.info(f"Test {test_id}: {'PASSED' if results['passed'] else 'FAILED'} (Score: {overall_score:.2f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in test case {test_id}: {e}")
            return {
                "test_id": test_id,
                "query": query,
                "error": str(e),
                "passed": False,
                "score": 0.0
            }

async def main():
    """Run evaluation tests"""
    
    # Define test cases
    test_cases = [
        {
            "test_id": "security_summary",
            "query": "Give me a security summary of the most critical issues",
            "expected_tools": ["query_security_data"],
            "expected_keywords": ["critical", "security", "firewall", "bucket", "recommendations"]
        },
        {
            "test_id": "iam_analysis", 
            "query": "Analyze IAM permissions for security issues",
            "expected_tools": ["query_security_data"],
            "expected_keywords": ["IAM", "permissions", "roles", "high-risk", "privilege"]
        },
        {
            "test_id": "storage_security",
            "query": "Check storage bucket security",
            "expected_tools": ["query_security_data"],
            "expected_keywords": ["storage", "bucket", "public", "versioning", "access"]
        },
        {
            "test_id": "firewall_rules",
            "query": "Review firewall rules for security problems",
            "expected_tools": ["query_security_data"],
            "expected_keywords": ["firewall", "rules", "0.0.0.0/0", "permissive", "restrict"]
        }
    ]
    
    logger.info(f"Starting evaluation with {len(test_cases)} test cases...")
    
    evaluator = SimpleEvaluator()
    results = []
    
    for test_case in test_cases:
        result = await evaluator.run_test_case(test_case)
        results.append(result)
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Calculate overall results
    passed_tests = [r for r in results if r.get("passed", False)]
    total_score = sum(r.get("score", 0) for r in results) / len(results)
    pass_rate = len(passed_tests) / len(results)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(results) - len(passed_tests)}")
    print(f"Pass Rate: {pass_rate:.1%}")
    print(f"Average Score: {total_score:.2f}")
    print(f"Overall Status: {'PASSED' if pass_rate >= 0.8 else 'FAILED'}")
    
    # Print detailed results
    print("\nDETAILED RESULTS:")
    print("-"*60)
    for result in results:
        status = "✅ PASS" if result.get("passed", False) else "❌ FAIL"
        score = result.get("score", 0)
        print(f"{status} {result.get('test_id', 'unknown'):20} Score: {score:.2f}")
        
        if result.get("issues"):
            for issue in result["issues"]:
                print(f"      Issue: {issue}")
        
        if result.get("error"):
            print(f"      Error: {result['error']}")
    
    # Save detailed results
    results_file = "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "total_tests": len(results),
                "passed": len(passed_tests),
                "failed": len(results) - len(passed_tests),
                "pass_rate": pass_rate,
                "average_score": total_score,
                "status": "PASSED" if pass_rate >= 0.8 else "FAILED"
            },
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if pass_rate >= 0.8 else 1)

if __name__ == "__main__":
    asyncio.run(main())