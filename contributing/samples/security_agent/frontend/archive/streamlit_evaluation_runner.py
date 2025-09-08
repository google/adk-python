"""
Streamlit-friendly Evaluation Runner
===================================

A wrapper around the evaluation system that works well with Streamlit's execution model.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import streamlit as st

# Setup logging
logger = logging.getLogger(__name__)

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "agents" / "gcp_security"))

class StreamlitEvaluationRunner:
    """Streamlit-compatible evaluation runner"""
    
    def __init__(self):
        self.results_file = project_root / "evaluation_results.json"
        
    def run_evaluation_sync(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run evaluation synchronously for Streamlit"""
        
        try:
            # Import the evaluation components with proper path handling
            import os
            import sys
            agent_dir = project_root / "agents" / "gcp_security"
            
            # Save current directory and change to agent directory for imports
            original_cwd = os.getcwd()
            try:
                os.chdir(agent_dir)
                sys.path.insert(0, str(agent_dir))
                from vertex_sqlite_agent import root_agent
            finally:
                os.chdir(original_cwd)
            
            from google.adk import Runner
            from google.adk.sessions import InMemorySessionService
            from google.genai import types
            
            # Create session service and runner with timeout
            session_service = InMemorySessionService()
            runner = Runner(
                app_name="streamlit_eval",
                agent=root_agent,
                session_service=session_service
            )
            
            results = []
            
            for i, test_case in enumerate(test_cases):
                try:
                    # Create session
                    session_id = f"streamlit_test_{test_case['test_id']}_{int(time.time())}"
                    session = session_service.create_session_sync(
                        app_name="streamlit_eval",
                        user_id="streamlit_user",
                        session_id=session_id,
                        state={}
                    )
                    
                    # Create message
                    message = types.Content(
                        role="user",
                        parts=[types.Part(text=test_case["query"])]
                    )
                    
                    # Run agent and collect response with timeout handling
                    response_parts = []
                    tools_used = []
                    event_count = 0
                    max_events = 100  # Prevent infinite loops
                    start_time = time.time()
                    timeout_seconds = 30  # 30 second timeout per test
                    
                    try:
                        for event in runner.run(
                            user_id="streamlit_user",
                            session_id=session_id,
                            new_message=message
                        ):
                            # Check for timeout
                            if time.time() - start_time > timeout_seconds:
                                logger.warning(f"Test {test_case['test_id']} timed out after {timeout_seconds} seconds")
                                response_parts.append("[TIMEOUT - Test execution exceeded time limit]")
                                break
                            
                            # Prevent infinite loops
                            event_count += 1
                            if event_count > max_events:
                                logger.warning(f"Test {test_case['test_id']} exceeded max event count")
                                response_parts.append("[MAX_EVENTS - Too many events generated]")
                                break
                            
                            if hasattr(event, 'content') and event.content:
                                if hasattr(event.content, 'parts'):
                                    for part in event.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            response_parts.append(part.text)
                                        elif hasattr(part, 'function_call') and part.function_call:
                                            tools_used.append(part.function_call.name)
                    except Exception as event_error:
                        logger.error(f"Error processing events for test {test_case['test_id']}: {event_error}")
                        response_parts.append(f"[ERROR - {str(event_error)}]")
                    
                    full_response = " ".join(response_parts)
                    
                    # Evaluate response
                    result = self._evaluate_response(test_case, full_response, tools_used)
                    results.append(result)
                    
                    # Show progress in Streamlit
                    progress = (i + 1) / len(test_cases)
                    if 'progress_bar' in st.session_state:
                        st.session_state.progress_bar.progress(progress)
                    if 'status_text' in st.session_state:
                        st.session_state.status_text.text(f"Completed test: {test_case['test_id']}")
                    
                except Exception as e:
                    logger.error(f"Error in test case {test_case['test_id']}: {e}")
                    results.append({
                        "test_id": test_case["test_id"],
                        "query": test_case["query"],
                        "error": str(e),
                        "passed": False,
                        "score": 0.0
                    })
            
            # Calculate summary
            passed_tests = [r for r in results if r.get("passed", False)]
            total_score = sum(r.get("score", 0) for r in results)
            avg_score = total_score / len(results) if results else 0
            
            evaluation_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tests": len(results),
                    "passed": len(passed_tests),
                    "failed": len(results) - len(passed_tests),
                    "pass_rate": len(passed_tests) / len(results) if results else 0,
                    "average_score": avg_score,
                    "status": "PASSED" if avg_score >= 0.8 else "FAILED"
                },
                "results": results
            }
            
            # Save results
            with open(self.results_file, 'w') as f:
                json.dump(evaluation_data, f, indent=2)
            
            return {
                "success": True,
                "data": evaluation_data
            }
            
        except Exception as e:
            logger.exception("Error running evaluation")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _evaluate_response(self, test_case: Dict[str, Any], response: str, tools_used: List[str]) -> Dict[str, Any]:
        """Evaluate a single response"""
        
        expected_tools = test_case.get("expected_tools", [])
        expected_keywords = test_case.get("expected_keywords", [])
        
        result = {
            "test_id": test_case["test_id"],
            "query": test_case["query"],
            "response": response,
            "tools_used": tools_used,
            "expected_tools": expected_tools,
            "expected_keywords": expected_keywords,
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
                result["issues"].append(f"Missing expected tools: {list(missing_tools)}")
                tools_score = len(expected_tools_set & found_tools) / len(expected_tools_set)
        
        # Check keywords
        keywords_score = 0.0
        if expected_keywords:
            response_lower = response.lower()
            found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
            keywords_score = len(found_keywords) / len(expected_keywords)
            if keywords_score < 1.0:
                missing_keywords = set(expected_keywords) - set(found_keywords)
                result["issues"].append(f"Missing expected keywords: {list(missing_keywords)}")
        
        # Calculate overall score
        if expected_tools and expected_keywords:
            overall_score = (tools_score + keywords_score) / 2
        else:
            overall_score = tools_score or keywords_score
        
        result["score"] = overall_score
        result["passed"] = overall_score >= 0.8
        
        return result
    
    def get_predefined_test_cases(self) -> List[Dict[str, Any]]:
        """Get predefined test cases for quick evaluation"""
        
        return [
            {
                "test_id": "security_summary",
                "query": "Give me a comprehensive security summary of the most critical issues",
                "expected_tools": ["query_security_data"],
                "expected_keywords": ["critical", "security", "firewall", "bucket", "issues"]
            },
            {
                "test_id": "iam_analysis",
                "query": "Analyze IAM permissions for potential security risks",
                "expected_tools": ["query_security_data"],
                "expected_keywords": ["IAM", "permissions", "roles", "risk", "privilege"]
            },
            {
                "test_id": "storage_security",
                "query": "Check storage bucket security and public access",
                "expected_tools": ["query_security_data"],
                "expected_keywords": ["storage", "bucket", "public", "access", "security"]
            },
            {
                "test_id": "firewall_analysis",
                "query": "Review firewall rules for security vulnerabilities",
                "expected_tools": ["query_security_data"],
                "expected_keywords": ["firewall", "rules", "security", "port", "access"]
            },
            {
                "test_id": "vulnerability_scan",
                "query": "Show me critical security findings and vulnerabilities",
                "expected_tools": ["query_security_data"],
                "expected_keywords": ["critical", "vulnerability", "security", "findings", "fix"]
            }
        ]
    
    def load_results(self) -> Optional[Dict[str, Any]]:
        """Load existing evaluation results"""
        
        if not self.results_file.exists():
            return None
        
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None

# Global instance
streamlit_evaluator = StreamlitEvaluationRunner()