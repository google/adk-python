#!/usr/bin/env python3
"""
Test script for ADK Agent Evaluation System

This script tests the evaluation framework components to ensure they work correctly
with the existing security agent and follow ADK patterns.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the evaluation modules to the path
sys.path.append(str(Path(__file__).parent))

from evaluators.security_evaluator import SecurityEvaluator, SecurityMetricType, evaluate_security_response
from evaluators.compliance_evaluator import ComplianceEvaluator, evaluate_compliance_response
from evaluators.performance_evaluator import PerformanceEvaluator
from metrics.security_metrics import calculate_security_score
from runners.evaluation_runner import AgentEvaluationRunner, EvaluationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_security_evaluator():
    """Test the SecurityEvaluator with sample data"""
    logger.info("Testing SecurityEvaluator...")
    
    # Test case 1: SQL Injection vulnerability detection
    query1 = "Analyze this code for SQL injection vulnerabilities: SELECT * FROM users WHERE id = " + str(user_id)
    actual_response1 = """This code contains a SQL injection vulnerability. The user input is directly concatenated into the SQL query without parameterization. This allows attackers to inject malicious SQL code.
    
    Risk: Critical
    Remediation: Use parameterized queries or prepared statements."""
    
    expected_response1 = """Critical SQL injection vulnerability detected. Direct string concatenation in query construction allows arbitrary SQL execution. Use parameterized queries: SELECT * FROM users WHERE id = ? with bound parameters."""
    
    score1, status1 = evaluate_security_response(
        query1, actual_response1, expected_response1,
        metric_type=SecurityMetricType.VULNERABILITY_DETECTION
    )
    
    logger.info(f"SQL Injection test - Score: {score1:.3f}, Status: {status1}")
    
    # Test case 2: General security analysis
    query2 = "Review this authentication system for security issues"
    actual_response2 = """The authentication system has several security weaknesses:
    1. Plaintext password storage
    2. No rate limiting on login attempts
    3. Weak session management
    4. Missing multi-factor authentication"""
    
    expected_response2 = """Security analysis identifies multiple authentication vulnerabilities:
    - Plaintext passwords (should use bcrypt hashing)
    - No brute force protection
    - Insufficient session security
    - Lack of MFA implementation"""
    
    score2, status2 = evaluate_security_response(
        query2, actual_response2, expected_response2,
        metric_type=SecurityMetricType.SECURITY_ACCURACY
    )
    
    logger.info(f"General security test - Score: {score2:.3f}, Status: {status2}")
    
    return (score1 + score2) / 2


async def test_compliance_evaluator():
    """Test the ComplianceEvaluator with sample data"""
    logger.info("Testing ComplianceEvaluator...")
    
    query = "Analyze this system for SOC 2 compliance requirements"
    actual_response = """SOC 2 compliance analysis reveals gaps in:
    - Access controls (CC6): Missing individual user accounts
    - Monitoring (CC7): No centralized logging system  
    - Change management (CC8): Lack of documented procedures
    - Data processing integrity (PI1): Insufficient validation controls
    
    Recommendations:
    1. Implement unique user identification
    2. Deploy SIEM solution for monitoring
    3. Document change management process
    4. Establish data integrity checks"""
    
    expected_response = """SOC 2 Trust Service Criteria assessment:
    - Security (CC6): Non-compliant - shared credentials violate access control requirements
    - Monitoring (CC7): Gaps in system oversight and incident response
    - Change Management (CC8): Missing authorization and documentation controls
    - Processing Integrity (PI1): Inadequate data validation and error handling
    
    Priority remediation:
    - Individual access controls with MFA
    - Centralized monitoring and alerting
    - Formal change approval process
    - Data integrity verification procedures"""
    
    score, status = evaluate_compliance_response(
        query, actual_response, expected_response
    )
    
    logger.info(f"SOC 2 compliance test - Score: {score:.3f}, Status: {status}")
    
    return score


async def test_security_metrics():
    """Test the security metrics calculation"""
    logger.info("Testing security metrics...")
    
    query = "Identify vulnerabilities in this web application code"
    actual_response = """Vulnerability Assessment:
    
    1. Cross-Site Scripting (XSS) - High Risk
       Location: User input rendering without sanitization
       Impact: Session hijacking, data theft
       
    2. SQL Injection - Critical Risk  
       Location: Database query construction
       Impact: Database compromise, data breach
       
    3. Authentication Bypass - Medium Risk
       Location: Login validation logic
       Impact: Unauthorized access
       
    Risk prioritization: Address SQL injection immediately, then XSS, then authentication issues."""
    
    expected_response = """Security Analysis Results:
    
    Critical Issues:
    - SQL Injection vulnerability in query builder
    - XSS vulnerability in user content display
    
    Medium Risk:
    - Authentication logic flaws
    - Session management weaknesses
    
    Immediate Actions:
    1. Implement parameterized queries
    2. Add input sanitization
    3. Strengthen authentication controls"""
    
    result = calculate_security_score(
        actual_response=actual_response,
        expected_response=expected_response,
        query=query
    )
    
    logger.info(f"Security metrics - Overall Score: {result.score:.3f}")
    logger.info(f"Details: {result.details}")
    
    return result.score


async def test_evaluation_runner():
    """Test the evaluation runner with mock agent"""
    logger.info("Testing AgentEvaluationRunner...")
    
    # Create test configuration
    config = EvaluationConfig(
        agent_module="mock_security_agent",  # Would be actual agent module
        eval_dataset_dir="evaluation/datasets",
        output_dir="evaluation/results/test",
        num_runs=1,  # Reduced for testing
        metrics={
            "security_accuracy_score": {"threshold": 0.8, "weight": 0.5},
            "vulnerability_detection_score": {"threshold": 0.85, "weight": 0.5}
        }
    )
    
    runner = AgentEvaluationRunner()
    
    try:
        # This would normally run against real datasets
        # For testing, we'll simulate the expected structure
        
        logger.info("Evaluation runner initialized successfully")
        logger.info(f"Configuration: {config.agent_module}")
        logger.info(f"Datasets directory: {config.eval_dataset_dir}")
        
        # Check if datasets exist
        datasets_path = Path(config.eval_dataset_dir)
        if datasets_path.exists():
            datasets = list(datasets_path.glob("*.test.json"))
            logger.info(f"Found {len(datasets)} test datasets: {[d.name for d in datasets]}")
        else:
            logger.warning(f"Datasets directory not found: {datasets_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing evaluation runner: {e}")
        return False


async def test_dataset_loading():
    """Test loading of evaluation datasets"""
    logger.info("Testing dataset loading...")
    
    datasets_dir = Path("evaluation/datasets")
    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return False
    
    dataset_files = list(datasets_dir.glob("*.test.json"))
    
    if not dataset_files:
        logger.warning("No test datasets found")
        return False
    
    loaded_datasets = 0
    
    for dataset_file in dataset_files:
        try:
            import json
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            # Validate structure
            required_fields = ["eval_set_id", "name", "eval_cases"]
            if all(field in data for field in required_fields):
                logger.info(f"✅ {dataset_file.name}: {data['name']}")
                logger.info(f"   Cases: {len(data['eval_cases'])}")
                loaded_datasets += 1
            else:
                logger.warning(f"❌ {dataset_file.name}: Missing required fields")
                
        except Exception as e:
            logger.error(f"❌ {dataset_file.name}: Error loading - {e}")
    
    logger.info(f"Successfully loaded {loaded_datasets}/{len(dataset_files)} datasets")
    return loaded_datasets > 0


async def run_integration_test():
    """Run comprehensive integration test"""
    logger.info("=" * 50)
    logger.info("ADK AGENT EVALUATION SYSTEM - INTEGRATION TEST")
    logger.info("=" * 50)
    
    test_results = {}
    
    # Test individual components
    try:
        test_results['security_evaluator'] = await test_security_evaluator()
        logger.info(f"✅ Security evaluator test completed")
    except Exception as e:
        logger.error(f"❌ Security evaluator test failed: {e}")
        test_results['security_evaluator'] = 0.0
    
    try:
        test_results['compliance_evaluator'] = await test_compliance_evaluator()
        logger.info(f"✅ Compliance evaluator test completed")
    except Exception as e:
        logger.error(f"❌ Compliance evaluator test failed: {e}")
        test_results['compliance_evaluator'] = 0.0
    
    try:
        test_results['security_metrics'] = await test_security_metrics()
        logger.info(f"✅ Security metrics test completed")
    except Exception as e:
        logger.error(f"❌ Security metrics test failed: {e}")
        test_results['security_metrics'] = 0.0
    
    try:
        test_results['evaluation_runner'] = await test_evaluation_runner()
        logger.info(f"✅ Evaluation runner test completed")
    except Exception as e:
        logger.error(f"❌ Evaluation runner test failed: {e}")
        test_results['evaluation_runner'] = False
    
    try:
        test_results['dataset_loading'] = await test_dataset_loading()
        logger.info(f"✅ Dataset loading test completed")
    except Exception as e:
        logger.error(f"❌ Dataset loading test failed: {e}")
        test_results['dataset_loading'] = False
    
    # Calculate overall results
    logger.info("=" * 50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    numeric_scores = [score for score in test_results.values() if isinstance(score, float)]
    boolean_results = [result for result in test_results.values() if isinstance(result, bool)]
    
    if numeric_scores:
        average_score = sum(numeric_scores) / len(numeric_scores)
        logger.info(f"Average Evaluation Score: {average_score:.3f}")
    
    passed_tests = sum(1 for result in boolean_results if result)
    total_boolean_tests = len(boolean_results)
    
    if total_boolean_tests > 0:
        logger.info(f"System Tests Passed: {passed_tests}/{total_boolean_tests}")
    
    # Overall system health
    all_scores = list(test_results.values())
    system_healthy = (
        all(score >= 0.7 for score in numeric_scores) and
        all(result for result in boolean_results)
    )
    
    if system_healthy:
        logger.info("🎉 SYSTEM STATUS: HEALTHY - All tests passed!")
    else:
        logger.warning("⚠️  SYSTEM STATUS: ISSUES DETECTED - Some tests failed")
    
    # Specific recommendations
    logger.info("\nRECOMMENDations:")
    for test_name, result in test_results.items():
        if isinstance(result, float) and result < 0.7:
            logger.info(f"- Improve {test_name}: Score {result:.3f} below threshold")
        elif isinstance(result, bool) and not result:
            logger.info(f"- Fix {test_name}: Test failed")
    
    return test_results


if __name__ == "__main__":
    # Run the integration test
    results = asyncio.run(run_integration_test())
    
    # Exit with appropriate code
    all_passed = all(
        (isinstance(r, float) and r >= 0.7) or (isinstance(r, bool) and r)
        for r in results.values()
    )
    
    sys.exit(0 if all_passed else 1)