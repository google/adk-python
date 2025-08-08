#!/usr/bin/env python3
"""
Example: Simple Agent Evaluation

Demonstrates basic usage of the ADK evaluation framework for testing agents.
This follows the exact patterns from Google ADK documentation.
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from adk_evaluator import ADKEvaluator, EvaluationCriteria


async def test_with_single_test_file():
    """
    Example from ADK docs: Evaluate with a single test file.
    
    This pattern is used for simple, unit-test style evaluations.
    """
    print("=" * 50)
    print("Testing with Single Test File (ADK Pattern)")
    print("=" * 50)
    
    # Create evaluator with default criteria
    evaluator = ADKEvaluator()
    
    # Evaluate agent against a single test file
    results = await evaluator.evaluate(
        agent_module="security_agent",  # Your agent module
        eval_dataset_file_path_or_dir="datasets/vulnerability_assessment.test.json"
    )
    
    # Check results
    for result in results:
        print(f"\nTest: {result.eval_id}")
        print(f"Passed: {result.passed}")
        print(f"Scores: {result.scores}")


async def test_with_custom_criteria():
    """
    Example: Evaluate with custom passing criteria.
    
    Allows adjusting thresholds for different use cases.
    """
    print("\n" + "=" * 50)
    print("Testing with Custom Criteria")
    print("=" * 50)
    
    # Define custom criteria (more lenient for development)
    criteria = EvaluationCriteria(
        tool_trajectory_avg_score=0.9,  # 90% tool accuracy required
        response_match_score=0.7,        # 70% response similarity required
        response_evaluation_score=0.6    # 60% evaluation score required
    )
    
    evaluator = ADKEvaluator(criteria)
    
    results = await evaluator.evaluate(
        agent_module="security_agent",
        eval_dataset_file_path_or_dir="datasets/compliance_check.test.json"
    )
    
    for result in results:
        print(f"\nTest: {result.eval_id}")
        print(f"Passed: {result.passed}")
        for metric, score in result.scores.items():
            threshold = getattr(criteria, metric, 0.0)
            status = "✅" if score >= threshold else "❌"
            print(f"  {metric}: {score:.2f} (threshold: {threshold}) {status}")


async def test_with_directory():
    """
    Example: Evaluate against all test files in a directory.
    
    Useful for comprehensive testing of an agent.
    """
    print("\n" + "=" * 50)
    print("Testing with Directory of Test Files")
    print("=" * 50)
    
    evaluator = ADKEvaluator()
    
    # Evaluate against all test files in datasets directory
    results = await evaluator.evaluate(
        agent_module="security_agent",
        eval_dataset_file_path_or_dir="datasets/",
        num_runs=2  # Run each test twice for consistency
    )
    
    # Summarize results
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    
    print(f"\nSummary:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Pass Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Group by test file
    test_groups = {}
    for result in results:
        test_name = result.details.get('test_name', result.eval_id)
        if test_name not in test_groups:
            test_groups[test_name] = []
        test_groups[test_name].append(result)
    
    print("\nPer-Test Results:")
    for test_name, test_results in test_groups.items():
        passed = sum(1 for r in test_results if r.passed)
        total = len(test_results)
        print(f"  {test_name}: {passed}/{total} passed")


async def test_evalset_format():
    """
    Example: Test with evalset format (multiple conversations).
    
    Evalsets are used for more complex, multi-turn conversation testing.
    """
    print("\n" + "=" * 50)
    print("Testing with Evalset Format")
    print("=" * 50)
    
    evaluator = ADKEvaluator()
    
    # Evalsets contain multiple test cases
    results = await evaluator.evaluate(
        agent_module="security_agent",
        eval_dataset_file_path_or_dir="datasets/incident_response.test.json"
    )
    
    for result in results:
        print(f"\nEvalset: {result.details.get('evalset_name', 'Unknown')}")
        print(f"Cases: {result.details.get('num_cases', 0)}")
        print(f"Passed: {result.passed}")
        
        # Show detailed scores
        print("Metrics:")
        for metric, score in result.scores.items():
            print(f"  {metric}: {score:.3f}")
        
        # Show any errors
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")


async def main():
    """Run all example tests"""
    print("ADK EVALUATION FRAMEWORK - EXAMPLES")
    print("Following Google ADK Patterns")
    print("=" * 50)
    
    try:
        # Run each example
        await test_with_single_test_file()
        await test_with_custom_criteria()
        await test_with_directory()
        await test_evalset_format()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)