#!/usr/bin/env python3
"""
Simple Test Script to Validate Evaluation Framework
===================================================

This script demonstrates that our evaluation framework is working by testing:
1. Agent import and functionality
2. SQLite tool operation
3. Basic evaluation metrics
"""

import sys
import asyncio
from pathlib import Path

# Add the agent directory to path
agent_dir = Path('../agents/gcp_security')
sys.path.insert(0, str(agent_dir))

def test_agent_import():
    """Test that we can import and use the agent"""
    try:
        from vertex_sqlite_agent import root_agent
        print("✅ Agent imported successfully")
        print(f"   Agent type: {type(root_agent)}")
        print(f"   Agent tools: {[tool.name for tool in root_agent.tools]}")
        return True
    except Exception as e:
        print(f"❌ Agent import failed: {e}")
        return False

def test_sqlite_tool():
    """Test that the SQLite tool is working"""
    try:
        from sqlite_tool import query_security_data
        
        # Test security summary
        result = query_security_data(query_type='security_summary')
        print("✅ SQLite tool working")
        print(f"   Sample result length: {len(result)} characters")
        print(f"   Contains 'SECURITY SUMMARY': {'SECURITY SUMMARY' in result}")
        
        # Test IAM analysis 
        iam_result = query_security_data(query_type='iam_analysis')
        print(f"   IAM analysis result length: {len(iam_result)} characters")
        print(f"   Contains 'IAM Analysis': {'IAM Analysis' in iam_result}")
        
        return True
    except Exception as e:
        print(f"❌ SQLite tool test failed: {e}")
        return False

def test_evaluation_dataset():
    """Test that our evaluation dataset is valid JSON"""
    try:
        import json
        dataset_path = Path('datasets/custom_roles_analyzer.evalset.json')
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        print("✅ Evaluation dataset is valid JSON")
        print(f"   Dataset ID: {data.get('eval_set_id')}")
        print(f"   Number of test cases: {len(data.get('eval_cases', []))}")
        
        # Check first test case
        if data.get('eval_cases'):
            first_case = data['eval_cases'][0]
            print(f"   First test case ID: {first_case.get('eval_id')}")
            print(f"   Has expected response: {'expected_final_response' in first_case.get('conversation', [{}])[0]}")
            print(f"   Has tool calls: {'expected_tool_calls' in first_case.get('conversation', [{}])[0]}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset validation failed: {e}")
        return False

async def test_adk_evaluator():
    """Test the ADK evaluator directly"""
    try:
        from adk_evaluator import ADKEvaluator, EvaluationCriteria
        
        criteria = EvaluationCriteria()
        evaluator = ADKEvaluator(criteria)
        
        print("✅ ADK Evaluator created successfully")
        print(f"   Evaluator type: {type(evaluator)}")
        
        # Try a simple evaluation (this might fail but we'll catch the error)
        try:
            results = await evaluator.evaluate(
                agent_module='vertex_sqlite_agent',
                eval_dataset_file_path_or_dir='datasets/custom_roles_analyzer.evalset.json',
                num_runs=1
            )
            
            print(f"✅ Evaluation completed successfully")
            print(f"   Number of results: {len(results)}")
            
            for result in results:
                print(f"   - Test {result.eval_id}: {'PASS' if result.passed else 'FAIL'}")
                if not result.passed and result.errors:
                    print(f"     Error: {result.errors[0] if result.errors else 'No error details'}")
            
            return len(results) > 0
            
        except Exception as eval_error:
            print(f"⚠️  Evaluation execution failed (expected): {eval_error}")
            print("   This is likely due to agent module structure issues")
            return False
            
    except Exception as e:
        print(f"❌ ADK Evaluator setup failed: {e}")
        return False

def main():
    """Run all tests and provide summary"""
    print("🧪 Testing ADK Security Agent Evaluation Framework")
    print("=" * 55)
    
    tests = [
        ("Agent Import", test_agent_import),
        ("SQLite Tool", test_sqlite_tool),
        ("Evaluation Dataset", test_evaluation_dataset),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        results[test_name] = test_func()
    
    # Async test
    print(f"\n📋 Testing ADK Evaluator...")
    results["ADK Evaluator"] = asyncio.run(test_adk_evaluator())
    
    # Summary
    print("\n" + "=" * 55)
    print("🎯 TEST SUMMARY")
    print("=" * 55)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Evaluation framework is ready!")
    elif passed >= total * 0.75:
        print("👍 Most tests passed. Framework is mostly functional.")
    else:
        print("⚠️  Several tests failed. Framework needs fixes.")
    
    return passed == total

if __name__ == "__main__":
    main()