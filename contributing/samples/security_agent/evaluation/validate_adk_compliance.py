#!/usr/bin/env python3
"""
ADK Compliance Validator

Validates that the evaluation framework follows Google ADK patterns correctly.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent))

from adk_evaluator import ADKEvaluator, EvaluationCriteria, EvaluationMetric


class ADKComplianceValidator:
    """Validates ADK compliance of the evaluation framework"""
    
    def __init__(self):
        self.validation_results = []
        self.passed_checks = 0
        self.total_checks = 0
    
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print("=" * 60)
        print("ADK COMPLIANCE VALIDATION")
        print("=" * 60)
        
        # Run validation checks
        self.check_file_formats()
        self.check_metrics_implementation()
        self.check_evaluator_interface()
        self.check_test_patterns()
        self.check_documentation()
        
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Passed: {self.passed_checks}/{self.total_checks} checks")
        
        if self.passed_checks == self.total_checks:
            print("✅ FULLY ADK COMPLIANT")
            return True
        else:
            print("❌ NOT FULLY COMPLIANT")
            print("\nFailed checks:")
            for result in self.validation_results:
                if not result['passed']:
                    print(f"  - {result['check']}: {result['reason']}")
            return False
    
    def add_check(self, check_name: str, passed: bool, reason: str = ""):
        """Record a validation check result"""
        self.total_checks += 1
        if passed:
            self.passed_checks += 1
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}: {reason}")
        
        self.validation_results.append({
            'check': check_name,
            'passed': passed,
            'reason': reason
        })
    
    def check_file_formats(self):
        """Validate test file formats comply with ADK standards"""
        print("\n📁 Checking File Formats...")
        
        datasets_dir = Path("datasets")
        
        # Check for test.json files
        test_files = list(datasets_dir.glob("*.test.json"))
        self.add_check(
            "Test files exist (.test.json)",
            len(test_files) > 0,
            f"Found {len(test_files)} test files"
        )
        
        # Validate test file structure
        if test_files:
            valid_structure = self._validate_test_file_structure(test_files[0])
            self.add_check(
                "Test file structure valid",
                valid_structure,
                "Must have user_content, final_response fields"
            )
        
        # Check for evalset files
        evalset_files = list(datasets_dir.glob("*.evalset.json"))
        has_evalsets = len(evalset_files) > 0 or any(
            self._is_evalset_format(f) for f in test_files
        )
        self.add_check(
            "Evalset support",
            has_evalsets,
            "Should support evalset format"
        )
    
    def _validate_test_file_structure(self, test_file: Path) -> bool:
        """Validate internal structure of test file"""
        try:
            with open(test_file, 'r') as f:
                data = json.load(f)
            
            # Check for evalset format
            if 'eval_cases' in data:
                # Evalset format
                return all(
                    'eval_id' in case and 'conversation' in case
                    for case in data.get('eval_cases', [])
                )
            else:
                # Single test format
                return 'user_content' in data or 'final_response' in data
        except:
            return False
    
    def _is_evalset_format(self, test_file: Path) -> bool:
        """Check if file is in evalset format"""
        try:
            with open(test_file, 'r') as f:
                data = json.load(f)
            return 'eval_cases' in data
        except:
            return False
    
    def check_metrics_implementation(self):
        """Validate that ADK standard metrics are implemented"""
        print("\n📊 Checking Metrics Implementation...")
        
        # Check metric enums
        metrics = [m.value for m in EvaluationMetric]
        required_metrics = [
            'tool_trajectory_avg_score',
            'response_match_score',
            'response_evaluation_score'
        ]
        
        for metric in required_metrics:
            self.add_check(
                f"Metric '{metric}' defined",
                metric in metrics
            )
        
        # Check default thresholds
        criteria = EvaluationCriteria()
        self.add_check(
            "Tool trajectory default threshold = 1.0",
            criteria.tool_trajectory_avg_score == 1.0
        )
        self.add_check(
            "Response match default threshold = 0.8",
            criteria.response_match_score == 0.8
        )
    
    def check_evaluator_interface(self):
        """Validate evaluator follows ADK interface patterns"""
        print("\n🔧 Checking Evaluator Interface...")
        
        evaluator = ADKEvaluator()
        
        # Check required methods
        self.add_check(
            "evaluate() method exists",
            hasattr(evaluator, 'evaluate') and callable(evaluator.evaluate)
        )
        
        # Check method signature
        import inspect
        sig = inspect.signature(evaluator.evaluate)
        params = list(sig.parameters.keys())
        
        self.add_check(
            "evaluate() has agent_module parameter",
            'agent_module' in params
        )
        self.add_check(
            "evaluate() has eval_dataset_file_path_or_dir parameter",
            'eval_dataset_file_path_or_dir' in params
        )
        self.add_check(
            "evaluate() has num_runs parameter",
            'num_runs' in params
        )
        
        # Check async support
        self.add_check(
            "evaluate() is async",
            inspect.iscoroutinefunction(evaluator.evaluate)
        )
    
    def check_test_patterns(self):
        """Validate test patterns follow ADK conventions"""
        print("\n🧪 Checking Test Patterns...")
        
        # Check for example files
        examples_dir = Path("examples")
        
        self.add_check(
            "Examples directory exists",
            examples_dir.exists()
        )
        
        if examples_dir.exists():
            example_files = list(examples_dir.glob("*.py"))
            self.add_check(
                "Example scripts exist",
                len(example_files) > 0,
                f"Found {len(example_files)} examples"
            )
            
            # Check for specific patterns
            has_simple = any('simple' in f.name for f in example_files)
            has_pytest = any('pytest' in f.name for f in example_files)
            has_web = any('web' in f.name for f in example_files)
            
            self.add_check("Simple test example", has_simple)
            self.add_check("Pytest integration example", has_pytest)
            self.add_check("Web UI example", has_web)
    
    def check_documentation(self):
        """Check documentation completeness"""
        print("\n📚 Checking Documentation...")
        
        self.add_check(
            "README.md exists",
            Path("README.md").exists()
        )
        
        self.add_check(
            "ADK patterns guide exists",
            Path("ADK_PATTERNS.md").exists()
        )
        
        self.add_check(
            "Migration guide exists",
            Path("MIGRATION_GUIDE.md").exists()
        )
        
        # Check README content
        if Path("README.md").exists():
            with open("README.md", 'r') as f:
                content = f.read().lower()
            
            self.add_check(
                "README mentions ADK compliance",
                'adk' in content and 'compliant' in content
            )
            
            self.add_check(
                "README includes usage examples",
                'adkevaluator' in content
            )


async def run_functional_tests():
    """Run functional tests to verify the evaluator works"""
    print("\n" + "=" * 60)
    print("FUNCTIONAL TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Create evaluator with default criteria
        print("\n1️⃣ Testing evaluator creation...")
        evaluator = ADKEvaluator()
        print("   ✅ Evaluator created successfully")
        
        # Test 2: Create with custom criteria
        print("\n2️⃣ Testing custom criteria...")
        criteria = EvaluationCriteria(
            tool_trajectory_avg_score=0.9,
            response_match_score=0.85
        )
        custom_evaluator = ADKEvaluator(criteria)
        print("   ✅ Custom criteria accepted")
        
        # Test 3: Check file discovery
        print("\n3️⃣ Testing file discovery...")
        datasets_dir = Path("datasets")
        if datasets_dir.exists():
            test_files = list(datasets_dir.glob("*.test.json"))
            print(f"   ✅ Found {len(test_files)} test files")
        
        # Test 4: Validate score calculation
        print("\n4️⃣ Testing score calculations...")
        
        # Test tool trajectory score
        actual_tools = [{'tool_name': 'search'}, {'tool_name': 'analyze'}]
        expected_tools = [{'tool_name': 'search'}, {'tool_name': 'analyze'}]
        score = evaluator._calculate_tool_trajectory_score(actual_tools, expected_tools)
        assert score == 1.0, f"Perfect match should be 1.0, got {score}"
        print(f"   ✅ Tool trajectory score: {score}")
        
        # Test response match score
        actual_response = "The system has a SQL injection vulnerability"
        expected_response = "SQL injection vulnerability found in the system"
        score = evaluator._calculate_response_match_score(actual_response, expected_response)
        assert score > 0.5, f"Similar responses should score > 0.5, got {score}"
        print(f"   ✅ Response match score: {score:.2f}")
        
        print("\n✅ All functional tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functional test failed: {e}")
        return False


def main():
    """Main validation entry point"""
    print("🔍 ADK Compliance Validator v1.0")
    print("Validating evaluation framework against Google ADK standards")
    print()
    
    # Run compliance checks
    validator = ADKComplianceValidator()
    compliance_passed = validator.validate_all()
    
    # Run functional tests
    functional_passed = asyncio.run(run_functional_tests())
    
    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULT")
    print("=" * 60)
    
    if compliance_passed and functional_passed:
        print("🎉 VALIDATION SUCCESSFUL!")
        print("The evaluation framework is fully ADK-compliant.")
        return 0
    else:
        print("⚠️  VALIDATION FAILED")
        if not compliance_passed:
            print("- Compliance checks failed")
        if not functional_passed:
            print("- Functional tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)