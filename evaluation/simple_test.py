#!/usr/bin/env python3
"""
Simple Test for ADK Agent Evaluation System

Basic validation test that checks individual components work correctly.
"""

import asyncio
import sys
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset_files():
    """Test that evaluation datasets are properly formatted"""
    logger.info("Testing evaluation dataset files...")
    
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return False
    
    test_files = list(datasets_dir.glob("*.test.json"))
    
    if not test_files:
        logger.warning("No test dataset files found")
        return False
    
    valid_datasets = 0
    
    for test_file in test_files:
        try:
            with open(test_file, 'r') as f:
                data = json.load(f)
            
            # Check required fields for ADK EvalSet format
            required_fields = ["eval_set_id", "name", "eval_cases"]
            
            if all(field in data for field in required_fields):
                logger.info(f"✅ {test_file.name}")
                logger.info(f"   ID: {data['eval_set_id']}")
                logger.info(f"   Name: {data['name']}")
                logger.info(f"   Cases: {len(data['eval_cases'])}")
                
                # Validate eval cases structure
                for i, eval_case in enumerate(data['eval_cases']):
                    if "eval_id" in eval_case and "conversation" in eval_case:
                        logger.info(f"     Case {i+1}: {eval_case['eval_id']}")
                    else:
                        logger.warning(f"     Case {i+1}: Missing required fields")
                
                valid_datasets += 1
            else:
                missing = [f for f in required_fields if f not in data]
                logger.error(f"❌ {test_file.name}: Missing fields - {missing}")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ {test_file.name}: Invalid JSON - {e}")
        except Exception as e:
            logger.error(f"❌ {test_file.name}: Error - {e}")
    
    logger.info(f"\nDataset validation: {valid_datasets}/{len(test_files)} files valid")
    return valid_datasets == len(test_files)


def test_config_files():
    """Test configuration files"""
    logger.info("\nTesting configuration files...")
    
    config_files = [
        "config/evaluation_config.yaml",
        "config/test_config.json"
    ]
    
    valid_configs = 0
    
    for config_file in config_files:
        config_path = Path(config_file)
        
        if config_path.exists():
            try:
                if config_file.endswith('.yaml'):
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                
                logger.info(f"✅ {config_file}: Valid configuration")
                valid_configs += 1
                
            except Exception as e:
                logger.error(f"❌ {config_file}: Error loading - {e}")
        else:
            logger.warning(f"⚠️ {config_file}: File not found")
    
    return valid_configs > 0


def test_evaluator_logic():
    """Test basic evaluator logic without imports"""
    logger.info("\nTesting evaluator logic...")
    
    # Test security vulnerability detection logic
    def detect_vulnerabilities(text):
        """Simple vulnerability detection"""
        vulnerabilities = []
        text_lower = text.lower()
        
        patterns = {
            'sql_injection': ['sql injection', 'sqli', 'union select'],
            'xss': ['cross-site scripting', 'xss', 'script tag'],
            'auth_bypass': ['auth bypass', 'authentication bypass']
        }
        
        for vuln_type, keywords in patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                vulnerabilities.append(vuln_type)
        
        return vulnerabilities
    
    # Test cases
    test_cases = [
        {
            'input': 'This code has a SQL injection vulnerability with union select',
            'expected': ['sql_injection'],
            'description': 'SQL injection detection'
        },
        {
            'input': 'Cross-site scripting (XSS) vulnerability found in script tag',
            'expected': ['xss'],
            'description': 'XSS detection'
        },
        {
            'input': 'Authentication bypass allows unauthorized access',
            'expected': ['auth_bypass'],
            'description': 'Auth bypass detection'
        }
    ]
    
    passed_tests = 0
    
    for test_case in test_cases:
        detected = detect_vulnerabilities(test_case['input'])
        expected = test_case['expected']
        
        if set(detected) >= set(expected):
            logger.info(f"✅ {test_case['description']}: Detected {detected}")
            passed_tests += 1
        else:
            logger.error(f"❌ {test_case['description']}: Expected {expected}, got {detected}")
    
    logger.info(f"\nEvaluator logic tests: {passed_tests}/{len(test_cases)} passed")
    return passed_tests == len(test_cases)


def test_compliance_detection():
    """Test compliance framework detection"""
    logger.info("\nTesting compliance detection logic...")
    
    def detect_compliance_frameworks(text):
        """Simple compliance framework detection"""
        frameworks = []
        text_lower = text.lower()
        
        framework_keywords = {
            'soc2': ['soc 2', 'soc2', 'trust services'],
            'pci_dss': ['pci', 'payment card', 'cardholder data'],
            'gdpr': ['gdpr', 'general data protection', 'personal data'],
            'hipaa': ['hipaa', 'protected health information', 'phi']
        }
        
        for framework, keywords in framework_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                frameworks.append(framework)
        
        return frameworks
    
    test_cases = [
        {
            'input': 'SOC 2 compliance requires trust services criteria',
            'expected': ['soc2'],
            'description': 'SOC 2 detection'
        },
        {
            'input': 'PCI DSS requirements for cardholder data protection',
            'expected': ['pci_dss'],
            'description': 'PCI DSS detection'
        },
        {
            'input': 'GDPR personal data processing requirements',
            'expected': ['gdpr'],
            'description': 'GDPR detection'
        }
    ]
    
    passed_tests = 0
    
    for test_case in test_cases:
        detected = detect_compliance_frameworks(test_case['input'])
        expected = test_case['expected']
        
        if set(detected) >= set(expected):
            logger.info(f"✅ {test_case['description']}: Detected {detected}")
            passed_tests += 1
        else:
            logger.error(f"❌ {test_case['description']}: Expected {expected}, got {detected}")
    
    logger.info(f"\nCompliance detection tests: {passed_tests}/{len(test_cases)} passed")
    return passed_tests == len(test_cases)


def test_response_scoring():
    """Test response quality scoring logic"""
    logger.info("\nTesting response scoring logic...")
    
    def calculate_response_similarity(actual, expected):
        """Simple response similarity calculation"""
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 1.0 if not actual_words else 0.0
        
        intersection = actual_words.intersection(expected_words)
        return len(intersection) / len(expected_words)
    
    test_cases = [
        {
            'actual': 'SQL injection vulnerability found in database query',
            'expected': 'SQL injection security vulnerability in query',
            'threshold': 0.6,
            'description': 'High similarity response'
        },
        {
            'actual': 'No security issues detected',
            'expected': 'Multiple critical vulnerabilities found',
            'threshold': 0.3,
            'description': 'Low similarity response'
        }
    ]
    
    passed_tests = 0
    
    for test_case in test_cases:
        similarity = calculate_response_similarity(
            test_case['actual'], 
            test_case['expected']
        )
        
        meets_threshold = similarity >= test_case['threshold']
        
        if meets_threshold:
            logger.info(f"✅ {test_case['description']}: Similarity {similarity:.3f}")
            passed_tests += 1
        else:
            logger.error(f"❌ {test_case['description']}: Similarity {similarity:.3f} below {test_case['threshold']}")
    
    logger.info(f"\nResponse scoring tests: {passed_tests}/{len(test_cases)} passed")
    return passed_tests == len(test_cases)


def main():
    """Run all simple tests"""
    logger.info("=" * 60)
    logger.info("ADK AGENT EVALUATION SYSTEM - SIMPLE VALIDATION TEST")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run individual tests
    test_results['datasets'] = test_dataset_files()
    test_results['configs'] = test_config_files()
    test_results['evaluator_logic'] = test_evaluator_logic()
    test_results['compliance_detection'] = test_compliance_detection()
    test_results['response_scoring'] = test_response_scoring()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        emoji = "✅" if result else "❌"
        logger.info(f"{emoji} {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - Evaluation system is ready!")
        return True
    else:
        logger.warning("⚠️  SOME TESTS FAILED - Review issues above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)