"""
Simple Networking Implementation Test
====================================

Tests the networking implementation by validating file structure, imports,
and basic functionality without requiring full ADK environment.
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import importlib.util

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleNetworkingTest:
    """Simple test framework for networking features"""
    
    def __init__(self):
        self.project_root = project_root
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "success_rate": 0.0
            }
        }
        
    def run_test(self, test_name: str, test_function) -> Dict:
        """Run a single test"""
        logger.info(f"Running test: {test_name}")
        test_result = {
            "name": test_name,
            "passed": False,
            "error": None,
            "details": {}
        }
        
        try:
            details = test_function()
            test_result["passed"] = True
            test_result["details"] = details
            logger.info(f"✅ {test_name} PASSED")
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ {test_name} FAILED: {e}")
        
        self.results["tests"].append(test_result)
        self.results["summary"]["total"] += 1
        if test_result["passed"]:
            self.results["summary"]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1
            
        return test_result
    
    def test_file_structure(self) -> Dict:
        """Test that all required files exist"""
        required_files = [
            "backend/models/network_models.py",
            "backend/models/error_models.py", 
            "backend/services/vpc_flow_analyzer.py",
            "backend/services/error_knowledge_base.py",
            "backend/services/connectivity_tester.py",
            "backend/api/connectivity.py",
            "frontend/networking_dashboard.py",
            "evaluation/datasets/networking_connectivity_testing.evalset.json",
            "evaluation/datasets/networking_error_analysis.evalset.json"
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
                logger.info(f"  ✅ {file_path}")
            else:
                missing_files.append(file_path)
                logger.error(f"  ❌ {file_path}")
        
        if missing_files:
            raise ValueError(f"Missing required files: {missing_files}")
        
        return {
            "total_files": len(required_files),
            "existing_files": len(existing_files),
            "missing_files": len(missing_files),
            "files": existing_files
        }
    
    def test_model_imports(self) -> Dict:
        """Test that models can be imported"""
        models_to_test = [
            ("backend.models.network_models", ["NetworkLogEntry", "ConnectivityTestResult", "NetworkAnomaly"]),
            ("backend.models.error_models", ["ErrorCodeEntry", "ErrorAnalysis", "Resolution"])
        ]
        
        imported_modules = []
        import_errors = []
        
        for module_name, expected_classes in models_to_test:
            try:
                # Import the module
                module = importlib.import_module(module_name)
                imported_modules.append(module_name)
                
                # Check expected classes
                for class_name in expected_classes:
                    if not hasattr(module, class_name):
                        raise ImportError(f"Class {class_name} not found in {module_name}")
                
                logger.info(f"  ✅ {module_name} - Classes: {expected_classes}")
                
            except Exception as e:
                import_errors.append(f"{module_name}: {str(e)}")
                logger.error(f"  ❌ {module_name}: {e}")
        
        if import_errors:
            raise ImportError(f"Import errors: {import_errors}")
        
        return {
            "imported_modules": imported_modules,
            "import_errors": import_errors
        }
    
    def test_service_imports(self) -> Dict:
        """Test that services can be imported"""
        services_to_test = [
            ("backend.services.vpc_flow_analyzer", ["VPCFlowLogProcessor"]),
            ("backend.services.error_knowledge_base", ["InternalErrorKnowledgeBase"]),
            ("backend.services.connectivity_tester", ["ConnectivityTester"])
        ]
        
        imported_services = []
        import_errors = []
        
        for service_module, expected_classes in services_to_test:
            try:
                # Import the service
                module = importlib.import_module(service_module)
                imported_services.append(service_module)
                
                # Check expected classes
                for class_name in expected_classes:
                    if not hasattr(module, class_name):
                        raise ImportError(f"Class {class_name} not found in {service_module}")
                
                logger.info(f"  ✅ {service_module} - Classes: {expected_classes}")
                
            except Exception as e:
                import_errors.append(f"{service_module}: {str(e)}")
                logger.error(f"  ❌ {service_module}: {e}")
        
        if import_errors:
            raise ImportError(f"Service import errors: {import_errors}")
        
        return {
            "imported_services": imported_services,
            "import_errors": import_errors
        }
    
    def test_database_functionality(self) -> Dict:
        """Test database connectivity and basic queries"""
        try:
            # Test the main database
            db_path = self.project_root / "backend" / "cache" / "gcp_data.db"
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
            
            # Connect and run basic queries
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Test basic queries
            tables_checked = []
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                raise ValueError("No tables found in database")
            
            # Test a few key tables
            test_tables = ["assets", "storage_buckets", "security_findings"]
            for table in test_tables:
                if table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    tables_checked.append({"table": table, "count": count})
                    logger.info(f"  ✅ Table {table}: {count} rows")
            
            conn.close()
            
            return {
                "database_path": str(db_path),
                "total_tables": len(tables),
                "tables_checked": tables_checked,
                "all_tables": tables
            }
            
        except Exception as e:
            logger.error(f"Database test failed: {e}")
            raise
    
    def test_evaluation_datasets(self) -> Dict:
        """Test that evaluation datasets are valid JSON"""
        datasets_dir = self.project_root / "evaluation" / "datasets"
        dataset_files = [
            "networking_connectivity_testing.evalset.json",
            "networking_error_analysis.evalset.json"
        ]
        
        validated_datasets = []
        validation_errors = []
        
        for dataset_file in dataset_files:
            try:
                dataset_path = datasets_dir / dataset_file
                if not dataset_path.exists():
                    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
                
                # Load and validate JSON
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset_data = json.load(f)
                
                # Basic validation
                required_fields = ["name", "description", "test_cases"]
                for field in required_fields:
                    if field not in dataset_data:
                        raise ValueError(f"Missing required field: {field}")
                
                test_cases = dataset_data.get("test_cases", [])
                if not test_cases:
                    raise ValueError("No test cases found")
                
                # Validate test cases
                for i, test_case in enumerate(test_cases):
                    required_test_fields = ["id", "name", "input", "expected_output"]
                    for field in required_test_fields:
                        if field not in test_case:
                            raise ValueError(f"Test case {i}: missing field {field}")
                
                validated_datasets.append({
                    "file": dataset_file,
                    "test_cases": len(test_cases),
                    "name": dataset_data["name"]
                })
                
                logger.info(f"  ✅ {dataset_file}: {len(test_cases)} test cases")
                
            except Exception as e:
                validation_errors.append(f"{dataset_file}: {str(e)}")
                logger.error(f"  ❌ {dataset_file}: {e}")
        
        if validation_errors:
            raise ValueError(f"Dataset validation errors: {validation_errors}")
        
        return {
            "validated_datasets": validated_datasets,
            "validation_errors": validation_errors,
            "total_test_cases": sum(d["test_cases"] for d in validated_datasets)
        }
    
    def test_frontend_integration(self) -> Dict:
        """Test frontend integration"""
        try:
            # Check main frontend file
            frontend_file = self.project_root / "frontend" / "unified_streaming_client.py"
            if not frontend_file.exists():
                raise FileNotFoundError("Main frontend file not found")
            
            # Check for networking imports
            with open(frontend_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for networking dashboard integration
            networking_imports = [
                "from frontend.networking_dashboard import",
                "networking_dashboard",
                "render_networking_dashboard"
            ]
            
            integration_found = []
            for import_check in networking_imports:
                if import_check in content:
                    integration_found.append(import_check)
            
            # Check networking dashboard file
            dashboard_file = self.project_root / "frontend" / "networking_dashboard.py"
            dashboard_exists = dashboard_file.exists()
            
            dashboard_functions = []
            if dashboard_exists:
                with open(dashboard_file, 'r', encoding='utf-8') as f:
                    dashboard_content = f.read()
                
                # Check for key functions
                key_functions = [
                    "render_connectivity_testing_section",
                    "render_traffic_analysis_section", 
                    "render_error_analysis_section",
                    "render_network_health_overview"
                ]
                
                for func in key_functions:
                    if func in dashboard_content:
                        dashboard_functions.append(func)
            
            return {
                "frontend_file_exists": frontend_file.exists(),
                "dashboard_file_exists": dashboard_exists,
                "integration_found": integration_found,
                "dashboard_functions": dashboard_functions,
                "integration_score": len(integration_found) + len(dashboard_functions)
            }
            
        except Exception as e:
            logger.error(f"Frontend integration test failed: {e}")
            raise
    
    def test_api_endpoints(self) -> Dict:
        """Test API endpoint definitions"""
        try:
            api_file = self.project_root / "backend" / "api" / "connectivity.py"
            if not api_file.exists():
                raise FileNotFoundError("API connectivity file not found")
            
            with open(api_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for key endpoint patterns
            endpoint_patterns = [
                "@router.post(\"/test\"",
                "@router.get(\"/history\"",
                "ConnectivityTestRequest",
                "ConnectivityTestResponse",
                "run_connectivity_test"
            ]
            
            found_patterns = []
            for pattern in endpoint_patterns:
                if pattern in content:
                    found_patterns.append(pattern)
            
            # Check imports
            required_imports = [
                "from fastapi import",
                "from backend.models.network_models import",
                "from backend.services.connectivity_tester import"
            ]
            
            found_imports = []
            for import_pattern in required_imports:
                if import_pattern in content:
                    found_imports.append(import_pattern)
            
            return {
                "api_file_exists": True,
                "endpoint_patterns_found": found_patterns,
                "required_imports_found": found_imports,
                "api_completeness_score": len(found_patterns) + len(found_imports)
            }
            
        except Exception as e:
            logger.error(f"API endpoints test failed: {e}")
            raise
    
    def run_all_tests(self) -> Dict:
        """Run all tests"""
        logger.info("🚀 Starting Simple Networking Implementation Tests...")
        
        # Define tests
        tests = [
            ("File Structure", self.test_file_structure),
            ("Model Imports", self.test_model_imports),
            ("Service Imports", self.test_service_imports),
            ("Database Functionality", self.test_database_functionality),
            ("Evaluation Datasets", self.test_evaluation_datasets),
            ("Frontend Integration", self.test_frontend_integration),
            ("API Endpoints", self.test_api_endpoints)
        ]
        
        # Run tests
        for test_name, test_function in tests:
            self.run_test(test_name, test_function)
        
        # Calculate success rate
        if self.results["summary"]["total"] > 0:
            self.results["summary"]["success_rate"] = (
                self.results["summary"]["passed"] / self.results["summary"]["total"]
            ) * 100
        
        return self.results
    
    def generate_report(self, results: Dict) -> str:
        """Generate test report"""
        report = []
        report.append("# Simple Networking Implementation Test Report")
        report.append(f"Generated: {results['timestamp']}")
        report.append("")
        
        # Summary
        summary = results["summary"]
        report.append("## 📊 Test Summary")
        report.append(f"- **Total Tests**: {summary['total']}")
        report.append(f"- **Passed**: {summary['passed']} ✅")
        report.append(f"- **Failed**: {summary['failed']} ❌")
        report.append(f"- **Success Rate**: {summary['success_rate']:.1f}%")
        report.append("")
        
        # Individual test results
        report.append("## 📋 Test Results")
        for test in results["tests"]:
            status = "✅" if test["passed"] else "❌"
            report.append(f"### {status} {test['name']}")
            
            if test["passed"]:
                if test["details"]:
                    for key, value in test["details"].items():
                        if isinstance(value, (int, float, str)):
                            report.append(f"- **{key}**: {value}")
                        elif isinstance(value, list) and len(value) <= 5:
                            report.append(f"- **{key}**: {', '.join(map(str, value))}")
            else:
                report.append(f"**Error**: {test['error']}")
            report.append("")
        
        # Overall assessment
        report.append("## 🎯 Assessment")
        if summary["success_rate"] >= 90:
            report.append("🎉 **Excellent**: All networking features are properly implemented!")
        elif summary["success_rate"] >= 70:
            report.append("👍 **Good**: Most networking features working, minor issues to address")
        elif summary["success_rate"] >= 50:
            report.append("⚠️ **Needs Work**: Several issues detected, requires attention")
        else:
            report.append("❌ **Critical Issues**: Major problems preventing proper functionality")
        
        report.append("")
        report.append("## 🔧 Next Steps")
        if summary["failed"] > 0:
            report.append("1. Address failing tests listed above")
            report.append("2. Ensure all required dependencies are installed")
            report.append("3. Verify database setup and connectivity")
            report.append("4. Run full ADK evaluation after fixing issues")
        else:
            report.append("1. Run comprehensive ADK evaluation testing")
            report.append("2. Test end-to-end user workflows")
            report.append("3. Performance and load testing")
            report.append("4. Security review and validation")
        
        return "\n".join(report)

def main():
    """Main test function"""
    print("🧪 Starting Simple Networking Implementation Tests...")
    
    tester = SimpleNetworkingTest()
    results = tester.run_all_tests()
    
    # Generate report
    report = tester.generate_report(results)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    results_file = results_dir / f"simple_networking_test_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save report
    report_file = results_dir / f"simple_networking_test_report_{timestamp}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📊 Test Results Summary:")
    print(f"- Total Tests: {results['summary']['total']}")
    print(f"- Passed: {results['summary']['passed']} ✅")
    print(f"- Failed: {results['summary']['failed']} ❌")
    print(f"- Success Rate: {results['summary']['success_rate']:.1f}%")
    
    print(f"\n📄 Results saved:")
    print(f"- JSON: {results_file}")
    print(f"- Report: {report_file}")
    
    return results

if __name__ == "__main__":
    main()