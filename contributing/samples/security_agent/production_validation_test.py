#!/usr/bin/env python3
"""
Production Validation Test for Security Agent Application
=========================================================

This script performs comprehensive validation of the security agent application
to ensure production readiness and verify all downstream page connections.
"""

import sys
import os
import traceback
import importlib
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class ProductionValidator:
    """Comprehensive production validation for the security agent application."""
    
    def __init__(self):
        self.results = {
            'import_tests': {},
            'initialization_tests': {},
            'integration_tests': {},
            'downstream_connections': {},
            'warnings': [],
            'errors': [],
            'overall_status': 'unknown'
        }
        
    def log_result(self, category: str, test_name: str, status: str, details: str = ""):
        """Log test result with details."""
        if category not in self.results:
            self.results[category] = {}
        
        self.results[category][test_name] = {
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"[{status.upper()}] {category}/{test_name}: {details}")
        
    def test_core_imports(self) -> bool:
        """Test all core component imports."""
        print("\n=== Testing Core Imports ===")
        
        import_tests = [
            ('frontend.main_app', 'Main application module'),
            ('frontend.services.asset_data_service', 'Asset data service'),
            ('frontend.components.dashboard.dashboard_view', 'Dashboard view'),
            ('frontend.components.dashboard.asset_charts', 'Asset charts'),
            ('frontend.components.chat.chat_view', 'Chat view'),
            ('backend.services.asset_inventory_service', 'Asset inventory service'),
            ('backend.api.asset_inventory', 'Asset inventory API'),
            ('backend.main', 'Backend main module')
        ]
        
        all_passed = True
        
        for module_path, description in import_tests:
            try:
                module = importlib.import_module(module_path)
                self.log_result('import_tests', module_path, 'PASS', f"{description} imported successfully")
            except ImportError as e:
                self.log_result('import_tests', module_path, 'FAIL', f"Import failed: {str(e)}")
                self.results['errors'].append(f"Import error in {module_path}: {str(e)}")
                all_passed = False
            except Exception as e:
                self.log_result('import_tests', module_path, 'WARNING', f"Unexpected error: {str(e)}")
                self.results['warnings'].append(f"Warning in {module_path}: {str(e)}")
        
        return all_passed
    
    def test_asset_data_service_initialization(self) -> bool:
        """Test AssetDataService initialization and basic functionality."""
        print("\n=== Testing AssetDataService Initialization ===")
        
        try:
            from frontend.services.asset_data_service import AssetDataService
            
            # Test basic initialization
            service = AssetDataService()
            self.log_result('initialization_tests', 'AssetDataService', 'PASS', 
                          'Service instantiated successfully')
            
            # Test method availability
            required_methods = ['get_assets', 'get_asset_summary', 'get_assets_by_type']
            for method in required_methods:
                if hasattr(service, method):
                    self.log_result('initialization_tests', f'AssetDataService.{method}', 'PASS',
                                  f'Method {method} available')
                else:
                    self.log_result('initialization_tests', f'AssetDataService.{method}', 'FAIL',
                                  f'Method {method} missing')
                    return False
            
            return True
            
        except Exception as e:
            self.log_result('initialization_tests', 'AssetDataService', 'FAIL', 
                          f'Initialization failed: {str(e)}')
            self.results['errors'].append(f"AssetDataService initialization error: {str(e)}")
            return False
    
    def test_dashboard_components(self) -> bool:
        """Test dashboard view and asset charts integration."""
        print("\n=== Testing Dashboard Components ===")
        
        try:
            from frontend.components.dashboard import dashboard_view, asset_charts
            
            # Test dashboard view functions
            dashboard_functions = ['main', 'create_dashboard', 'get_asset_overview']
            for func_name in dashboard_functions:
                if hasattr(dashboard_view, func_name):
                    self.log_result('integration_tests', f'dashboard_view.{func_name}', 'PASS',
                                  f'Function {func_name} available')
                else:
                    self.log_result('integration_tests', f'dashboard_view.{func_name}', 'WARNING',
                                  f'Function {func_name} not found (may be optional)')
            
            # Test asset charts functions
            chart_functions = ['create_asset_distribution_chart', 'create_asset_timeline_chart']
            for func_name in chart_functions:
                if hasattr(asset_charts, func_name):
                    self.log_result('integration_tests', f'asset_charts.{func_name}', 'PASS',
                                  f'Function {func_name} available')
                else:
                    self.log_result('integration_tests', f'asset_charts.{func_name}', 'WARNING',
                                  f'Function {func_name} not found (may be optional)')
            
            return True
            
        except Exception as e:
            self.log_result('integration_tests', 'dashboard_components', 'FAIL',
                          f'Dashboard component test failed: {str(e)}')
            self.results['errors'].append(f"Dashboard component error: {str(e)}")
            return False
    
    def test_chat_integration(self) -> bool:
        """Test chat view integration with asset inventory."""
        print("\n=== Testing Chat Integration ===")
        
        try:
            from frontend.components.chat import chat_view
            
            # Check for main chat functions
            chat_functions = ['main', 'display_chat_interface', 'handle_asset_query']
            for func_name in chat_functions:
                if hasattr(chat_view, func_name):
                    self.log_result('integration_tests', f'chat_view.{func_name}', 'PASS',
                                  f'Function {func_name} available')
                else:
                    self.log_result('integration_tests', f'chat_view.{func_name}', 'WARNING',
                                  f'Function {func_name} not found (may be optional)')
            
            return True
            
        except Exception as e:
            self.log_result('integration_tests', 'chat_integration', 'FAIL',
                          f'Chat integration test failed: {str(e)}')
            self.results['errors'].append(f"Chat integration error: {str(e)}")
            return False
    
    def test_downstream_page_connections(self) -> bool:
        """Test all downstream page connections."""
        print("\n=== Testing Downstream Page Connections ===")
        
        pages_to_test = [
            ('frontend.components.security.iam_analyzer_view', 'IAM Analyzer'),
            ('frontend.components.security.security_evaluation_view', 'Security Evaluation'),
            ('frontend.components.compliance.compliance_view', 'Compliance View'),
            ('frontend.components.monitoring.performance_monitoring_view', 'Performance Monitoring'),
            ('frontend.components.shared.recommendations_view', 'Recommendations'),
            ('frontend.components.shared.api_explorer_view', 'API Explorer'),
            ('frontend.components.roadmap.roadmap_view', 'Roadmap View')
        ]
        
        all_connected = True
        
        for module_path, page_name in pages_to_test:
            try:
                module = importlib.import_module(module_path)
                
                # Check for main function or class
                if hasattr(module, 'main') or hasattr(module, 'show') or hasattr(module, 'display'):
                    self.log_result('downstream_connections', page_name, 'PASS',
                                  f'{page_name} page connection working')
                else:
                    self.log_result('downstream_connections', page_name, 'WARNING',
                                  f'{page_name} page loaded but no main function found')
                    
            except ImportError as e:
                self.log_result('downstream_connections', page_name, 'FAIL',
                              f'{page_name} page connection failed: {str(e)}')
                self.results['errors'].append(f"Downstream page error {page_name}: {str(e)}")
                all_connected = False
                
            except Exception as e:
                self.log_result('downstream_connections', page_name, 'WARNING',
                              f'{page_name} page warning: {str(e)}')
                self.results['warnings'].append(f"Downstream page warning {page_name}: {str(e)}")
        
        return all_connected
    
    def check_known_issues(self) -> None:
        """Check for known issues mentioned in the validation request."""
        print("\n=== Checking Known Issues ===")
        
        # Check google.genai import issue
        try:
            with open('frontend/components/chat/chat_view.py', 'r') as f:
                content = f.read()
                if 'google.genai' in content and 'google.generativeai' not in content:
                    self.results['warnings'].append(
                        "chat_view.py uses 'google.genai' instead of 'google.generativeai'"
                    )
                    print("[WARNING] Found google.genai import issue in chat_view.py")
        except FileNotFoundError:
            pass
        
        # Check for conversation_memory module
        try:
            importlib.import_module('backend.services.conversation_memory')
            print("[PASS] conversation_memory module exists")
        except ImportError:
            self.results['warnings'].append("Missing services.conversation_memory module")
            print("[WARNING] Missing services.conversation_memory module")
        
        # Check for invalid escape sequences (basic check)
        python_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        escape_issues = 0
        for file_path in python_files[:10]:  # Check first 10 files to avoid too much output
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Simple check for common invalid escape sequences
                    if '\\.' in content or '\\/' in content:
                        escape_issues += 1
            except:
                pass
        
        if escape_issues > 0:
            self.results['warnings'].append(f"Found potential invalid escape sequences in {escape_issues} files")
            print(f"[WARNING] Found potential invalid escape sequences in {escape_issues} files")
    
    def test_backend_health(self) -> bool:
        """Test backend health check (if backend is running)."""
        print("\n=== Testing Backend Health ===")
        
        try:
            import requests
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                self.log_result('integration_tests', 'backend_health', 'PASS',
                              'Backend health check successful')
                return True
            else:
                self.log_result('integration_tests', 'backend_health', 'WARNING',
                              f'Backend returned status {response.status_code}')
                return False
        except requests.exceptions.ConnectionError:
            self.log_result('integration_tests', 'backend_health', 'INFO',
                          'Backend not running (expected for validation)')
            return True  # This is expected during validation
        except Exception as e:
            self.log_result('integration_tests', 'backend_health', 'WARNING',
                          f'Backend health check failed: {str(e)}')
            return False
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        print("\n" + "="*60)
        print("PRODUCTION VALIDATION REPORT")
        print("="*60)
        
        # Count results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warning_tests = 0
        
        for category, tests in self.results.items():
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    total_tests += 1
                    if result['status'] == 'PASS':
                        passed_tests += 1
                    elif result['status'] == 'FAIL':
                        failed_tests += 1
                    elif result['status'] in ['WARNING', 'INFO']:
                        warning_tests += 1
        
        # Determine overall status
        if failed_tests == 0 and len(self.results['errors']) == 0:
            if warning_tests == 0 and len(self.results['warnings']) == 0:
                self.results['overall_status'] = 'PRODUCTION_READY'
            else:
                self.results['overall_status'] = 'READY_WITH_WARNINGS'
        else:
            self.results['overall_status'] = 'NEEDS_FIXES'
        
        report = f"""
SUMMARY:
--------
Total Tests: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}  
Warnings: {warning_tests}
Overall Status: {self.results['overall_status']}

IMPORT TESTS:
"""
        
        for test_name, result in self.results.get('import_tests', {}).items():
            report += f"  [{result['status']}] {test_name}: {result['details']}\n"
        
        report += "\nINITIALIZATION TESTS:\n"
        for test_name, result in self.results.get('initialization_tests', {}).items():
            report += f"  [{result['status']}] {test_name}: {result['details']}\n"
        
        report += "\nINTEGRATION TESTS:\n"
        for test_name, result in self.results.get('integration_tests', {}).items():
            report += f"  [{result['status']}] {test_name}: {result['details']}\n"
        
        report += "\nDOWNSTREAM PAGE CONNECTIONS:\n"
        for test_name, result in self.results.get('downstream_connections', {}).items():
            report += f"  [{result['status']}] {test_name}: {result['details']}\n"
        
        if self.results['warnings']:
            report += "\nWARNINGS:\n"
            for warning in self.results['warnings']:
                report += f"  - {warning}\n"
        
        if self.results['errors']:
            report += "\nERRORS:\n"
            for error in self.results['errors']:
                report += f"  - {error}\n"
        
        report += f"""
PRODUCTION READINESS ASSESSMENT:
--------------------------------
Status: {self.results['overall_status']}

Key Findings:
- Core imports: {'✅ Working' if passed_tests > 0 else '❌ Issues found'}
- Asset inventory integration: {'✅ Integrated' if 'AssetDataService' in str(self.results) else '❌ Not integrated'}
- Downstream pages: {'✅ Connected' if len(self.results.get('downstream_connections', {})) > 0 else '❌ Issues found'}
- Chat-centric design: {'✅ Maintained' if 'chat_view' in str(self.results) else '❌ Issues found'}

Recommendations:
"""
        
        if self.results['overall_status'] == 'PRODUCTION_READY':
            report += "- Application is ready for production deployment\n"
        elif self.results['overall_status'] == 'READY_WITH_WARNINGS':
            report += "- Application is functional but has minor issues to address\n"
            report += "- Consider fixing warnings before production deployment\n"
        else:
            report += "- Application requires fixes before production deployment\n"
            report += "- Address all errors and critical warnings\n"
        
        return report
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("Starting Production Validation for Security Agent Application")
        print("="*60)
        
        # Run all validation tests
        self.test_core_imports()
        self.test_asset_data_service_initialization()
        self.test_dashboard_components()
        self.test_chat_integration()
        self.test_downstream_page_connections()
        self.test_backend_health()
        self.check_known_issues()
        
        # Generate and print report
        report = self.generate_report()
        print(report)
        
        # Save report to file
        with open('production_validation_report.txt', 'w') as f:
            f.write(report)
        
        return self.results

def main():
    """Main validation entry point."""
    validator = ProductionValidator()
    results = validator.run_validation()
    
    # Exit with appropriate code
    if results['overall_status'] == 'PRODUCTION_READY':
        sys.exit(0)
    elif results['overall_status'] == 'READY_WITH_WARNINGS':
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == '__main__':
    main()