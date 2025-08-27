"""
Networking Evaluation Framework
===============================

Comprehensive evaluation framework for the Networking Troubleshooting Ninja
features, including connectivity testing, error analysis, and VPC flow log analysis.
"""

import os
import sys
import json
import asyncio
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib.util

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkingEvaluationRunner:
    """Enhanced evaluation runner for networking features"""
    
    def __init__(self):
        self.project_root = project_root
        self.agent_dir = project_root / "agents" / "gcp_security"
        self.datasets_dir = current_dir / "datasets"
        self.results = {}
        
        # Evaluation datasets for networking
        self.networking_datasets = [
            "networking_connectivity_testing.evalset.json",
            "networking_error_analysis.evalset.json"
        ]
        
        # Initialize test database
        self.setup_test_database()
        
    def setup_test_database(self):
        """Setup test database with networking data"""
        db_path = project_root / "backend" / "cache" / "test_networking.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Create networking tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS connectivity_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    destination_ip TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    is_successful BOOLEAN NOT NULL,
                    latency_ms REAL,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS network_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    error_code TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    source_service TEXT,
                    severity TEXT,
                    resolution_attempted TEXT,
                    resolution_successful BOOLEAN,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vpc_flow_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    destination_ip TEXT NOT NULL,
                    source_port INTEGER,
                    destination_port INTEGER,
                    protocol TEXT,
                    action TEXT,
                    bytes_transferred INTEGER,
                    packets_transferred INTEGER,
                    metadata TEXT
                )
            """)
            
            # Insert test data
            self.insert_test_data(cursor)
            
            conn.commit()
            conn.close()
            logger.info(f"Test database setup complete: {db_path}")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def insert_test_data(self, cursor):
        """Insert test data for evaluation"""
        # Connectivity test data
        connectivity_data = [
            ('2025-01-27 10:00:00', '127.0.0.1', '8.8.8.8', 'PING', True, 2.1, None, '{"test_id": "ping_001"}'),
            ('2025-01-27 10:05:00', '127.0.0.1', '1.1.1.1', 'PING', True, 1.8, None, '{"test_id": "ping_002"}'),
            ('2025-01-27 10:10:00', '127.0.0.1', '192.168.1.100', 'TCP_CONNECT', False, None, 'Connection refused', '{"test_id": "tcp_001"}'),
            ('2025-01-27 10:15:00', '127.0.0.1', '9.9.9.9', 'TRACEROUTE', True, 15.3, None, '{"test_id": "trace_001"}'),
            ('2025-01-27 10:20:00', '127.0.0.1', '10.0.1.1', 'PING', False, None, 'Network unreachable', '{"test_id": "ping_003"}')
        ]
        
        cursor.executemany("""
            INSERT INTO connectivity_tests 
            (timestamp, source_ip, destination_ip, test_type, is_successful, latency_ms, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, connectivity_data)
        
        # Network error data
        error_data = [
            ('2025-01-27 09:30:00', 'NETWORK_UNREACHABLE', 'Network is unreachable', 'VPC', 'HIGH', 'Check routing tables', False, '{"source": "gcp_compute"}'),
            ('2025-01-27 09:45:00', 'CONNECTION_REFUSED', 'Connection refused by target', 'Compute Engine', 'MEDIUM', 'Check firewall rules', True, '{"source": "gcp_compute"}'),
            ('2025-01-27 10:00:00', 'DNS_RESOLUTION_FAILED', 'DNS resolution failed', 'DNS', 'HIGH', 'Check DNS configuration', False, '{"source": "gcp_dns"}'),
            ('2025-01-27 10:15:00', 'TIMEOUT', 'Connection timeout after 30s', 'Load Balancer', 'MEDIUM', 'Increase timeout', True, '{"source": "gcp_lb"}'),
            ('2025-01-27 10:30:00', 'FIREWALL_RULE_DENIED', 'Traffic blocked by firewall', 'Firewall', 'HIGH', 'Update firewall rules', True, '{"source": "gcp_firewall"}')
        ]
        
        cursor.executemany("""
            INSERT INTO network_errors
            (timestamp, error_code, error_message, source_service, severity, resolution_attempted, resolution_successful, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, error_data)
        
        # VPC Flow Log data
        flow_data = [
            ('2025-01-27 10:00:00', '10.0.1.15', '10.0.2.23', 33445, 443, 'TCP', 'ACCEPT', 1024, 10, '{"vpc": "production"}'),
            ('2025-01-27 10:01:00', '10.0.1.23', '8.8.8.8', 54321, 53, 'UDP', 'ACCEPT', 512, 4, '{"vpc": "production"}'),
            ('2025-01-27 10:02:00', '192.168.1.100', '10.0.1.15', 12345, 22, 'TCP', 'REJECT', 0, 0, '{"vpc": "production"}'),
            ('2025-01-27 10:03:00', '10.0.1.45', '203.0.113.1', 44567, 80, 'TCP', 'ACCEPT', 2048, 15, '{"vpc": "production"}'),
            ('2025-01-27 10:04:00', '172.16.0.10', '10.0.1.15', 55678, 3389, 'TCP', 'REJECT', 0, 0, '{"vpc": "development"}')
        ]
        
        cursor.executemany("""
            INSERT INTO vpc_flow_logs
            (timestamp, source_ip, destination_ip, source_port, destination_port, protocol, action, bytes_transferred, packets_transferred, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, flow_data)
    
    async def load_agent(self):
        """Load the vertex_sqlite agent"""
        try:
            # Change to agent directory for imports
            original_cwd = os.getcwd()
            os.chdir(self.agent_dir)
            
            # Import the agent
            spec = importlib.util.spec_from_file_location(
                "vertex_sqlite_agent", 
                self.agent_dir / "vertex_sqlite_agent.py"
            )
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)
            
            # Get the root agent
            root_agent = agent_module.root_agent
            os.chdir(original_cwd)
            
            logger.info("Agent loaded successfully")
            return root_agent
            
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            return None
    
    def load_evaluation_dataset(self, dataset_path: Path) -> Dict:
        """Load an evaluation dataset"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            return {"test_cases": []}
    
    async def run_test_case(self, agent, test_case: Dict) -> Dict:
        """Run a single test case"""
        try:
            # Try both import patterns for compatibility
            try:
                from google.genai import Agent, Runner, types
                from google.genai.memory import InMemorySessionService
            except ImportError:
                from google.adk import Agent, Runner, types
                from google.adk.memory import InMemorySessionService
            
            # Create runner
            session_service = InMemorySessionService()
            runner = Runner(
                app_name="networking_evaluation",
                agent=agent,
                session_service=session_service
            )
            
            # Create session
            user_id = f"test_user_{test_case['id']}"
            session_id = f"session_{test_case['id']}"
            
            session = session_service.create_session_sync(
                app_name="networking_evaluation",
                user_id=user_id,
                session_id=session_id,
                state={}
            )
            
            # Create message
            new_message = types.Content(
                role="user",
                parts=[types.Part(text=test_case["input"])]
            )
            
            # Run agent
            response_parts = []
            tool_calls = []
            
            for event in runner.run(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message
            ):
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_parts.append(part.text)
                            elif hasattr(part, 'tool_call'):
                                tool_calls.append({
                                    "name": part.tool_call.name,
                                    "parameters": dict(part.tool_call.parameters) if hasattr(part.tool_call, 'parameters') else {}
                                })
            
            response_text = ''.join(response_parts)
            
            return {
                "test_id": test_case["id"],
                "input": test_case["input"],
                "response": response_text,
                "tool_calls": tool_calls,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Test case {test_case['id']} failed: {e}")
            return {
                "test_id": test_case["id"],
                "input": test_case["input"],
                "response": "",
                "tool_calls": [],
                "success": False,
                "error": str(e)
            }
    
    def validate_test_result(self, result: Dict, expected: Dict) -> Dict:
        """Validate test result against expected output"""
        validation = {
            "contains_check": True,
            "tool_calls_check": True,
            "missing_content": [],
            "missing_tools": [],
            "score": 0.0
        }
        
        response_lower = result["response"].lower()
        
        # Check required content
        expected_contains = expected.get("contains", [])
        for content in expected_contains:
            if content.lower() not in response_lower:
                validation["contains_check"] = False
                validation["missing_content"].append(content)
        
        # Check tool calls
        expected_tools = expected.get("tool_calls", [])
        actual_tools = result["tool_calls"]
        
        for expected_tool in expected_tools:
            tool_found = False
            for actual_tool in actual_tools:
                if (actual_tool.get("name") == expected_tool.get("name") and
                    actual_tool.get("parameters", {}).get("query_type") == 
                    expected_tool.get("parameters", {}).get("query_type")):
                    tool_found = True
                    break
            
            if not tool_found:
                validation["tool_calls_check"] = False
                validation["missing_tools"].append(expected_tool["name"])
        
        # Calculate score
        total_checks = 2
        passed_checks = sum([
            validation["contains_check"],
            validation["tool_calls_check"]
        ])
        validation["score"] = (passed_checks / total_checks) * 100
        
        return validation
    
    async def run_dataset_evaluation(self, dataset_name: str) -> Dict:
        """Run evaluation for a specific dataset"""
        logger.info(f"Running evaluation for dataset: {dataset_name}")
        
        # Load dataset
        dataset_path = self.datasets_dir / dataset_name
        dataset = self.load_evaluation_dataset(dataset_path)
        
        if not dataset.get("test_cases"):
            logger.error(f"No test cases found in {dataset_name}")
            return {"error": f"No test cases in {dataset_name}"}
        
        # Load agent
        agent = await self.load_agent()
        if not agent:
            return {"error": "Failed to load agent"}
        
        results = {
            "dataset": dataset_name,
            "total_tests": len(dataset["test_cases"]),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "summary": {
                "success_rate": 0.0,
                "avg_score": 0.0,
                "common_failures": []
            }
        }
        
        total_score = 0.0
        
        # Run each test case
        for test_case in dataset["test_cases"]:
            logger.info(f"Running test case: {test_case['id']}")
            
            # Run test
            result = await self.run_test_case(agent, test_case)
            
            if result["success"]:
                # Validate result
                validation = self.validate_test_result(result, test_case["expected_output"])
                
                test_result = {
                    "test_id": test_case["id"],
                    "test_name": test_case.get("name", ""),
                    "category": test_case.get("category", ""),
                    "input": test_case["input"],
                    "response": result["response"],
                    "tool_calls": result["tool_calls"],
                    "validation": validation,
                    "passed": validation["score"] >= 70.0  # 70% threshold
                }
                
                if test_result["passed"]:
                    results["passed_tests"] += 1
                else:
                    results["failed_tests"] += 1
                
                total_score += validation["score"]
                
            else:
                test_result = {
                    "test_id": test_case["id"],
                    "test_name": test_case.get("name", ""),
                    "category": test_case.get("category", ""),
                    "input": test_case["input"],
                    "response": "",
                    "tool_calls": [],
                    "validation": {"score": 0.0},
                    "passed": False,
                    "error": result["error"]
                }
                results["failed_tests"] += 1
            
            results["test_results"].append(test_result)
        
        # Calculate summary
        results["summary"]["success_rate"] = (results["passed_tests"] / results["total_tests"]) * 100
        results["summary"]["avg_score"] = total_score / results["total_tests"] if results["total_tests"] > 0 else 0.0
        
        return results
    
    async def run_all_evaluations(self) -> Dict:
        """Run all networking evaluations"""
        logger.info("Starting comprehensive networking evaluation...")
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "datasets": {},
            "overall_summary": {
                "total_datasets": len(self.networking_datasets),
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "overall_success_rate": 0.0,
                "avg_score": 0.0
            }
        }
        
        total_score = 0.0
        
        # Run each dataset
        for dataset_name in self.networking_datasets:
            try:
                results = await self.run_dataset_evaluation(dataset_name)
                all_results["datasets"][dataset_name] = results
                
                if "error" not in results:
                    all_results["overall_summary"]["total_tests"] += results["total_tests"]
                    all_results["overall_summary"]["total_passed"] += results["passed_tests"]
                    all_results["overall_summary"]["total_failed"] += results["failed_tests"]
                    total_score += results["summary"]["avg_score"]
                
            except Exception as e:
                logger.error(f"Failed to run dataset {dataset_name}: {e}")
                all_results["datasets"][dataset_name] = {"error": str(e)}
        
        # Calculate overall summary
        if all_results["overall_summary"]["total_tests"] > 0:
            all_results["overall_summary"]["overall_success_rate"] = (
                all_results["overall_summary"]["total_passed"] / 
                all_results["overall_summary"]["total_tests"]
            ) * 100
        
        successful_datasets = len([d for d in all_results["datasets"].values() if "error" not in d])
        if successful_datasets > 0:
            all_results["overall_summary"]["avg_score"] = total_score / successful_datasets
        
        return all_results
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate a comprehensive evaluation report"""
        report = []
        report.append("# Networking Troubleshooting Ninja - Evaluation Report")
        report.append(f"Generated: {results['timestamp']}")
        report.append("")
        
        # Overall Summary
        summary = results["overall_summary"]
        report.append("## 📊 Overall Summary")
        report.append(f"- **Total Datasets**: {summary['total_datasets']}")
        report.append(f"- **Total Test Cases**: {summary['total_tests']}")
        report.append(f"- **Passed**: {summary['total_passed']} ✅")
        report.append(f"- **Failed**: {summary['total_failed']} ❌")
        report.append(f"- **Success Rate**: {summary['overall_success_rate']:.1f}%")
        report.append(f"- **Average Score**: {summary['avg_score']:.1f}%")
        report.append("")
        
        # Dataset Results
        report.append("## 📋 Dataset Results")
        for dataset_name, dataset_results in results["datasets"].items():
            if "error" in dataset_results:
                report.append(f"### ❌ {dataset_name}")
                report.append(f"**Error**: {dataset_results['error']}")
                report.append("")
                continue
            
            report.append(f"### {dataset_name}")
            report.append(f"- **Test Cases**: {dataset_results['total_tests']}")
            report.append(f"- **Passed**: {dataset_results['passed_tests']} ✅")
            report.append(f"- **Failed**: {dataset_results['failed_tests']} ❌")
            report.append(f"- **Success Rate**: {dataset_results['summary']['success_rate']:.1f}%")
            report.append(f"- **Average Score**: {dataset_results['summary']['avg_score']:.1f}%")
            report.append("")
            
            # Test Case Details
            if dataset_results.get("test_results"):
                report.append("#### Test Case Details")
                for test in dataset_results["test_results"]:
                    status = "✅" if test["passed"] else "❌"
                    score = test["validation"]["score"] if "validation" in test else 0.0
                    report.append(f"- {status} **{test['test_id']}**: {test.get('test_name', 'N/A')} ({score:.1f}%)")
                report.append("")
        
        # Recommendations
        report.append("## 💡 Recommendations")
        if summary["overall_success_rate"] >= 90:
            report.append("🎉 Excellent performance! All networking features are working well.")
        elif summary["overall_success_rate"] >= 70:
            report.append("👍 Good performance with some areas for improvement.")
            report.append("- Review failed test cases for optimization opportunities")
            report.append("- Consider enhancing error handling and responses")
        else:
            report.append("⚠️ Significant issues detected that need attention.")
            report.append("- Review agent instructions and tool implementations")
            report.append("- Ensure database connectivity and data availability")
            report.append("- Check for missing dependencies or configuration issues")
        
        report.append("")
        report.append("## 🔧 Next Steps")
        report.append("1. Address any failing test cases")
        report.append("2. Enhance agent responses based on validation feedback")
        report.append("3. Add more comprehensive test cases for edge cases")
        report.append("4. Implement continuous evaluation for regression testing")
        
        return "\n".join(report)

async def main():
    """Main evaluation function"""
    print("🚀 Starting Networking Troubleshooting Ninja Evaluation...")
    
    runner = NetworkingEvaluationRunner()
    results = await runner.run_all_evaluations()
    
    # Generate report
    report = runner.generate_evaluation_report(results)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    results_file = results_dir / f"networking_evaluation_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save report
    report_file = results_dir / f"networking_evaluation_report_{timestamp}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📊 Evaluation Results:")
    print(f"- Total Tests: {results['overall_summary']['total_tests']}")
    print(f"- Passed: {results['overall_summary']['total_passed']} ✅")
    print(f"- Failed: {results['overall_summary']['total_failed']} ❌")
    print(f"- Success Rate: {results['overall_summary']['overall_success_rate']:.1f}%")
    print(f"- Average Score: {results['overall_summary']['avg_score']:.1f}%")
    
    print(f"\n📄 Results saved:")
    print(f"- JSON: {results_file}")
    print(f"- Report: {report_file}")
    
    print(f"\n📋 Report Summary:")
    print("-" * 50)
    for line in report.split('\n')[:20]:  # Show first 20 lines
        print(line)
    print("...")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())