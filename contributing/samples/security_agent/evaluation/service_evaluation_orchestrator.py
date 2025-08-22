#!/usr/bin/env python3
"""
Service Evaluation Orchestrator
===============================

Unified orchestrator for comprehensive service evaluation of the ADK Security Agent.
Coordinates health monitoring, performance profiling, security scanning, and
operational validation to provide a complete service assessment.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Import evaluation components
from service_health_monitor import ServiceHealthMonitor
from performance_profiler import PerformanceProfiler  
from security_scanner import SecurityScanner
from operational_validator import OperationalValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServiceEvaluationSummary:
    """Overall service evaluation summary"""
    evaluation_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    overall_score: float
    readiness_level: str
    health_score: float
    performance_score: float
    security_score: float
    operational_score: float
    critical_issues: List[str]
    recommendations: List[str]


class ServiceEvaluationOrchestrator:
    """Unified service evaluation orchestrator"""
    
    def __init__(self, config_file: str = None):
        """Initialize the evaluation orchestrator"""
        self.config = self._load_config(config_file)
        self.evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        
        # Initialize evaluation components
        self.health_monitor = ServiceHealthMonitor(config_file)
        self.performance_profiler = PerformanceProfiler(config_file)
        self.security_scanner = SecurityScanner(config_file)
        self.operational_validator = OperationalValidator(config_file)
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load orchestrator configuration"""
        default_config = {
            "backend_url": "http://localhost:8000",
            "frontend_url": "http://localhost:8501",
            "evaluation_modules": {
                "health_monitoring": True,
                "performance_profiling": True,
                "security_scanning": True,
                "operational_validation": True
            },
            "execution_mode": "parallel",  # "parallel" or "sequential"
            "output_formats": ["json", "html", "txt"],
            "report_destination": "evaluation_reports/",
            "notification_webhooks": [],
            "scoring_weights": {
                "health": 0.3,
                "performance": 0.25,
                "security": 0.25,
                "operational": 0.2
            },
            "quality_gates": {
                "minimum_overall_score": 80,
                "maximum_critical_issues": 0,
                "required_readiness_level": "STAGING_READY"
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive service evaluation"""
        logger.info(f"🚀 Starting Comprehensive Service Evaluation - {self.evaluation_id}")
        
        evaluation_results = {
            "evaluation_metadata": {
                "evaluation_id": self.evaluation_id,
                "start_time": self.start_time.isoformat(),
                "orchestrator_version": "1.0.0",
                "target_services": {
                    "backend": self.config["backend_url"],
                    "frontend": self.config["frontend_url"]
                },
                "enabled_modules": [
                    module for module, enabled in self.config["evaluation_modules"].items() 
                    if enabled
                ]
            },
            "evaluation_results": {},
            "evaluation_summary": {},
            "quality_gate_results": {},
            "recommendations": []
        }
        
        # Run evaluation modules
        if self.config["execution_mode"] == "parallel":
            logger.info("🔄 Running evaluations in parallel")
            results = await self._run_parallel_evaluation()
        else:
            logger.info("📋 Running evaluations sequentially")
            results = await self._run_sequential_evaluation()
        
        evaluation_results["evaluation_results"] = results
        
        # Calculate summary
        evaluation_summary = self._calculate_evaluation_summary(results)
        evaluation_results["evaluation_summary"] = asdict(evaluation_summary)
        
        # Check quality gates
        evaluation_results["quality_gate_results"] = self._check_quality_gates(evaluation_results)
        
        # Generate unified recommendations
        evaluation_results["recommendations"] = self._generate_unified_recommendations(results)
        
        # Finalize metadata
        end_time = datetime.now()
        evaluation_results["evaluation_metadata"]["end_time"] = end_time.isoformat()
        evaluation_results["evaluation_metadata"]["duration_seconds"] = (end_time - self.start_time).total_seconds()
        
        return evaluation_results
    
    async def _run_parallel_evaluation(self) -> Dict[str, Any]:
        """Run all evaluation modules in parallel"""
        tasks = []
        module_names = []
        
        if self.config["evaluation_modules"]["health_monitoring"]:
            tasks.append(self._run_health_monitoring())
            module_names.append("health_monitoring")
        
        if self.config["evaluation_modules"]["performance_profiling"]:
            tasks.append(self._run_performance_profiling())
            module_names.append("performance_profiling")
        
        if self.config["evaluation_modules"]["security_scanning"]:
            tasks.append(self._run_security_scanning())
            module_names.append("security_scanning")
        
        if self.config["evaluation_modules"]["operational_validation"]:
            tasks.append(self._run_operational_validation())
            module_names.append("operational_validation")
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        combined_results = {}
        for i, result in enumerate(results):
            module_name = module_names[i]
            if isinstance(result, Exception):
                combined_results[module_name] = {
                    "status": "ERROR",
                    "error": str(result),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                combined_results[module_name] = {
                    "status": "SUCCESS",
                    "results": result,
                    "timestamp": datetime.now().isoformat()
                }
        
        return combined_results
    
    async def _run_sequential_evaluation(self) -> Dict[str, Any]:
        """Run evaluation modules sequentially"""
        results = {}
        
        if self.config["evaluation_modules"]["health_monitoring"]:
            logger.info("📊 Running health monitoring...")
            try:
                health_results = await self._run_health_monitoring()
                results["health_monitoring"] = {
                    "status": "SUCCESS",
                    "results": health_results,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                results["health_monitoring"] = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        if self.config["evaluation_modules"]["performance_profiling"]:
            logger.info("⚡ Running performance profiling...")
            try:
                perf_results = await self._run_performance_profiling()
                results["performance_profiling"] = {
                    "status": "SUCCESS",
                    "results": perf_results,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                results["performance_profiling"] = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        if self.config["evaluation_modules"]["security_scanning"]:
            logger.info("🔒 Running security scanning...")
            try:
                security_results = await self._run_security_scanning()
                results["security_scanning"] = {
                    "status": "SUCCESS",
                    "results": security_results,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                results["security_scanning"] = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        if self.config["evaluation_modules"]["operational_validation"]:
            logger.info("🛠️ Running operational validation...")
            try:
                ops_results = await self._run_operational_validation()
                results["operational_validation"] = {
                    "status": "SUCCESS", 
                    "results": ops_results,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                results["operational_validation"] = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return results
    
    async def _run_health_monitoring(self) -> Dict[str, Any]:
        """Run health monitoring evaluation"""
        # For orchestrator, we run a one-time health check instead of continuous monitoring
        # Check all endpoints once
        health_results = await self.health_monitor._check_all_endpoints()
        system_health = self.health_monitor._check_system_health()
        db_health = self.health_monitor._check_database_health()
        
        # Calculate overall health status
        overall_status = self.health_monitor._calculate_overall_status(
            health_results, system_health, db_health
        )
        
        return {
            "overall_status": asdict(overall_status),
            "endpoint_results": health_results,
            "system_metrics": system_health,
            "database_health": db_health
        }
    
    async def _run_performance_profiling(self) -> Dict[str, Any]:
        """Run performance profiling evaluation"""
        return await self.performance_profiler.run_performance_analysis()
    
    async def _run_security_scanning(self) -> Dict[str, Any]:
        """Run security scanning evaluation"""
        return await self.security_scanner.run_security_assessment()
    
    async def _run_operational_validation(self) -> Dict[str, Any]:
        """Run operational validation evaluation"""
        return await self.operational_validator.run_operational_validation()
    
    def _calculate_evaluation_summary(self, results: Dict[str, Any]) -> ServiceEvaluationSummary:
        """Calculate overall evaluation summary"""
        
        # Extract scores from each module
        health_score = self._extract_health_score(results.get("health_monitoring", {}))
        performance_score = self._extract_performance_score(results.get("performance_profiling", {}))
        security_score = self._extract_security_score(results.get("security_scanning", {}))
        operational_score = self._extract_operational_score(results.get("operational_validation", {}))
        
        # Calculate weighted overall score
        weights = self.config["scoring_weights"]
        overall_score = (
            health_score * weights["health"] +
            performance_score * weights["performance"] +
            security_score * weights["security"] +
            operational_score * weights["operational"]
        )
        
        # Determine readiness level
        readiness_level = self._determine_readiness_level(overall_score, results)
        
        # Extract critical issues
        critical_issues = self._extract_critical_issues(results)
        
        # Generate summary recommendations
        recommendations = self._generate_summary_recommendations(results, overall_score)
        
        end_time = datetime.now()
        
        return ServiceEvaluationSummary(
            evaluation_id=self.evaluation_id,
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=(end_time - self.start_time).total_seconds(),
            overall_score=overall_score,
            readiness_level=readiness_level,
            health_score=health_score,
            performance_score=performance_score,
            security_score=security_score,
            operational_score=operational_score,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
    
    def _extract_health_score(self, health_results: Dict[str, Any]) -> float:
        """Extract health score from health monitoring results"""
        if health_results.get("status") != "SUCCESS":
            return 0.0
        
        results = health_results.get("results", {})
        overall_status = results.get("overall_status", {})
        
        # Calculate health score based on various factors
        status = overall_status.get("status", "DOWN")
        response_time = overall_status.get("response_time_ms", float('inf'))
        cpu_usage = overall_status.get("cpu_percent", 100)
        memory_usage = overall_status.get("memory_percent", 100)
        error_count = overall_status.get("error_count", 10)
        
        score = 100.0
        
        # Status penalty
        if status == "DOWN":
            score -= 60
        elif status == "DEGRADED":
            score -= 30
        
        # Response time penalty
        if response_time > 5000:  # >5 seconds
            score -= 20
        elif response_time > 2000:  # >2 seconds
            score -= 10
        
        # Resource usage penalty
        if cpu_usage > 90:
            score -= 15
        elif cpu_usage > 80:
            score -= 5
        
        if memory_usage > 90:
            score -= 15
        elif memory_usage > 80:
            score -= 5
        
        # Error penalty
        score -= min(error_count * 5, 20)
        
        return max(0, score)
    
    def _extract_performance_score(self, perf_results: Dict[str, Any]) -> float:
        """Extract performance score from performance profiling results"""
        if perf_results.get("status") != "SUCCESS":
            return 0.0
        
        results = perf_results.get("results", {})
        
        # Get bottleneck analysis
        bottlenecks = results.get("bottleneck_analysis", {})
        identified_bottlenecks = bottlenecks.get("identified_bottlenecks", [])
        
        # Base score
        score = 100.0
        
        # Deduct points for each bottleneck
        score -= len(identified_bottlenecks) * 15
        
        # Check load test results
        load_results = results.get("load_test_results", [])
        for load_result in load_results:
            if "error" not in load_result:
                # Check if performance targets are met
                rps = load_result.get("requests_per_second", 0)
                p95_time = load_result.get("response_times", {}).get("p95_ms", float('inf'))
                error_rate = load_result.get("error_rate_percent", 100)
                
                if rps < 10:  # Very low throughput
                    score -= 10
                if p95_time > 5000:  # Very slow responses
                    score -= 10
                if error_rate > 10:  # High error rate
                    score -= 10
        
        return max(0, score)
    
    def _extract_security_score(self, security_results: Dict[str, Any]) -> float:
        """Extract security score from security scanning results"""
        if security_results.get("status") != "SUCCESS":
            return 0.0
        
        results = security_results.get("results", {})
        findings = results.get("findings", [])
        
        # Base score
        score = 100.0
        
        # Deduct points based on finding severity
        for finding in findings:
            severity = finding.get("severity", "LOW")
            if severity == "CRITICAL":
                score -= 25
            elif severity == "HIGH":
                score -= 15
            elif severity == "MEDIUM":
                score -= 5
            elif severity == "LOW":
                score -= 1
        
        return max(0, score)
    
    def _extract_operational_score(self, ops_results: Dict[str, Any]) -> float:
        """Extract operational score from operational validation results"""
        if ops_results.get("status") != "SUCCESS":
            return 0.0
        
        results = ops_results.get("results", {})
        overall_readiness = results.get("overall_readiness", {})
        
        return overall_readiness.get("overall_score", 0.0)
    
    def _determine_readiness_level(self, overall_score: float, results: Dict[str, Any]) -> str:
        """Determine service readiness level"""
        
        # Check for critical blockers
        critical_issues = self._extract_critical_issues(results)
        if len(critical_issues) > 0:
            return "NOT_READY"
        
        # Score-based determination
        if overall_score >= 95:
            return "PRODUCTION_READY"
        elif overall_score >= 85:
            return "STAGING_READY"
        elif overall_score >= 70:
            return "DEVELOPMENT_READY"
        else:
            return "NOT_READY"
    
    def _extract_critical_issues(self, results: Dict[str, Any]) -> List[str]:
        """Extract critical issues from all evaluation results"""
        critical_issues = []
        
        # Health critical issues
        health_results = results.get("health_monitoring", {})
        if health_results.get("status") == "SUCCESS":
            health_data = health_results.get("results", {})
            overall_status = health_data.get("overall_status", {})
            if overall_status.get("status") == "DOWN":
                critical_issues.append("Service is DOWN")
        
        # Security critical issues
        security_results = results.get("security_scanning", {})
        if security_results.get("status") == "SUCCESS":
            security_data = security_results.get("results", {})
            findings = security_data.get("findings", [])
            critical_security = [f for f in findings if f.get("severity") == "CRITICAL"]
            for finding in critical_security:
                critical_issues.append(f"CRITICAL Security: {finding.get('title', 'Unknown')}")
        
        # Performance critical issues
        perf_results = results.get("performance_profiling", {})
        if perf_results.get("status") == "SUCCESS":
            perf_data = perf_results.get("results", {})
            bottlenecks = perf_data.get("bottleneck_analysis", {})
            critical_bottlenecks = bottlenecks.get("identified_bottlenecks", [])
            for bottleneck in critical_bottlenecks[:3]:  # Top 3 bottlenecks
                if "High response time" in bottleneck or "Low throughput" in bottleneck:
                    critical_issues.append(f"Performance: {bottleneck}")
        
        # Operational critical issues
        ops_results = results.get("operational_validation", {})
        if ops_results.get("status") == "SUCCESS":
            ops_data = ops_results.get("results", {})
            validation_results = ops_data.get("validation_results", [])
            failed_validations = [v for v in validation_results if v.get("status") == "FAIL"]
            for validation in failed_validations[:3]:  # Top 3 failures
                critical_issues.append(f"Operational: {validation.get('test_name', 'Unknown')}")
        
        return critical_issues
    
    def _generate_summary_recommendations(self, results: Dict[str, Any], overall_score: float) -> List[str]:
        """Generate summary recommendations"""
        recommendations = []
        
        if overall_score < 70:
            recommendations.append("System requires significant improvements before deployment")
        elif overall_score < 85:
            recommendations.append("Address remaining issues before production deployment")
        else:
            recommendations.append("System is ready for deployment with minor optimizations")
        
        # Add specific recommendations from each module
        for module_name, module_results in results.items():
            if module_results.get("status") == "SUCCESS":
                module_data = module_results.get("results", {})
                module_recommendations = module_data.get("recommendations", [])
                # Add top 2 recommendations from each module
                recommendations.extend(module_recommendations[:2])
        
        return recommendations[:10]  # Limit to top 10
    
    def _check_quality_gates(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check quality gates and determine pass/fail"""
        
        summary = evaluation_results.get("evaluation_summary", {})
        quality_gates = self.config["quality_gates"]
        
        gate_results = {
            "overall_pass": True,
            "gate_checks": {}
        }
        
        # Check minimum overall score
        overall_score = summary.get("overall_score", 0)
        min_score = quality_gates["minimum_overall_score"]
        score_pass = overall_score >= min_score
        
        gate_results["gate_checks"]["minimum_score"] = {
            "required": min_score,
            "actual": overall_score,
            "pass": score_pass
        }
        
        if not score_pass:
            gate_results["overall_pass"] = False
        
        # Check maximum critical issues
        critical_issues = summary.get("critical_issues", [])
        max_critical = quality_gates["maximum_critical_issues"]
        critical_pass = len(critical_issues) <= max_critical
        
        gate_results["gate_checks"]["maximum_critical_issues"] = {
            "required": f"<= {max_critical}",
            "actual": len(critical_issues),
            "pass": critical_pass
        }
        
        if not critical_pass:
            gate_results["overall_pass"] = False
        
        # Check required readiness level
        readiness_level = summary.get("readiness_level", "NOT_READY")
        required_level = quality_gates["required_readiness_level"]
        
        readiness_hierarchy = {
            "NOT_READY": 0,
            "DEVELOPMENT_READY": 1,
            "STAGING_READY": 2,
            "PRODUCTION_READY": 3
        }
        
        readiness_pass = readiness_hierarchy.get(readiness_level, 0) >= readiness_hierarchy.get(required_level, 3)
        
        gate_results["gate_checks"]["required_readiness_level"] = {
            "required": required_level,
            "actual": readiness_level,
            "pass": readiness_pass
        }
        
        if not readiness_pass:
            gate_results["overall_pass"] = False
        
        return gate_results
    
    def _generate_unified_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate unified recommendations across all modules"""
        unified_recommendations = []
        
        # Collect recommendations from all modules
        all_recommendations = []
        for module_name, module_results in results.items():
            if module_results.get("status") == "SUCCESS":
                module_data = module_results.get("results", {})
                module_recs = module_data.get("recommendations", [])
                all_recommendations.extend(module_recs)
        
        # Deduplicate and prioritize
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unified_recommendations.append(rec)
        
        # Add orchestrator-level recommendations
        unified_recommendations.extend([
            "Implement comprehensive monitoring across all service components",
            "Establish automated quality gates in CI/CD pipeline",
            "Create incident response runbooks for operational issues",
            "Set up regular security scanning and vulnerability assessment",
            "Implement performance regression testing",
            "Create disaster recovery and business continuity plans"
        ])
        
        # Remove duplicates again and limit
        final_recommendations = []
        seen = set()
        for rec in unified_recommendations:
            if rec not in seen:
                seen.add(rec)
                final_recommendations.append(rec)
        
        return final_recommendations[:20]  # Top 20 recommendations
    
    def save_evaluation_report(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save evaluation report in multiple formats"""
        
        # Create reports directory
        reports_dir = Path(self.config["report_destination"])
        reports_dir.mkdir(exist_ok=True)
        
        report_files = {}
        
        # JSON format
        if "json" in self.config["output_formats"]:
            json_file = reports_dir / f"{self.evaluation_id}_comprehensive_report.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            report_files["json"] = str(json_file)
            logger.info(f"📄 JSON report saved to: {json_file}")
        
        # Text format
        if "txt" in self.config["output_formats"]:
            txt_file = reports_dir / f"{self.evaluation_id}_summary.txt"
            self._generate_text_report(results, txt_file)
            report_files["txt"] = str(txt_file)
            logger.info(f"📋 Text report saved to: {txt_file}")
        
        # HTML format
        if "html" in self.config["output_formats"]:
            html_file = reports_dir / f"{self.evaluation_id}_dashboard.html"
            self._generate_html_report(results, html_file)
            report_files["html"] = str(html_file)
            logger.info(f"🌐 HTML report saved to: {html_file}")
        
        return report_files
    
    def _generate_text_report(self, results: Dict[str, Any], output_file: Path):
        """Generate comprehensive text report"""
        
        with open(output_file, 'w') as f:
            f.write("ADK Security Agent - Comprehensive Service Evaluation Report\n")
            f.write("=" * 65 + "\n\n")
            
            # Evaluation metadata
            metadata = results.get("evaluation_metadata", {})
            f.write(f"Evaluation ID: {metadata.get('evaluation_id', 'Unknown')}\n")
            f.write(f"Start Time: {metadata.get('start_time', 'Unknown')}\n")
            f.write(f"Duration: {metadata.get('duration_seconds', 0):.1f} seconds\n")
            f.write(f"Target Backend: {metadata.get('target_services', {}).get('backend', 'Unknown')}\n\n")
            
            # Overall summary
            summary = results.get("evaluation_summary", {})
            f.write("OVERALL ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Score: {summary.get('overall_score', 0):.1f}/100\n")
            f.write(f"Readiness Level: {summary.get('readiness_level', 'UNKNOWN')}\n")
            f.write(f"Critical Issues: {len(summary.get('critical_issues', []))}\n\n")
            
            # Component scores
            f.write("COMPONENT SCORES\n")
            f.write("-" * 16 + "\n")
            f.write(f"Health Monitoring: {summary.get('health_score', 0):.1f}/100\n")
            f.write(f"Performance: {summary.get('performance_score', 0):.1f}/100\n")
            f.write(f"Security: {summary.get('security_score', 0):.1f}/100\n")
            f.write(f"Operational: {summary.get('operational_score', 0):.1f}/100\n\n")
            
            # Quality gates
            quality_gates = results.get("quality_gate_results", {})
            f.write("QUALITY GATES\n")
            f.write("-" * 13 + "\n")
            f.write(f"Overall Pass: {'✅ PASS' if quality_gates.get('overall_pass') else '❌ FAIL'}\n")
            
            gate_checks = quality_gates.get("gate_checks", {})
            for gate_name, gate_data in gate_checks.items():
                status = "✅ PASS" if gate_data.get("pass") else "❌ FAIL"
                f.write(f"{gate_name}: {status}\n")
            f.write("\n")
            
            # Critical issues
            critical_issues = summary.get("critical_issues", [])
            if critical_issues:
                f.write("CRITICAL ISSUES\n")
                f.write("-" * 15 + "\n")
                for issue in critical_issues:
                    f.write(f"❌ {issue}\n")
                f.write("\n")
            
            # Top recommendations
            recommendations = results.get("recommendations", [])
            if recommendations:
                f.write("TOP RECOMMENDATIONS\n")
                f.write("-" * 19 + "\n")
                for i, rec in enumerate(recommendations[:10], 1):
                    f.write(f"{i:2d}. {rec}\n")
                f.write("\n")
            
            # Module status
            f.write("MODULE EXECUTION STATUS\n")
            f.write("-" * 23 + "\n")
            eval_results = results.get("evaluation_results", {})
            for module_name, module_data in eval_results.items():
                status = "✅ SUCCESS" if module_data.get("status") == "SUCCESS" else "❌ ERROR"
                f.write(f"{module_name}: {status}\n")
                if module_data.get("status") == "ERROR":
                    f.write(f"  Error: {module_data.get('error', 'Unknown')}\n")
    
    def _generate_html_report(self, results: Dict[str, Any], output_file: Path):
        """Generate HTML dashboard report"""
        
        summary = results.get("evaluation_summary", {})
        metadata = results.get("evaluation_metadata", {})
        quality_gates = results.get("quality_gate_results", {})
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Service Evaluation Dashboard - {metadata.get('evaluation_id', 'Unknown')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .score-card {{ display: inline-block; margin: 10px; padding: 20px; border-radius: 8px; text-align: center; min-width: 150px; }}
        .score-excellent {{ background-color: #d4edda; border-left: 5px solid #28a745; }}
        .score-good {{ background-color: #d1ecf1; border-left: 5px solid #17a2b8; }}
        .score-warning {{ background-color: #fff3cd; border-left: 5px solid #ffc107; }}
        .score-danger {{ background-color: #f8d7da; border-left: 5px solid #dc3545; }}
        .section {{ margin: 20px 0; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
        .critical-issue {{ color: #dc3545; font-weight: bold; }}
        .recommendation {{ margin: 5px 0; padding: 8px; background-color: #f8f9fa; border-left: 3px solid #007bff; }}
        .status-pass {{ color: #28a745; font-weight: bold; }}
        .status-fail {{ color: #dc3545; font-weight: bold; }}
        .module-status {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .module-success {{ background-color: #d4edda; }}
        .module-error {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ ADK Security Agent - Service Evaluation Dashboard</h1>
            <p><strong>Evaluation ID:</strong> {metadata.get('evaluation_id', 'Unknown')}</p>
            <p><strong>Generated:</strong> {metadata.get('start_time', 'Unknown')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Overall Assessment</h2>
            <div class="score-card {self._get_score_class(summary.get('overall_score', 0))}">
                <h3>Overall Score</h3>
                <div style="font-size: 2em; font-weight: bold;">{summary.get('overall_score', 0):.1f}/100</div>
                <div>Readiness: {summary.get('readiness_level', 'UNKNOWN')}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Component Scores</h2>
            <div class="score-card {self._get_score_class(summary.get('health_score', 0))}">
                <h4>Health</h4>
                <div style="font-size: 1.5em;">{summary.get('health_score', 0):.1f}/100</div>
            </div>
            <div class="score-card {self._get_score_class(summary.get('performance_score', 0))}">
                <h4>Performance</h4>
                <div style="font-size: 1.5em;">{summary.get('performance_score', 0):.1f}/100</div>
            </div>
            <div class="score-card {self._get_score_class(summary.get('security_score', 0))}">
                <h4>Security</h4>
                <div style="font-size: 1.5em;">{summary.get('security_score', 0):.1f}/100</div>
            </div>
            <div class="score-card {self._get_score_class(summary.get('operational_score', 0))}">
                <h4>Operational</h4>
                <div style="font-size: 1.5em;">{summary.get('operational_score', 0):.1f}/100</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🚪 Quality Gates</h2>
            <p><strong>Overall Status:</strong> <span class="{'status-pass' if quality_gates.get('overall_pass') else 'status-fail'}">
                {'✅ PASS' if quality_gates.get('overall_pass') else '❌ FAIL'}
            </span></p>
        </div>
        
        {self._generate_critical_issues_html(summary.get('critical_issues', []))}
        
        {self._generate_recommendations_html(results.get('recommendations', []))}
        
        {self._generate_module_status_html(results.get('evaluation_results', {}))}
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score"""
        if score >= 90:
            return "score-excellent"
        elif score >= 75:
            return "score-good"
        elif score >= 60:
            return "score-warning"
        else:
            return "score-danger"
    
    def _generate_critical_issues_html(self, critical_issues: List[str]) -> str:
        """Generate HTML for critical issues section"""
        if not critical_issues:
            return '<div class="section"><h2>✅ Critical Issues</h2><p>No critical issues found!</p></div>'
        
        html = '<div class="section"><h2>🚨 Critical Issues</h2>'
        for issue in critical_issues:
            html += f'<div class="critical-issue">❌ {issue}</div>'
        html += '</div>'
        return html
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations section"""
        if not recommendations:
            return ''
        
        html = '<div class="section"><h2>💡 Top Recommendations</h2>'
        for i, rec in enumerate(recommendations[:10], 1):
            html += f'<div class="recommendation">{i}. {rec}</div>'
        html += '</div>'
        return html
    
    def _generate_module_status_html(self, eval_results: Dict[str, Any]) -> str:
        """Generate HTML for module execution status"""
        html = '<div class="section"><h2>🔧 Module Execution Status</h2>'
        
        for module_name, module_data in eval_results.items():
            status = module_data.get("status", "UNKNOWN")
            css_class = "module-success" if status == "SUCCESS" else "module-error"
            status_text = "✅ SUCCESS" if status == "SUCCESS" else "❌ ERROR"
            
            html += f'<div class="module-status {css_class}">'
            html += f'<strong>{module_name}:</strong> {status_text}'
            
            if status == "ERROR":
                html += f'<br><small>Error: {module_data.get("error", "Unknown")}</small>'
            
            html += '</div>'
        
        html += '</div>'
        return html


async def main():
    """Main entry point for service evaluation orchestrator"""
    parser = argparse.ArgumentParser(description="Comprehensive Service Evaluation Orchestrator")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--parallel", action="store_true", help="Run evaluations in parallel")
    parser.add_argument("--modules", nargs="*", choices=["health", "performance", "security", "operational"], 
                       help="Specific modules to run")
    parser.add_argument("--output-formats", nargs="*", choices=["json", "html", "txt"], 
                       default=["json", "txt"], help="Output formats")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = ServiceEvaluationOrchestrator(args.config)
    
    # Override config with command line arguments
    if args.parallel:
        orchestrator.config["execution_mode"] = "parallel"
    
    if args.modules:
        # Disable all modules first
        for module in orchestrator.config["evaluation_modules"]:
            orchestrator.config["evaluation_modules"][module] = False
        
        # Enable specified modules
        module_map = {
            "health": "health_monitoring",
            "performance": "performance_profiling", 
            "security": "security_scanning",
            "operational": "operational_validation"
        }
        
        for module in args.modules:
            if module in module_map:
                orchestrator.config["evaluation_modules"][module_map[module]] = True
    
    if args.output_formats:
        orchestrator.config["output_formats"] = args.output_formats
    
    try:
        # Run comprehensive evaluation
        logger.info("🚀 Starting Comprehensive Service Evaluation")
        results = await orchestrator.run_comprehensive_evaluation()
        
        # Save reports
        report_files = orchestrator.save_evaluation_report(results)
        
        # Print summary
        summary = results.get("evaluation_summary", {})
        quality_gates = results.get("quality_gate_results", {})
        
        print(f"\n🎯 Service Evaluation Complete!")
        print(f"Evaluation ID: {summary.get('evaluation_id', 'Unknown')}")
        print(f"Overall Score: {summary.get('overall_score', 0):.1f}/100")
        print(f"Readiness Level: {summary.get('readiness_level', 'UNKNOWN')}")
        print(f"Quality Gates: {'✅ PASS' if quality_gates.get('overall_pass') else '❌ FAIL'}")
        
        critical_issues = summary.get("critical_issues", [])
        if critical_issues:
            print(f"\n🚨 Critical Issues ({len(critical_issues)}):")
            for issue in critical_issues[:5]:  # Show first 5
                print(f"  ❌ {issue}")
        else:
            print("\n✅ No critical issues found!")
        
        print(f"\n📄 Reports Generated:")
        for format_type, file_path in report_files.items():
            print(f"  {format_type.upper()}: {file_path}")
        
        # Exit with appropriate code based on quality gates
        if quality_gates.get("overall_pass"):
            print("\n🎉 All quality gates passed - service is ready!")
            exit(0)
        else:
            print("\n⚠️ Some quality gates failed - review issues before deployment")
            exit(1)
            
    except Exception as e:
        logger.error(f"❌ Service evaluation failed: {e}")
        exit(2)


if __name__ == "__main__":
    asyncio.run(main())