#!/usr/bin/env python3
"""
Operational Validator
=====================

Comprehensive operational readiness validation for the ADK Security Agent.
Validates deployment readiness, monitoring setup, documentation completeness,
and production operational requirements.
"""

import asyncio
import json
import logging
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Operational validation result"""
    category: str
    test_name: str
    status: str  # PASS, FAIL, WARNING, SKIP
    score: float  # 0-100
    message: str
    evidence: Dict[str, Any]
    recommendations: List[str]


@dataclass
class OperationalScore:
    """Operational readiness score"""
    category: str
    score: float
    max_score: float
    passed_tests: int
    failed_tests: int
    warning_tests: int


class OperationalValidator:
    """Comprehensive operational readiness validation"""
    
    def __init__(self, config_file: str = None):
        """Initialize the operational validator"""
        self.config = self._load_config(config_file)
        self.results = []
        self.project_root = Path(__file__).parent.parent
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load validator configuration"""
        default_config = {
            "backend_url": "http://localhost:8000",
            "frontend_url": "http://localhost:8501",
            "project_root": "..",
            "validation_categories": {
                "deployment": True,
                "monitoring": True,
                "documentation": True,
                "configuration": True,
                "security": True,
                "backup": True,
                "performance": True
            },
            "deployment_files": [
                "docker-compose.yml",
                "Dockerfile",
                "requirements.txt",
                ".env.template"
            ],
            "documentation_files": [
                "README.md",
                "CLAUDE.md",
                "docs/deployment.md",
                "docs/troubleshooting.md"
            ],
            "monitoring_endpoints": [
                "/health",
                "/metrics",
                "/status"
            ]
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def run_operational_validation(self) -> Dict[str, Any]:
        """Run comprehensive operational validation"""
        logger.info("🛠️ Starting Operational Validation")
        
        validation_results = {
            "validation_metadata": {
                "start_time": datetime.now().isoformat(),
                "validator_version": "1.0.0",
                "project_root": str(self.project_root),
                "categories_tested": list(self.config["validation_categories"].keys())
            },
            "category_scores": {},
            "validation_results": [],
            "overall_readiness": {},
            "recommendations": []
        }
        
        # Run validation categories
        if self.config["validation_categories"]["deployment"]:
            await self._validate_deployment_readiness()
        
        if self.config["validation_categories"]["monitoring"]:
            await self._validate_monitoring_setup()
        
        if self.config["validation_categories"]["documentation"]:
            await self._validate_documentation()
        
        if self.config["validation_categories"]["configuration"]:
            await self._validate_configuration()
        
        if self.config["validation_categories"]["security"]:
            await self._validate_security_setup()
        
        if self.config["validation_categories"]["backup"]:
            await self._validate_backup_systems()
        
        if self.config["validation_categories"]["performance"]:
            await self._validate_performance_setup()
        
        # Calculate scores
        validation_results["category_scores"] = self._calculate_category_scores()
        
        # Add results
        validation_results["validation_results"] = [asdict(r) for r in self.results]
        
        # Calculate overall readiness
        validation_results["overall_readiness"] = self._calculate_overall_readiness()
        
        # Generate recommendations
        validation_results["recommendations"] = self._generate_recommendations()
        
        validation_results["validation_metadata"]["end_time"] = datetime.now().isoformat()
        
        return validation_results
    
    async def _validate_deployment_readiness(self):
        """Validate deployment configuration and readiness"""
        logger.info("📦 Validating deployment readiness...")
        
        # Check for deployment files
        for file_name in self.config["deployment_files"]:
            file_path = self.project_root / file_name
            
            if file_path.exists():
                self._add_result(
                    category="Deployment",
                    test_name=f"Deployment file: {file_name}",
                    status="PASS",
                    score=25,
                    message=f"Required deployment file exists",
                    evidence={"file_path": str(file_path), "exists": True},
                    recommendations=[]
                )
            else:
                self._add_result(
                    category="Deployment",
                    test_name=f"Deployment file: {file_name}",
                    status="FAIL",
                    score=0,
                    message=f"Required deployment file missing",
                    evidence={"file_path": str(file_path), "exists": False},
                    recommendations=[f"Create {file_name} for deployment"]
                )
        
        # Check Docker configuration
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            try:
                with open(dockerfile_path, 'r') as f:
                    dockerfile_content = f.read()
                
                # Check for security best practices
                security_checks = {
                    "non_root_user": "USER " in dockerfile_content,
                    "specific_versions": not re.search(r":latest|FROM.*ubuntu$|FROM.*python$", dockerfile_content),
                    "minimal_layers": dockerfile_content.count("RUN") < 10,
                }
                
                score = sum(security_checks.values()) / len(security_checks) * 100
                
                self._add_result(
                    category="Deployment",
                    test_name="Dockerfile security practices",
                    status="PASS" if score > 70 else "WARNING",
                    score=score,
                    message=f"Dockerfile follows {score:.0f}% of security best practices",
                    evidence=security_checks,
                    recommendations=self._get_dockerfile_recommendations(security_checks)
                )
                
            except Exception as e:
                self._add_result(
                    category="Deployment",
                    test_name="Dockerfile validation",
                    status="WARNING",
                    score=50,
                    message=f"Could not validate Dockerfile: {e}",
                    evidence={"error": str(e)},
                    recommendations=["Review Dockerfile syntax and content"]
                )
        
        # Check environment configuration
        env_template = self.project_root / ".env.template"
        env_file = self.project_root / ".env"
        
        if env_template.exists():
            try:
                with open(env_template, 'r') as f:
                    template_vars = self._extract_env_vars(f.read())
                
                env_vars = {}
                if env_file.exists():
                    with open(env_file, 'r') as f:
                        env_vars = self._extract_env_vars(f.read())
                
                missing_vars = set(template_vars.keys()) - set(env_vars.keys())
                configured_vars = set(template_vars.keys()) & set(env_vars.keys())
                
                score = len(configured_vars) / len(template_vars) * 100 if template_vars else 100
                
                self._add_result(
                    category="Deployment",
                    test_name="Environment configuration",
                    status="PASS" if score > 80 else "WARNING",
                    score=score,
                    message=f"{len(configured_vars)}/{len(template_vars)} environment variables configured",
                    evidence={
                        "template_vars": list(template_vars.keys()),
                        "configured_vars": list(configured_vars),
                        "missing_vars": list(missing_vars)
                    },
                    recommendations=[f"Configure missing environment variables: {', '.join(missing_vars)}"] if missing_vars else []
                )
                
            except Exception as e:
                self._add_result(
                    category="Deployment",
                    test_name="Environment configuration",
                    status="WARNING",
                    score=50,
                    message=f"Could not validate environment config: {e}",
                    evidence={"error": str(e)},
                    recommendations=["Review environment configuration files"]
                )
    
    async def _validate_monitoring_setup(self):
        """Validate monitoring and observability setup"""
        logger.info("📊 Validating monitoring setup...")
        
        # Check monitoring endpoints
        for endpoint in self.config["monitoring_endpoints"]:
            url = f"{self.config['backend_url']}{endpoint}"
            
            try:
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    self._add_result(
                        category="Monitoring",
                        test_name=f"Monitoring endpoint: {endpoint}",
                        status="PASS",
                        score=33,
                        message="Monitoring endpoint is accessible",
                        evidence={"status_code": response.status_code, "response_size": len(response.text)},
                        recommendations=[]
                    )
                else:
                    self._add_result(
                        category="Monitoring",
                        test_name=f"Monitoring endpoint: {endpoint}",
                        status="FAIL",
                        score=0,
                        message=f"Monitoring endpoint returned {response.status_code}",
                        evidence={"status_code": response.status_code},
                        recommendations=[f"Fix monitoring endpoint {endpoint}"]
                    )
                    
            except Exception as e:
                self._add_result(
                    category="Monitoring",
                    test_name=f"Monitoring endpoint: {endpoint}",
                    status="FAIL",
                    score=0,
                    message=f"Monitoring endpoint not accessible: {e}",
                    evidence={"error": str(e)},
                    recommendations=[f"Ensure monitoring endpoint {endpoint} is available"]
                )
        
        # Check for monitoring configuration files
        monitoring_files = [
            "monitoring/prometheus.yml",
            "monitoring/grafana.yml",
            "monitoring/alerts.yml",
            "docker-compose.monitoring.yml"
        ]
        
        monitoring_file_count = 0
        for file_name in monitoring_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                monitoring_file_count += 1
        
        score = (monitoring_file_count / len(monitoring_files)) * 100
        
        self._add_result(
            category="Monitoring",
            test_name="Monitoring configuration files",
            status="PASS" if score > 50 else "WARNING",
            score=score,
            message=f"{monitoring_file_count}/{len(monitoring_files)} monitoring files present",
            evidence={"files_found": monitoring_file_count, "total_files": len(monitoring_files)},
            recommendations=["Add monitoring configuration files"] if score < 50 else []
        )
        
        # Check logging configuration
        log_config_files = [
            "logging.conf",
            "log4j.properties",
            "logback.xml"
        ]
        
        log_config_exists = any((self.project_root / f).exists() for f in log_config_files)
        
        self._add_result(
            category="Monitoring",
            test_name="Logging configuration",
            status="PASS" if log_config_exists else "WARNING",
            score=100 if log_config_exists else 50,
            message="Logging configuration found" if log_config_exists else "No explicit logging configuration",
            evidence={"log_config_exists": log_config_exists},
            recommendations=["Add structured logging configuration"] if not log_config_exists else []
        )
    
    async def _validate_documentation(self):
        """Validate documentation completeness"""
        logger.info("📚 Validating documentation...")
        
        # Check for required documentation files
        for doc_file in self.config["documentation_files"]:
            file_path = self.project_root / doc_file
            
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Basic content validation
                    word_count = len(content.split())
                    has_headers = bool(re.search(r'^#+\s+', content, re.MULTILINE))
                    has_code_blocks = '```' in content or '`' in content
                    
                    quality_score = 0
                    if word_count > 100:
                        quality_score += 40
                    if has_headers:
                        quality_score += 30
                    if has_code_blocks:
                        quality_score += 30
                    
                    self._add_result(
                        category="Documentation",
                        test_name=f"Documentation: {doc_file}",
                        status="PASS" if quality_score > 70 else "WARNING",
                        score=quality_score,
                        message=f"Documentation exists with {word_count} words",
                        evidence={"word_count": word_count, "has_headers": has_headers, "has_code": has_code_blocks},
                        recommendations=["Improve documentation quality"] if quality_score <= 70 else []
                    )
                    
                except Exception as e:
                    self._add_result(
                        category="Documentation",
                        test_name=f"Documentation: {doc_file}",
                        status="WARNING",
                        score=50,
                        message=f"Documentation exists but could not be validated: {e}",
                        evidence={"error": str(e)},
                        recommendations=["Review documentation file format"]
                    )
            else:
                self._add_result(
                    category="Documentation",
                    test_name=f"Documentation: {doc_file}",
                    status="FAIL",
                    score=0,
                    message="Required documentation file missing",
                    evidence={"file_path": str(file_path), "exists": False},
                    recommendations=[f"Create {doc_file}"]
                )
        
        # Check for API documentation
        api_docs_paths = [
            "docs/api.md",
            "docs/swagger.json",
            "docs/openapi.yaml"
        ]
        
        api_docs_exist = any((self.project_root / p).exists() for p in api_docs_paths)
        
        self._add_result(
            category="Documentation",
            test_name="API documentation",
            status="PASS" if api_docs_exist else "WARNING",
            score=100 if api_docs_exist else 25,
            message="API documentation found" if api_docs_exist else "No API documentation found",
            evidence={"api_docs_exist": api_docs_exist},
            recommendations=["Create API documentation"] if not api_docs_exist else []
        )
        
        # Check for runbooks
        runbook_paths = [
            "runbooks/",
            "docs/runbooks/",
            "operations/"
        ]
        
        runbooks_exist = any((self.project_root / p).exists() and any((self.project_root / p).iterdir()) for p in runbook_paths)
        
        self._add_result(
            category="Documentation",
            test_name="Operational runbooks",
            status="PASS" if runbooks_exist else "WARNING",
            score=100 if runbooks_exist else 25,
            message="Operational runbooks found" if runbooks_exist else "No operational runbooks found",
            evidence={"runbooks_exist": runbooks_exist},
            recommendations=["Create operational runbooks"] if not runbooks_exist else []
        )
    
    async def _validate_configuration(self):
        """Validate configuration management"""
        logger.info("⚙️ Validating configuration...")
        
        # Check configuration files
        config_files = [
            ".env.template",
            "config.json",
            "settings.yaml",
            "application.yml"
        ]
        
        config_files_found = sum(1 for f in config_files if (self.project_root / f).exists())
        score = (config_files_found / len(config_files)) * 100
        
        self._add_result(
            category="Configuration",
            test_name="Configuration files",
            status="PASS" if score > 25 else "WARNING",
            score=score,
            message=f"{config_files_found}/{len(config_files)} configuration files found",
            evidence={"files_found": config_files_found},
            recommendations=["Add configuration files"] if score < 50 else []
        )
        
        # Check for secrets management
        secrets_indicators = [
            ".env.template",
            "secrets/",
            "vault/",
            "k8s-secrets/"
        ]
        
        secrets_setup = any((self.project_root / s).exists() for s in secrets_indicators)
        
        self._add_result(
            category="Configuration",
            test_name="Secrets management",
            status="PASS" if secrets_setup else "WARNING",
            score=100 if secrets_setup else 25,
            message="Secrets management setup found" if secrets_setup else "No secrets management setup",
            evidence={"secrets_setup": secrets_setup},
            recommendations=["Implement secrets management"] if not secrets_setup else []
        )
        
        # Check environment separation
        env_files = [
            ".env.development",
            ".env.staging", 
            ".env.production",
            "config/development.yml",
            "config/production.yml"
        ]
        
        env_separation = sum(1 for f in env_files if (self.project_root / f).exists())
        score = min((env_separation / 2) * 100, 100)  # Need at least 2 environments
        
        self._add_result(
            category="Configuration",
            test_name="Environment separation",
            status="PASS" if score > 50 else "WARNING",
            score=score,
            message=f"{env_separation} environment configurations found",
            evidence={"env_configs": env_separation},
            recommendations=["Add environment-specific configurations"] if score < 50 else []
        )
    
    async def _validate_security_setup(self):
        """Validate security configuration"""
        logger.info("🔒 Validating security setup...")
        
        # Check for security configuration files
        security_files = [
            "security.yml",
            "cors.conf",
            "ssl/",
            "certs/"
        ]
        
        security_files_found = sum(1 for f in security_files if (self.project_root / f).exists())
        score = (security_files_found / len(security_files)) * 100
        
        self._add_result(
            category="Security",
            test_name="Security configuration files",
            status="PASS" if score > 25 else "WARNING",
            score=score,
            message=f"{security_files_found}/{len(security_files)} security files found",
            evidence={"files_found": security_files_found},
            recommendations=["Add security configuration files"] if score < 50 else []
        )
        
        # Check for authentication setup
        auth_indicators = [
            "auth/",
            "authentication.yml",
            "oauth.conf",
            "jwt.key"
        ]
        
        auth_setup = any((self.project_root / a).exists() for a in auth_indicators)
        
        self._add_result(
            category="Security",
            test_name="Authentication setup",
            status="PASS" if auth_setup else "WARNING",
            score=100 if auth_setup else 25,
            message="Authentication setup found" if auth_setup else "No authentication setup found",
            evidence={"auth_setup": auth_setup},
            recommendations=["Implement authentication"] if not auth_setup else []
        )
        
        # Check for HTTPS configuration
        https_indicators = [
            "ssl/",
            "certs/",
            "tls.conf",
            "nginx.ssl.conf"
        ]
        
        https_setup = any((self.project_root / h).exists() for h in https_indicators)
        
        self._add_result(
            category="Security",
            test_name="HTTPS/TLS setup",
            status="PASS" if https_setup else "WARNING",
            score=100 if https_setup else 25,
            message="HTTPS/TLS setup found" if https_setup else "No HTTPS/TLS setup found",
            evidence={"https_setup": https_setup},
            recommendations=["Configure HTTPS/TLS"] if not https_setup else []
        )
    
    async def _validate_backup_systems(self):
        """Validate backup and recovery systems"""
        logger.info("💾 Validating backup systems...")
        
        # Check for backup scripts
        backup_files = [
            "scripts/backup.sh",
            "backup/",
            "scripts/restore.sh",
            "backup.yml"
        ]
        
        backup_files_found = sum(1 for f in backup_files if (self.project_root / f).exists())
        score = (backup_files_found / len(backup_files)) * 100
        
        self._add_result(
            category="Backup",
            test_name="Backup scripts and configuration",
            status="PASS" if score > 50 else "WARNING",
            score=score,
            message=f"{backup_files_found}/{len(backup_files)} backup files found",
            evidence={"files_found": backup_files_found},
            recommendations=["Implement backup scripts"] if score < 50 else []
        )
        
        # Check for disaster recovery plan
        dr_files = [
            "docs/disaster-recovery.md",
            "runbooks/disaster-recovery.md",
            "dr-plan.md"
        ]
        
        dr_plan_exists = any((self.project_root / f).exists() for f in dr_files)
        
        self._add_result(
            category="Backup",
            test_name="Disaster recovery plan",
            status="PASS" if dr_plan_exists else "WARNING",
            score=100 if dr_plan_exists else 25,
            message="Disaster recovery plan found" if dr_plan_exists else "No disaster recovery plan",
            evidence={"dr_plan_exists": dr_plan_exists},
            recommendations=["Create disaster recovery plan"] if not dr_plan_exists else []
        )
        
        # Check database backup configuration
        db_backup_indicators = [
            "scripts/db_backup.sh",
            "backup/database/",
            "db-backup.yml"
        ]
        
        db_backup_setup = any((self.project_root / b).exists() for b in db_backup_indicators)
        
        self._add_result(
            category="Backup",
            test_name="Database backup setup",
            status="PASS" if db_backup_setup else "WARNING",
            score=100 if db_backup_setup else 25,
            message="Database backup setup found" if db_backup_setup else "No database backup setup",
            evidence={"db_backup_setup": db_backup_setup},
            recommendations=["Implement database backup"] if not db_backup_setup else []
        )
    
    async def _validate_performance_setup(self):
        """Validate performance monitoring and optimization"""
        logger.info("⚡ Validating performance setup...")
        
        # Check for performance monitoring
        perf_files = [
            "monitoring/performance.yml",
            "apm.conf",
            "newrelic.yml",
            "datadog.yml"
        ]
        
        perf_monitoring = any((self.project_root / f).exists() for f in perf_files)
        
        self._add_result(
            category="Performance",
            test_name="Performance monitoring setup",
            status="PASS" if perf_monitoring else "WARNING",
            score=100 if perf_monitoring else 25,
            message="Performance monitoring found" if perf_monitoring else "No performance monitoring setup",
            evidence={"perf_monitoring": perf_monitoring},
            recommendations=["Implement performance monitoring"] if not perf_monitoring else []
        )
        
        # Check for load testing
        load_test_files = [
            "tests/load/",
            "performance/",
            "locustfile.py",
            "jmeter.jmx"
        ]
        
        load_testing = any((self.project_root / f).exists() for f in load_test_files)
        
        self._add_result(
            category="Performance",
            test_name="Load testing setup",
            status="PASS" if load_testing else "WARNING",
            score=100 if load_testing else 25,
            message="Load testing setup found" if load_testing else "No load testing setup",
            evidence={"load_testing": load_testing},
            recommendations=["Implement load testing"] if not load_testing else []
        )
        
        # Check for caching configuration
        cache_indicators = [
            "redis.conf",
            "memcached.conf",
            "cache/",
            "caching.yml"
        ]
        
        caching_setup = any((self.project_root / c).exists() for c in cache_indicators)
        
        self._add_result(
            category="Performance",
            test_name="Caching setup",
            status="PASS" if caching_setup else "WARNING",
            score=100 if caching_setup else 50,
            message="Caching setup found" if caching_setup else "No caching setup found",
            evidence={"caching_setup": caching_setup},
            recommendations=["Implement caching"] if not caching_setup else []
        )
    
    def _add_result(self, category: str, test_name: str, status: str, 
                   score: float, message: str, evidence: Dict[str, Any], 
                   recommendations: List[str]):
        """Add a validation result"""
        result = ValidationResult(
            category=category,
            test_name=test_name,
            status=status,
            score=score,
            message=message,
            evidence=evidence,
            recommendations=recommendations
        )
        
        self.results.append(result)
        
        # Log failures and warnings
        if status in ["FAIL", "WARNING"]:
            logger.warning(f"⚠️ {category} - {test_name}: {message}")
    
    def _calculate_category_scores(self) -> Dict[str, OperationalScore]:
        """Calculate scores by category"""
        categories = set(r.category for r in self.results)
        scores = {}
        
        for category in categories:
            category_results = [r for r in self.results if r.category == category]
            
            total_score = sum(r.score for r in category_results)
            max_score = len(category_results) * 100
            
            passed = sum(1 for r in category_results if r.status == "PASS")
            failed = sum(1 for r in category_results if r.status == "FAIL")
            warnings = sum(1 for r in category_results if r.status == "WARNING")
            
            scores[category] = OperationalScore(
                category=category,
                score=total_score / max_score * 100 if max_score > 0 else 0,
                max_score=max_score,
                passed_tests=passed,
                failed_tests=failed,
                warning_tests=warnings
            )
        
        return scores
    
    def _calculate_overall_readiness(self) -> Dict[str, Any]:
        """Calculate overall operational readiness"""
        category_scores = self._calculate_category_scores()
        
        if not category_scores:
            return {"readiness_level": "NOT_ASSESSED", "overall_score": 0}
        
        # Calculate weighted average
        weights = {
            "Deployment": 0.25,
            "Monitoring": 0.20,
            "Documentation": 0.15,
            "Configuration": 0.15,
            "Security": 0.15,
            "Backup": 0.05,
            "Performance": 0.05
        }
        
        weighted_score = 0
        total_weight = 0
        
        for category, score_data in category_scores.items():
            weight = weights.get(category, 0.1)
            weighted_score += score_data.score * weight
            total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine readiness level
        if overall_score >= 90:
            readiness_level = "PRODUCTION_READY"
        elif overall_score >= 75:
            readiness_level = "STAGING_READY"
        elif overall_score >= 60:
            readiness_level = "DEVELOPMENT_READY"
        else:
            readiness_level = "NOT_READY"
        
        return {
            "readiness_level": readiness_level,
            "overall_score": overall_score,
            "category_breakdown": {cat: data.score for cat, data in category_scores.items()}
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate operational recommendations"""
        recommendations = []
        
        # Collect all recommendations from failed tests
        failed_results = [r for r in self.results if r.status == "FAIL"]
        warning_results = [r for r in self.results if r.status == "WARNING"]
        
        # Priority recommendations from failed tests
        for result in failed_results:
            recommendations.extend(result.recommendations)
        
        # Secondary recommendations from warnings
        for result in warning_results[:5]:  # Limit to top 5 warnings
            recommendations.extend(result.recommendations)
        
        # General operational recommendations
        recommendations.extend([
            "Implement comprehensive monitoring and alerting",
            "Create detailed operational documentation",
            "Set up automated deployment pipeline",
            "Establish backup and recovery procedures",
            "Implement security best practices",
            "Create performance benchmarks",
            "Set up disaster recovery plan"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:20]  # Limit to top 20
    
    def _extract_env_vars(self, content: str) -> Dict[str, str]:
        """Extract environment variables from file content"""
        env_vars = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
        
        return env_vars
    
    def _get_dockerfile_recommendations(self, security_checks: Dict[str, bool]) -> List[str]:
        """Get Dockerfile security recommendations"""
        recommendations = []
        
        if not security_checks.get("non_root_user"):
            recommendations.append("Add non-root user to Dockerfile")
        
        if not security_checks.get("specific_versions"):
            recommendations.append("Use specific version tags instead of 'latest'")
        
        if not security_checks.get("minimal_layers"):
            recommendations.append("Minimize Docker layers by combining RUN commands")
        
        return recommendations
    
    def save_validation_report(self, results: Dict[str, Any], filename: str = "operational_validation.json"):
        """Save operational validation report"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Operational validation report saved to {filename}")
        
        # Generate summary report
        self._generate_summary_report(results)
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate human-readable summary report"""
        summary_path = "operational_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("ADK Security Agent - Operational Validation Summary\n")
            f.write("=" * 55 + "\n\n")
            
            # Overall readiness
            readiness = results.get("overall_readiness", {})
            f.write(f"Readiness Level: {readiness.get('readiness_level', 'UNKNOWN')}\n")
            f.write(f"Overall Score: {readiness.get('overall_score', 0):.1f}/100\n\n")
            
            # Category scores
            f.write("Category Scores:\n")
            category_scores = results.get("category_scores", {})
            for category, score_data in category_scores.items():
                f.write(f"  {category}: {score_data['score']:.1f}/100 ")
                f.write(f"({score_data['passed_tests']} pass, {score_data['failed_tests']} fail, {score_data['warning_tests']} warn)\n")
            f.write("\n")
            
            # Top recommendations
            recommendations = results.get("recommendations", [])
            if recommendations:
                f.write("Top Recommendations:\n")
                for i, rec in enumerate(recommendations[:10], 1):
                    f.write(f"  {i}. {rec}\n")
            
        logger.info(f"📋 Operational summary saved to {summary_path}")


async def main():
    """Run operational validator"""
    validator = OperationalValidator()
    
    try:
        results = await validator.run_operational_validation()
        
        # Save results
        validator.save_validation_report(results)
        
        # Print summary
        readiness = results.get("overall_readiness", {})
        readiness_level = readiness.get("readiness_level", "UNKNOWN")
        overall_score = readiness.get("overall_score", 0)
        
        print(f"\n🛠️ Operational Validation Complete!")
        print(f"Readiness Level: {readiness_level}")
        print(f"Overall Score: {overall_score:.1f}/100")
        
        if readiness_level == "PRODUCTION_READY":
            print("✅ Service is ready for production deployment!")
        elif readiness_level == "STAGING_READY":
            print("🟡 Service is ready for staging environment")
        else:
            print("⚠️ Service needs operational improvements")
            
        # Show category breakdown
        category_scores = results.get("category_scores", {})
        if category_scores:
            print("\nCategory Scores:")
            for category, score_data in category_scores.items():
                status_icon = "✅" if score_data["score"] > 80 else "⚠️" if score_data["score"] > 60 else "❌"
                print(f"  {status_icon} {category}: {score_data['score']:.1f}/100")
            
    except Exception as e:
        logger.error(f"❌ Operational validation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())