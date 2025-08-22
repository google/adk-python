#!/usr/bin/env python3
"""
Security Scanner
================

Comprehensive security assessment tool for the ADK Security Agent.
Performs security testing, vulnerability scanning, and compliance validation.
"""

import asyncio
import json
import logging
import time
import requests
import ssl
import socket
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import urllib.parse
import hashlib
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Security assessment finding"""
    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    title: str
    description: str
    affected_component: str
    recommendation: str
    evidence: Dict[str, Any]
    timestamp: str


@dataclass
class SecurityScore:
    """Security assessment score"""
    category: str
    score: float  # 0-100
    max_score: float
    findings_count: int
    critical_findings: int
    high_findings: int


class SecurityScanner:
    """Comprehensive security assessment tool"""
    
    def __init__(self, config_file: str = None):
        """Initialize the security scanner"""
        self.config = self._load_config(config_file)
        self.findings = []
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load scanner configuration"""
        default_config = {
            "backend_url": "http://localhost:8000",
            "frontend_url": "http://localhost:8501",
            "test_endpoints": [
                "/api/v1/custom-roles/stats",
                "/api/v1/knowledge/stats",
                "/api/v1/iam/policies",
                "/health"
            ],
            "sensitive_endpoints": [
                "/api/v1/custom-roles/analyze",
                "/api/v1/knowledge/policies"
            ],
            "security_tests": {
                "input_validation": True,
                "authentication": True,
                "authorization": True,
                "ssl_tls": True,
                "headers": True,
                "information_disclosure": True,
                "injection_attacks": True
            },
            "compliance_frameworks": ["OWASP_TOP10", "SOC2", "GDPR"]
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def run_security_assessment(self) -> Dict[str, Any]:
        """Run comprehensive security assessment"""
        logger.info("🔒 Starting Security Assessment")
        
        assessment_results = {
            "assessment_metadata": {
                "start_time": datetime.now().isoformat(),
                "scanner_version": "1.0.0",
                "target_url": self.config["backend_url"],
                "frameworks": self.config["compliance_frameworks"]
            },
            "security_scores": {},
            "findings": [],
            "compliance_status": {},
            "recommendations": []
        }
        
        # Run security tests
        if self.config["security_tests"]["input_validation"]:
            await self._test_input_validation()
        
        if self.config["security_tests"]["authentication"]:
            await self._test_authentication()
        
        if self.config["security_tests"]["authorization"]:
            await self._test_authorization()
        
        if self.config["security_tests"]["ssl_tls"]:
            await self._test_ssl_tls()
        
        if self.config["security_tests"]["headers"]:
            await self._test_security_headers()
        
        if self.config["security_tests"]["information_disclosure"]:
            await self._test_information_disclosure()
        
        if self.config["security_tests"]["injection_attacks"]:
            await self._test_injection_attacks()
        
        # Calculate security scores
        assessment_results["security_scores"] = self._calculate_security_scores()
        
        # Add findings
        assessment_results["findings"] = [asdict(f) for f in self.findings]
        
        # Check compliance
        assessment_results["compliance_status"] = self._check_compliance()
        
        # Generate recommendations
        assessment_results["recommendations"] = self._generate_security_recommendations()
        
        assessment_results["assessment_metadata"]["end_time"] = datetime.now().isoformat()
        
        return assessment_results
    
    async def _test_input_validation(self):
        """Test input validation and sanitization"""
        logger.info("🧪 Testing input validation...")
        
        # Test various malicious inputs
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "%3Cscript%3Ealert('xss')%3C/script%3E",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "{{7*7}}",  # Template injection
            "${7*7}",   # Expression injection
            "../../../../windows/system32/cmd.exe",
            "\x00\x01\x02\x03",  # Null bytes and control chars
        ]
        
        for endpoint in self.config["test_endpoints"]:
            url = f"{self.config['backend_url']}{endpoint}"
            
            for malicious_input in malicious_inputs:
                # Test in query parameters
                test_url = f"{url}?search={urllib.parse.quote(malicious_input)}"
                
                try:
                    response = requests.get(test_url, timeout=5)
                    
                    # Check if malicious input is reflected in response
                    if malicious_input in response.text or malicious_input.replace("'", "\\'") in response.text:
                        self._add_finding(
                            category="Input Validation",
                            severity="HIGH",
                            title="Potential XSS Vulnerability",
                            description=f"Malicious input reflected in response without sanitization",
                            affected_component=endpoint,
                            recommendation="Implement proper input sanitization and output encoding",
                            evidence={"input": malicious_input, "reflected": True}
                        )
                    
                    # Check for SQL error messages
                    sql_errors = ["sql", "database", "mysql", "postgres", "sqlite", "syntax error"]
                    response_lower = response.text.lower()
                    
                    if any(error in response_lower for error in sql_errors):
                        self._add_finding(
                            category="Input Validation",
                            severity="MEDIUM",
                            title="Database Error Information Disclosure",
                            description="Database error messages exposed in response",
                            affected_component=endpoint,
                            recommendation="Implement generic error handling",
                            evidence={"input": malicious_input, "error_disclosed": True}
                        )
                        
                except Exception as e:
                    # Server errors might indicate successful injection
                    if "500" in str(e) or "error" in str(e).lower():
                        self._add_finding(
                            category="Input Validation",
                            severity="MEDIUM",
                            title="Server Error on Malicious Input",
                            description="Server generated error when processing malicious input",
                            affected_component=endpoint,
                            recommendation="Implement robust input validation",
                            evidence={"input": malicious_input, "error": str(e)}
                        )
    
    async def _test_authentication(self):
        """Test authentication mechanisms"""
        logger.info("🔐 Testing authentication...")
        
        # Test endpoints without authentication
        for endpoint in self.config["sensitive_endpoints"]:
            url = f"{self.config['backend_url']}{endpoint}"
            
            try:
                response = requests.get(url, timeout=5)
                
                # If sensitive endpoint returns data without auth, it's a problem
                if response.status_code == 200 and len(response.text) > 100:
                    self._add_finding(
                        category="Authentication",
                        severity="HIGH",
                        title="Missing Authentication on Sensitive Endpoint",
                        description="Sensitive endpoint accessible without authentication",
                        affected_component=endpoint,
                        recommendation="Implement authentication requirements for sensitive endpoints",
                        evidence={"status_code": response.status_code, "accessible": True}
                    )
                
            except Exception as e:
                logger.debug(f"Auth test error for {endpoint}: {e}")
        
        # Test for common authentication bypasses
        bypass_headers = [
            {"X-Forwarded-For": "127.0.0.1"},
            {"X-Real-IP": "127.0.0.1"},
            {"X-Originating-IP": "127.0.0.1"},
            {"X-Remote-IP": "127.0.0.1"},
            {"X-Client-IP": "127.0.0.1"},
            {"Authorization": "Bearer invalid_token"},
            {"Authorization": "Basic YWRtaW46YWRtaW4="},  # admin:admin
        ]
        
        for endpoint in self.config["sensitive_endpoints"]:
            url = f"{self.config['backend_url']}{endpoint}"
            
            for headers in bypass_headers:
                try:
                    response = requests.get(url, headers=headers, timeout=5)
                    
                    if response.status_code == 200 and len(response.text) > 100:
                        self._add_finding(
                            category="Authentication",
                            severity="MEDIUM",
                            title="Potential Authentication Bypass",
                            description="Endpoint may be bypassable with specific headers",
                            affected_component=endpoint,
                            recommendation="Review authentication logic for header-based bypasses",
                            evidence={"bypass_headers": headers, "accessible": True}
                        )
                        
                except Exception as e:
                    logger.debug(f"Auth bypass test error: {e}")
    
    async def _test_authorization(self):
        """Test authorization and access controls"""
        logger.info("🛡️ Testing authorization...")
        
        # Test for horizontal privilege escalation
        test_ids = ["1", "2", "admin", "test", "../admin", "1'", "1 OR 1=1"]
        
        for endpoint in self.config["sensitive_endpoints"]:
            for test_id in test_ids:
                url = f"{self.config['backend_url']}{endpoint}/{test_id}"
                
                try:
                    response = requests.get(url, timeout=5)
                    
                    # If we get data for different IDs, might be IDOR
                    if response.status_code == 200 and "error" not in response.text.lower():
                        self._add_finding(
                            category="Authorization",
                            severity="MEDIUM",
                            title="Potential Insecure Direct Object Reference",
                            description="Endpoint may allow access to unauthorized resources",
                            affected_component=endpoint,
                            recommendation="Implement proper authorization checks for resource access",
                            evidence={"test_id": test_id, "accessible": True}
                        )
                        
                except Exception as e:
                    logger.debug(f"Authorization test error: {e}")
    
    async def _test_ssl_tls(self):
        """Test SSL/TLS configuration"""
        logger.info("🔒 Testing SSL/TLS configuration...")
        
        # Extract hostname from backend URL
        hostname = self.config['backend_url'].replace('http://', '').replace('https://', '').split(':')[0]
        
        if hostname == 'localhost' or self.config['backend_url'].startswith('http://'):
            self._add_finding(
                category="SSL/TLS",
                severity="HIGH",
                title="No SSL/TLS Encryption",
                description="Service is not using HTTPS encryption",
                affected_component="Transport Layer",
                recommendation="Implement HTTPS with proper SSL/TLS configuration",
                evidence={"protocol": "HTTP", "encrypted": False}
            )
            return
        
        try:
            # Test SSL/TLS configuration
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    protocol = ssock.version()
                    cipher = ssock.cipher()
                    
                    # Check protocol version
                    if protocol in ['SSLv2', 'SSLv3', 'TLSv1', 'TLSv1.1']:
                        self._add_finding(
                            category="SSL/TLS",
                            severity="HIGH",
                            title="Weak TLS Protocol Version",
                            description=f"Using weak TLS protocol: {protocol}",
                            affected_component="Transport Layer",
                            recommendation="Upgrade to TLS 1.2 or 1.3",
                            evidence={"protocol": protocol, "secure": False}
                        )
                    
                    # Check cipher strength
                    if cipher and cipher[1] < 128:
                        self._add_finding(
                            category="SSL/TLS",
                            severity="MEDIUM",
                            title="Weak Cipher Suite",
                            description=f"Using weak cipher with {cipher[1]} bit encryption",
                            affected_component="Transport Layer",
                            recommendation="Configure strong cipher suites (256-bit minimum)",
                            evidence={"cipher": cipher, "bits": cipher[1]}
                        )
                    
                    # Check certificate expiration
                    if cert:
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (not_after - datetime.now()).days
                        
                        if days_until_expiry < 30:
                            self._add_finding(
                                category="SSL/TLS",
                                severity="MEDIUM",
                                title="Certificate Expiring Soon",
                                description=f"SSL certificate expires in {days_until_expiry} days",
                                affected_component="Transport Layer",
                                recommendation="Renew SSL certificate before expiration",
                                evidence={"expires_in_days": days_until_expiry}
                            )
                        
        except Exception as e:
            self._add_finding(
                category="SSL/TLS",
                severity="INFO",
                title="SSL/TLS Test Incomplete",
                description="Could not fully test SSL/TLS configuration",
                affected_component="Transport Layer",
                recommendation="Manually verify SSL/TLS configuration",
                evidence={"error": str(e)}
            )
    
    async def _test_security_headers(self):
        """Test security headers"""
        logger.info("🛡️ Testing security headers...")
        
        url = f"{self.config['backend_url']}/health"
        
        try:
            response = requests.get(url, timeout=5)
            headers = response.headers
            
            # Required security headers
            required_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=',
                'Content-Security-Policy': 'default-src',
                'Referrer-Policy': 'strict-origin-when-cross-origin'
            }
            
            for header_name, expected_value in required_headers.items():
                if header_name not in headers:
                    self._add_finding(
                        category="Security Headers",
                        severity="MEDIUM",
                        title=f"Missing Security Header: {header_name}",
                        description=f"Security header {header_name} is not set",
                        affected_component="HTTP Headers",
                        recommendation=f"Add {header_name} header with appropriate value",
                        evidence={"missing_header": header_name}
                    )
                else:
                    header_value = headers[header_name]
                    
                    if isinstance(expected_value, list):
                        if not any(exp in header_value for exp in expected_value):
                            self._add_finding(
                                category="Security Headers",
                                severity="LOW",
                                title=f"Weak Security Header: {header_name}",
                                description=f"Security header {header_name} may have weak configuration",
                                affected_component="HTTP Headers",
                                recommendation=f"Review {header_name} header configuration",
                                evidence={"header_value": header_value}
                            )
                    elif expected_value not in header_value:
                        self._add_finding(
                            category="Security Headers",
                            severity="LOW",
                            title=f"Weak Security Header: {header_name}",
                            description=f"Security header {header_name} may have weak configuration",
                            affected_component="HTTP Headers",
                            recommendation=f"Review {header_name} header configuration",
                            evidence={"header_value": header_value}
                        )
            
            # Check for information disclosure headers
            disclosure_headers = ['Server', 'X-Powered-By', 'X-AspNet-Version']
            for header_name in disclosure_headers:
                if header_name in headers:
                    self._add_finding(
                        category="Information Disclosure",
                        severity="LOW",
                        title=f"Information Disclosure Header: {header_name}",
                        description=f"Header {header_name} discloses server information",
                        affected_component="HTTP Headers",
                        recommendation=f"Remove or mask {header_name} header",
                        evidence={"disclosed_info": headers[header_name]}
                    )
                    
        except Exception as e:
            logger.debug(f"Security headers test error: {e}")
    
    async def _test_information_disclosure(self):
        """Test for information disclosure vulnerabilities"""
        logger.info("📄 Testing information disclosure...")
        
        # Test for sensitive files
        sensitive_paths = [
            "/.env",
            "/config.json",
            "/backup.sql",
            "/debug.log",
            "/error.log",
            "/admin",
            "/api/docs",
            "/swagger",
            "/api/swagger",
            "/api/v1/docs"
        ]
        
        for path in sensitive_paths:
            url = f"{self.config['backend_url']}{path}"
            
            try:
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    self._add_finding(
                        category="Information Disclosure",
                        severity="MEDIUM",
                        title=f"Sensitive Path Accessible: {path}",
                        description="Sensitive file or endpoint is publicly accessible",
                        affected_component=path,
                        recommendation="Restrict access to sensitive files and directories",
                        evidence={"status_code": response.status_code, "accessible": True}
                    )
                    
            except Exception as e:
                logger.debug(f"Info disclosure test error for {path}: {e}")
        
        # Test for error message disclosure
        error_inducing_requests = [
            ("Invalid JSON", {"Content-Type": "application/json"}, "invalid json"),
            ("Long URL", {}, "A" * 10000),
            ("Invalid method", {}, "")
        ]
        
        for test_name, headers, data in error_inducing_requests:
            for endpoint in self.config["test_endpoints"]:
                url = f"{self.config['backend_url']}{endpoint}"
                
                try:
                    if data:
                        response = requests.post(url, headers=headers, data=data, timeout=5)
                    else:
                        response = requests.request("INVALID", url, timeout=5)
                    
                    # Check for detailed error messages
                    error_indicators = ["traceback", "exception", "stack trace", "debug", "file not found"]
                    response_lower = response.text.lower()
                    
                    if any(indicator in response_lower for indicator in error_indicators):
                        self._add_finding(
                            category="Information Disclosure",
                            severity="LOW",
                            title="Detailed Error Messages",
                            description="Application returns detailed error messages",
                            affected_component=endpoint,
                            recommendation="Implement generic error handling",
                            evidence={"test": test_name, "error_disclosed": True}
                        )
                        
                except Exception as e:
                    logger.debug(f"Error disclosure test error: {e}")
    
    async def _test_injection_attacks(self):
        """Test for injection vulnerabilities"""
        logger.info("💉 Testing injection attacks...")
        
        # SQL injection payloads
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT 1,2,3 --",
            "admin'--",
            "1' OR 1=1#"
        ]
        
        # NoSQL injection payloads
        nosql_payloads = [
            '{"$ne": null}',
            '{"$gt": ""}',
            '{"$regex": ".*"}'
        ]
        
        # Command injection payloads
        command_payloads = [
            "; ls -la",
            "| whoami",
            "&& id",
            "`id`",
            "$(id)"
        ]
        
        all_payloads = sql_payloads + nosql_payloads + command_payloads
        
        for endpoint in self.config["test_endpoints"]:
            for payload in all_payloads:
                # Test in URL parameters
                url = f"{self.config['backend_url']}{endpoint}?id={urllib.parse.quote(payload)}"
                
                try:
                    response = requests.get(url, timeout=10)
                    
                    # Check for signs of successful injection
                    success_indicators = [
                        "syntax error", "mysql", "postgresql", "sqlite", "oracle",
                        "root:", "uid=", "gid=", "groups=",  # Command injection
                        "database", "table", "column",
                        "permission denied", "access denied"
                    ]
                    
                    response_lower = response.text.lower()
                    
                    for indicator in success_indicators:
                        if indicator in response_lower:
                            self._add_finding(
                                category="Injection",
                                severity="HIGH",
                                title="Potential Injection Vulnerability",
                                description=f"Injection payload may have been executed: {payload}",
                                affected_component=endpoint,
                                recommendation="Implement parameterized queries and input sanitization",
                                evidence={"payload": payload, "indicator": indicator}
                            )
                            break
                            
                except Exception as e:
                    # Timeouts might indicate successful injection
                    if "timeout" in str(e).lower():
                        self._add_finding(
                            category="Injection",
                            severity="MEDIUM",
                            title="Potential Injection - Timeout",
                            description="Request timed out with injection payload",
                            affected_component=endpoint,
                            recommendation="Investigate potential injection vulnerability",
                            evidence={"payload": payload, "error": "timeout"}
                        )
    
    def _add_finding(self, category: str, severity: str, title: str, 
                    description: str, affected_component: str, 
                    recommendation: str, evidence: Dict[str, Any]):
        """Add a security finding"""
        finding = SecurityFinding(
            category=category,
            severity=severity,
            title=title,
            description=description,
            affected_component=affected_component,
            recommendation=recommendation,
            evidence=evidence,
            timestamp=datetime.now().isoformat()
        )
        
        self.findings.append(finding)
        
        # Log critical and high severity findings
        if severity in ["CRITICAL", "HIGH"]:
            logger.warning(f"🚨 {severity}: {title} - {affected_component}")
    
    def _calculate_security_scores(self) -> Dict[str, SecurityScore]:
        """Calculate security scores by category"""
        categories = ["Input Validation", "Authentication", "Authorization", 
                     "SSL/TLS", "Security Headers", "Information Disclosure", "Injection"]
        
        scores = {}
        
        for category in categories:
            category_findings = [f for f in self.findings if f.category == category]
            
            # Base score is 100, deduct points for findings
            score = 100.0
            critical_count = sum(1 for f in category_findings if f.severity == "CRITICAL")
            high_count = sum(1 for f in category_findings if f.severity == "HIGH")
            medium_count = sum(1 for f in category_findings if f.severity == "MEDIUM")
            low_count = sum(1 for f in category_findings if f.severity == "LOW")
            
            # Deduct points based on severity
            score -= critical_count * 30
            score -= high_count * 20
            score -= medium_count * 10
            score -= low_count * 5
            
            score = max(0, score)  # Don't go below 0
            
            scores[category] = SecurityScore(
                category=category,
                score=score,
                max_score=100.0,
                findings_count=len(category_findings),
                critical_findings=critical_count,
                high_findings=high_count
            )
        
        return scores
    
    def _check_compliance(self) -> Dict[str, Any]:
        """Check compliance with security frameworks"""
        compliance_status = {}
        
        for framework in self.config["compliance_frameworks"]:
            if framework == "OWASP_TOP10":
                compliance_status[framework] = self._check_owasp_compliance()
            elif framework == "SOC2":
                compliance_status[framework] = self._check_soc2_compliance()
            elif framework == "GDPR":
                compliance_status[framework] = self._check_gdpr_compliance()
        
        return compliance_status
    
    def _check_owasp_compliance(self) -> Dict[str, Any]:
        """Check OWASP Top 10 compliance"""
        owasp_categories = {
            "A01_Broken_Access_Control": ["Authorization", "Authentication"],
            "A02_Cryptographic_Failures": ["SSL/TLS"],
            "A03_Injection": ["Injection", "Input Validation"],
            "A05_Security_Misconfiguration": ["Security Headers"],
            "A06_Vulnerable_Components": ["Information Disclosure"],
            "A07_Authentication_Failures": ["Authentication"],
            "A09_Security_Logging": ["Information Disclosure"],
            "A10_Server_Side_Request_Forgery": ["Input Validation"]
        }
        
        compliance_results = {}
        
        for owasp_cat, finding_categories in owasp_categories.items():
            relevant_findings = [
                f for f in self.findings 
                if f.category in finding_categories and f.severity in ["CRITICAL", "HIGH"]
            ]
            
            compliance_results[owasp_cat] = {
                "compliant": len(relevant_findings) == 0,
                "findings_count": len(relevant_findings),
                "risk_level": "HIGH" if relevant_findings else "LOW"
            }
        
        return compliance_results
    
    def _check_soc2_compliance(self) -> Dict[str, Any]:
        """Check SOC 2 compliance basics"""
        # Basic SOC 2 security criteria
        encryption_findings = [f for f in self.findings if f.category == "SSL/TLS" and f.severity in ["CRITICAL", "HIGH"]]
        access_findings = [f for f in self.findings if f.category in ["Authentication", "Authorization"] and f.severity in ["CRITICAL", "HIGH"]]
        
        return {
            "security_principle": {
                "compliant": len(encryption_findings + access_findings) == 0,
                "findings_count": len(encryption_findings + access_findings),
                "issues": ["Encryption", "Access Control"] if encryption_findings + access_findings else []
            }
        }
    
    def _check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check basic GDPR compliance"""
        # Basic GDPR security requirements
        security_findings = [f for f in self.findings if f.severity in ["CRITICAL", "HIGH"]]
        
        return {
            "data_protection": {
                "compliant": len(security_findings) == 0,
                "findings_count": len(security_findings),
                "security_measures": "ADEQUATE" if len(security_findings) == 0 else "NEEDS_IMPROVEMENT"
            }
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Critical and high severity findings
        critical_high_findings = [f for f in self.findings if f.severity in ["CRITICAL", "HIGH"]]
        
        if critical_high_findings:
            recommendations.append("Address all critical and high severity security findings immediately")
        
        # Category-specific recommendations
        categories_with_findings = set(f.category for f in self.findings)
        
        if "SSL/TLS" in categories_with_findings:
            recommendations.append("Implement proper SSL/TLS configuration with strong ciphers")
        
        if "Authentication" in categories_with_findings:
            recommendations.append("Implement robust authentication mechanisms for all sensitive endpoints")
        
        if "Input Validation" in categories_with_findings:
            recommendations.append("Implement comprehensive input validation and output encoding")
        
        if "Security Headers" in categories_with_findings:
            recommendations.append("Configure all required security headers")
        
        if "Injection" in categories_with_findings:
            recommendations.append("Use parameterized queries and implement injection protection")
        
        # General recommendations
        recommendations.extend([
            "Implement security monitoring and alerting",
            "Conduct regular security assessments",
            "Establish secure coding practices",
            "Implement security awareness training",
            "Create incident response procedures"
        ])
        
        return recommendations
    
    def save_security_report(self, results: Dict[str, Any], filename: str = "security_report.json"):
        """Save security assessment report"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Security report saved to {filename}")
        
        # Generate summary report
        self._generate_summary_report(results)
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate human-readable summary report"""
        summary_path = "security_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("ADK Security Agent - Security Assessment Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            findings = results.get("findings", [])
            critical_count = sum(1 for f in findings if f["severity"] == "CRITICAL")
            high_count = sum(1 for f in findings if f["severity"] == "HIGH")
            medium_count = sum(1 for f in findings if f["severity"] == "MEDIUM")
            low_count = sum(1 for f in findings if f["severity"] == "LOW")
            
            f.write(f"Total Findings: {len(findings)}\n")
            f.write(f"Critical: {critical_count}\n")
            f.write(f"High: {high_count}\n")
            f.write(f"Medium: {medium_count}\n")
            f.write(f"Low: {low_count}\n\n")
            
            # Security scores
            scores = results.get("security_scores", {})
            f.write("Security Scores:\n")
            for category, score_data in scores.items():
                f.write(f"  {category}: {score_data['score']:.1f}/100\n")
            f.write("\n")
            
            # Top findings
            if critical_count > 0 or high_count > 0:
                f.write("Critical/High Severity Findings:\n")
                for finding in findings:
                    if finding["severity"] in ["CRITICAL", "HIGH"]:
                        f.write(f"  - {finding['title']} ({finding['affected_component']})\n")
                f.write("\n")
            
            # Compliance status
            compliance = results.get("compliance_status", {})
            f.write("Compliance Status:\n")
            for framework, status in compliance.items():
                f.write(f"  {framework}: {status}\n")
            f.write("\n")
            
            # Recommendations
            recommendations = results.get("recommendations", [])
            if recommendations:
                f.write("Top Recommendations:\n")
                for i, rec in enumerate(recommendations[:10], 1):
                    f.write(f"  {i}. {rec}\n")
        
        logger.info(f"📋 Security summary saved to {summary_path}")


async def main():
    """Run security scanner"""
    scanner = SecurityScanner()
    
    try:
        results = await scanner.run_security_assessment()
        
        # Save results
        scanner.save_security_report(results)
        
        # Print summary
        findings = results.get("findings", [])
        critical_count = sum(1 for f in findings if f["severity"] == "CRITICAL")
        high_count = sum(1 for f in findings if f["severity"] == "HIGH")
        
        print(f"\n🔒 Security Assessment Complete!")
        print(f"Total Findings: {len(findings)}")
        print(f"Critical: {critical_count}")
        print(f"High: {high_count}")
        
        if critical_count > 0:
            print("❌ Critical security issues found - immediate action required")
        elif high_count > 0:
            print("⚠️ High severity issues found - prompt action recommended")
        else:
            print("✅ No critical or high severity issues found")
            
    except Exception as e:
        logger.error(f"❌ Security assessment failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())