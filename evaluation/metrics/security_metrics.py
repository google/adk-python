"""
Security-specific evaluation metrics

Custom metrics for evaluating security agent performance including
vulnerability detection accuracy, risk assessment quality, and security response scoring.
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from google.adk.evaluation.evaluator import Evaluator, EvaluationResult, EvalStatus, PerInvocationResult
from google.adk.evaluation.eval_case import Invocation

logger = logging.getLogger(__name__)


class VulnerabilityType(Enum):
    """Common vulnerability types for detection scoring"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    AUTHORIZATION_FLAWS = "authorization_flaws"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXPOSURE = "data_exposure"
    CONFIGURATION_WEAKNESS = "configuration_weakness"
    CRYPTOGRAPHIC_FAILURES = "cryptographic_failures"
    INJECTION_ATTACKS = "injection_attacks"


class RiskLevel(Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityFinding:
    """Represents a security finding for evaluation"""
    finding_id: str
    vulnerability_type: VulnerabilityType
    risk_level: RiskLevel
    title: str
    description: str
    location: str
    evidence: List[str]
    remediation: Optional[str] = None
    confidence: float = 1.0


@dataclass
class SecurityMetricResult:
    """Result of security metric evaluation"""
    metric_name: str
    score: float
    max_score: float
    details: Dict[str, Any]
    findings_comparison: Optional[Dict[str, Any]] = None


class SecurityAccuracyMetric(Evaluator):
    """
    Evaluates overall security analysis accuracy by comparing
    actual vs expected security findings and assessments.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.vulnerability_patterns = {
            VulnerabilityType.SQL_INJECTION: [
                r'sql\s+injection', r'sqli', r'union\s+select', r'drop\s+table',
                r'insert\s+into', r'delete\s+from', r'update\s+.*\s+set'
            ],
            VulnerabilityType.XSS: [
                r'cross[- ]?site\s+scripting', r'xss', r'<script[^>]*>',
                r'javascript:', r'onerror=', r'onload='
            ],
            VulnerabilityType.AUTHENTICATION_BYPASS: [
                r'auth\w*\s+bypass', r'authentication\s+bypass', r'login\s+bypass',
                r'session\s+hijack', r'credential\s+bypass'
            ],
            VulnerabilityType.PRIVILEGE_ESCALATION: [
                r'privilege\s+escalation', r'escalation', r'elevated\s+privileges',
                r'admin\s+access', r'root\s+access', r'unauthorized\s+access'
            ]
        }
        
    def evaluate_invocations(
        self,
        actual_invocations: List[Invocation],
        expected_invocations: List[Invocation]
    ) -> EvaluationResult:
        """Evaluate security accuracy across invocations"""
        
        per_invocation_results = []
        total_score = 0.0
        
        for actual, expected in zip(actual_invocations, expected_invocations):
            result = self._evaluate_single_invocation(actual, expected)
            per_invocation_results.append(result)
            total_score += result.score or 0.0
        
        overall_score = total_score / len(actual_invocations) if actual_invocations else 0.0
        overall_status = EvalStatus.PASSED if overall_score >= self.threshold else EvalStatus.FAILED
        
        return EvaluationResult(
            overall_score=overall_score,
            overall_eval_status=overall_status,
            per_invocation_results=per_invocation_results
        )
    
    def _evaluate_single_invocation(self, actual: Invocation, expected: Invocation) -> PerInvocationResult:
        """Evaluate security accuracy for a single invocation"""
        
        actual_text = self._extract_text(actual.final_response)
        expected_text = self._extract_text(expected.final_response)
        
        # Extract security findings from both responses
        actual_findings = self._extract_security_findings(actual_text)
        expected_findings = self._extract_security_findings(expected_text)
        
        # Calculate accuracy metrics
        precision, recall, f1_score = self._calculate_finding_metrics(actual_findings, expected_findings)
        
        # Calculate content similarity for security-specific terms
        content_similarity = self._calculate_security_content_similarity(actual_text, expected_text)
        
        # Combined score
        score = (f1_score * 0.6) + (content_similarity * 0.4)
        status = EvalStatus.PASSED if score >= self.threshold else EvalStatus.FAILED
        
        return PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=score,
            eval_status=status
        )
    
    def _extract_text(self, content) -> str:
        """Extract text from response content"""
        if not content or not content.parts:
            return ""
        return " ".join(part.text for part in content.parts if hasattr(part, 'text'))
    
    def _extract_security_findings(self, text: str) -> Set[VulnerabilityType]:
        """Extract security findings from text using pattern matching"""
        text_lower = text.lower()
        findings = set()
        
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    findings.add(vuln_type)
                    break
        
        return findings
    
    def _calculate_finding_metrics(self, actual: Set[VulnerabilityType], expected: Set[VulnerabilityType]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score for findings"""
        if not expected and not actual:
            return 1.0, 1.0, 1.0
        
        if not expected:
            return 0.0, 1.0, 0.0  # No expected findings, but agent found some
        
        if not actual:
            return 1.0, 0.0, 0.0  # Expected findings, but agent found none
        
        intersection = actual.intersection(expected)
        precision = len(intersection) / len(actual)
        recall = len(intersection) / len(expected)
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score
    
    def _calculate_security_content_similarity(self, actual: str, expected: str) -> float:
        """Calculate similarity based on security-specific terms"""
        security_terms = [
            'vulnerability', 'security', 'risk', 'threat', 'exploit', 'attack',
            'malicious', 'unauthorized', 'breach', 'compromise', 'exposure',
            'encryption', 'authentication', 'authorization', 'access control',
            'firewall', 'monitoring', 'logging', 'audit', 'compliance'
        ]
        
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        actual_terms = set(term for term in security_terms if term in actual_lower)
        expected_terms = set(term for term in security_terms if term in expected_lower)
        
        if not expected_terms:
            return 1.0 if not actual_terms else 0.5
        
        intersection = actual_terms.intersection(expected_terms)
        return len(intersection) / len(expected_terms)


class VulnerabilityDetectionMetric(Evaluator):
    """
    Specialized metric for evaluating vulnerability detection capabilities.
    Focuses on accuracy of identifying specific vulnerability types.
    """
    
    def __init__(self, threshold: float = 0.9, weight_by_severity: bool = True):
        self.threshold = threshold
        self.weight_by_severity = weight_by_severity
        self.severity_weights = {
            RiskLevel.CRITICAL: 1.0,
            RiskLevel.HIGH: 0.8,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.LOW: 0.4,
            RiskLevel.INFO: 0.2
        }
    
    def evaluate_invocations(
        self,
        actual_invocations: List[Invocation],
        expected_invocations: List[Invocation]
    ) -> EvaluationResult:
        """Evaluate vulnerability detection across invocations"""
        
        per_invocation_results = []
        total_score = 0.0
        
        for actual, expected in zip(actual_invocations, expected_invocations):
            result = self._evaluate_vulnerability_detection(actual, expected)
            per_invocation_results.append(result)
            total_score += result.score or 0.0
        
        overall_score = total_score / len(actual_invocations) if actual_invocations else 0.0
        overall_status = EvalStatus.PASSED if overall_score >= self.threshold else EvalStatus.FAILED
        
        return EvaluationResult(
            overall_score=overall_score,
            overall_eval_status=overall_status,
            per_invocation_results=per_invocation_results
        )
    
    def _evaluate_vulnerability_detection(self, actual: Invocation, expected: Invocation) -> PerInvocationResult:
        """Evaluate vulnerability detection for a single invocation"""
        
        actual_text = self._extract_text(actual.final_response)
        expected_text = self._extract_text(expected.final_response)
        
        # Parse structured vulnerability data if available
        actual_vulns = self._parse_vulnerability_data(actual_text)
        expected_vulns = self._parse_vulnerability_data(expected_text)
        
        # Calculate detection score
        detection_score = self._calculate_detection_score(actual_vulns, expected_vulns)
        
        status = EvalStatus.PASSED if detection_score >= self.threshold else EvalStatus.FAILED
        
        return PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=detection_score,
            eval_status=status
        )
    
    def _extract_text(self, content) -> str:
        """Extract text from response content"""
        if not content or not content.parts:
            return ""
        return " ".join(part.text for part in content.parts if hasattr(part, 'text'))
    
    def _parse_vulnerability_data(self, text: str) -> List[Dict[str, Any]]:
        """Parse vulnerability data from text"""
        vulnerabilities = []
        
        # Try to parse JSON-structured vulnerability data
        try:
            # Look for JSON blocks in the text
            json_pattern = r'\{[^{}]*"vulnerability"[^{}]*\}'
            json_matches = re.findall(json_pattern, text, re.IGNORECASE | re.DOTALL)
            
            for match in json_matches:
                try:
                    vuln_data = json.loads(match)
                    vulnerabilities.append(vuln_data)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        # Fallback to pattern-based extraction
        if not vulnerabilities:
            vulnerabilities = self._extract_vulnerabilities_by_pattern(text)
        
        return vulnerabilities
    
    def _extract_vulnerabilities_by_pattern(self, text: str) -> List[Dict[str, Any]]:
        """Extract vulnerabilities using pattern matching"""
        vulnerabilities = []
        
        # Pattern for vulnerability entries
        vuln_pattern = r'(?:vulnerability|finding|issue):\s*([^\n]+)'
        severity_pattern = r'(?:severity|risk|priority):\s*(\w+)'
        
        vuln_matches = re.findall(vuln_pattern, text, re.IGNORECASE)
        severity_matches = re.findall(severity_pattern, text, re.IGNORECASE)
        
        for i, vuln in enumerate(vuln_matches):
            severity = severity_matches[i] if i < len(severity_matches) else 'medium'
            vulnerabilities.append({
                'title': vuln.strip(),
                'severity': severity.lower(),
                'type': self._classify_vulnerability_type(vuln)
            })
        
        return vulnerabilities
    
    def _classify_vulnerability_type(self, description: str) -> str:
        """Classify vulnerability type from description"""
        description_lower = description.lower()
        
        if any(term in description_lower for term in ['sql', 'injection']):
            return 'sql_injection'
        elif any(term in description_lower for term in ['xss', 'script', 'javascript']):
            return 'xss'
        elif any(term in description_lower for term in ['auth', 'login', 'bypass']):
            return 'authentication_bypass'
        elif any(term in description_lower for term in ['privilege', 'escalation', 'elevated']):
            return 'privilege_escalation'
        else:
            return 'other'
    
    def _calculate_detection_score(self, actual: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> float:
        """Calculate vulnerability detection score"""
        if not expected:
            return 1.0 if not actual else 0.5
        
        if not actual:
            return 0.0
        
        # Match vulnerabilities by type and severity
        matched_vulns = 0
        total_weight = 0
        matched_weight = 0
        
        for expected_vuln in expected:
            expected_type = expected_vuln.get('type', 'other')
            expected_severity = expected_vuln.get('severity', 'medium')
            
            # Calculate weight for this vulnerability
            severity_weight = self.severity_weights.get(
                RiskLevel(expected_severity), 0.6
            ) if self.weight_by_severity else 1.0
            total_weight += severity_weight
            
            # Look for matching vulnerability in actual results
            for actual_vuln in actual:
                actual_type = actual_vuln.get('type', 'other')
                actual_severity = actual_vuln.get('severity', 'medium')
                
                # Check for type match (exact or similar)
                if (expected_type == actual_type or 
                    self._types_similar(expected_type, actual_type)):
                    
                    # Bonus for severity match
                    severity_bonus = 1.0 if expected_severity == actual_severity else 0.8
                    matched_weight += severity_weight * severity_bonus
                    matched_vulns += 1
                    break
        
        # Calculate final score
        if total_weight > 0:
            return min(matched_weight / total_weight, 1.0)
        else:
            return 1.0 if not actual else 0.5
    
    def _types_similar(self, type1: str, type2: str) -> bool:
        """Check if two vulnerability types are similar"""
        similarity_groups = [
            {'sql_injection', 'injection_attacks'},
            {'xss', 'javascript_injection'},
            {'authentication_bypass', 'authorization_flaws'},
            {'privilege_escalation', 'unauthorized_access'}
        ]
        
        for group in similarity_groups:
            if type1 in group and type2 in group:
                return True
        
        return False


class RiskAssessmentMetric(Evaluator):
    """
    Evaluates the quality of risk assessment including
    risk scoring, impact analysis, and prioritization.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.risk_indicators = {
            'impact_terms': ['impact', 'consequence', 'damage', 'effect', 'result'],
            'likelihood_terms': ['likelihood', 'probability', 'chance', 'risk', 'potential'],
            'priority_terms': ['priority', 'urgent', 'critical', 'immediate', 'high'],
            'mitigation_terms': ['mitigation', 'remediation', 'fix', 'solution', 'patch']
        }
    
    def evaluate_invocations(
        self,
        actual_invocations: List[Invocation],
        expected_invocations: List[Invocation]
    ) -> EvaluationResult:
        """Evaluate risk assessment quality across invocations"""
        
        per_invocation_results = []
        total_score = 0.0
        
        for actual, expected in zip(actual_invocations, expected_invocations):
            result = self._evaluate_risk_assessment(actual, expected)
            per_invocation_results.append(result)
            total_score += result.score or 0.0
        
        overall_score = total_score / len(actual_invocations) if actual_invocations else 0.0
        overall_status = EvalStatus.PASSED if overall_score >= self.threshold else EvalStatus.FAILED
        
        return EvaluationResult(
            overall_score=overall_score,
            overall_eval_status=overall_status,
            per_invocation_results=per_invocation_results
        )
    
    def _evaluate_risk_assessment(self, actual: Invocation, expected: Invocation) -> PerInvocationResult:
        """Evaluate risk assessment for a single invocation"""
        
        actual_text = self._extract_text(actual.final_response)
        expected_text = self._extract_text(expected.final_response)
        
        # Evaluate different aspects of risk assessment
        impact_score = self._evaluate_impact_analysis(actual_text, expected_text)
        likelihood_score = self._evaluate_likelihood_analysis(actual_text, expected_text)
        priority_score = self._evaluate_prioritization(actual_text, expected_text)
        mitigation_score = self._evaluate_mitigation_recommendations(actual_text, expected_text)
        
        # Combined risk assessment score
        overall_score = (
            impact_score * 0.3 +
            likelihood_score * 0.25 +
            priority_score * 0.25 +
            mitigation_score * 0.2
        )
        
        status = EvalStatus.PASSED if overall_score >= self.threshold else EvalStatus.FAILED
        
        return PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=overall_score,
            eval_status=status
        )
    
    def _extract_text(self, content) -> str:
        """Extract text from response content"""
        if not content or not content.parts:
            return ""
        return " ".join(part.text for part in content.parts if hasattr(part, 'text'))
    
    def _evaluate_impact_analysis(self, actual: str, expected: str) -> float:
        """Evaluate quality of impact analysis"""
        return self._evaluate_term_coverage(actual, expected, self.risk_indicators['impact_terms'])
    
    def _evaluate_likelihood_analysis(self, actual: str, expected: str) -> float:
        """Evaluate quality of likelihood analysis"""
        return self._evaluate_term_coverage(actual, expected, self.risk_indicators['likelihood_terms'])
    
    def _evaluate_prioritization(self, actual: str, expected: str) -> float:
        """Evaluate quality of risk prioritization"""
        return self._evaluate_term_coverage(actual, expected, self.risk_indicators['priority_terms'])
    
    def _evaluate_mitigation_recommendations(self, actual: str, expected: str) -> float:
        """Evaluate quality of mitigation recommendations"""
        return self._evaluate_term_coverage(actual, expected, self.risk_indicators['mitigation_terms'])
    
    def _evaluate_term_coverage(self, actual: str, expected: str, terms: List[str]) -> float:
        """Evaluate coverage of specific terms"""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        actual_terms = set(term for term in terms if term in actual_lower)
        expected_terms = set(term for term in terms if term in expected_lower)
        
        if not expected_terms:
            return 1.0 if not actual_terms else 0.5
        
        intersection = actual_terms.intersection(expected_terms)
        return len(intersection) / len(expected_terms)


def calculate_security_score(
    actual_response: str,
    expected_response: str,
    query: str = "",
    weights: Optional[Dict[str, float]] = None
) -> SecurityMetricResult:
    """
    Calculate comprehensive security evaluation score.
    
    Args:
        actual_response: Agent's actual response
        expected_response: Expected/reference response
        query: Original user query
        weights: Custom weights for different metrics
        
    Returns:
        SecurityMetricResult with detailed scoring
    """
    default_weights = {
        'accuracy': 0.4,
        'vulnerability_detection': 0.3,
        'risk_assessment': 0.3
    }
    
    if weights:
        default_weights.update(weights)
    
    # Create mock invocations for evaluation
    from google.genai import types as genai_types
    
    user_content = genai_types.Content(parts=[genai_types.Part(text=query)])
    actual_content = genai_types.Content(parts=[genai_types.Part(text=actual_response)])
    expected_content = genai_types.Content(parts=[genai_types.Part(text=expected_response)])
    
    actual_invocation = Invocation(
        user_content=user_content,
        final_response=actual_content,
        invocation_id="eval_actual"
    )
    
    expected_invocation = Invocation(
        user_content=user_content,
        final_response=expected_content,
        invocation_id="eval_expected"
    )
    
    # Calculate individual metric scores
    accuracy_evaluator = SecurityAccuracyMetric()
    accuracy_result = accuracy_evaluator.evaluate_invocations([actual_invocation], [expected_invocation])
    
    vuln_evaluator = VulnerabilityDetectionMetric()
    vuln_result = vuln_evaluator.evaluate_invocations([actual_invocation], [expected_invocation])
    
    risk_evaluator = RiskAssessmentMetric()
    risk_result = risk_evaluator.evaluate_invocations([actual_invocation], [expected_invocation])
    
    # Calculate weighted overall score
    overall_score = (
        accuracy_result.overall_score * default_weights['accuracy'] +
        vuln_result.overall_score * default_weights['vulnerability_detection'] +
        risk_result.overall_score * default_weights['risk_assessment']
    )
    
    return SecurityMetricResult(
        metric_name="security_comprehensive",
        score=overall_score,
        max_score=1.0,
        details={
            'accuracy_score': accuracy_result.overall_score,
            'vulnerability_detection_score': vuln_result.overall_score,
            'risk_assessment_score': risk_result.overall_score,
            'weights': default_weights
        }
    )