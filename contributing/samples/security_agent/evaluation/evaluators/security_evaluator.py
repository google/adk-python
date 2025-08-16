"""
Security Agent Evaluator

Specialized evaluator for security agent performance using ADK evaluation patterns.
Focuses on vulnerability detection, risk assessment, and security response quality.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from google.adk.evaluation.evaluator import Evaluator, EvaluationResult, EvalStatus, PerInvocationResult
from google.adk.evaluation.eval_case import Invocation
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class SecurityMetricType(Enum):
    """Types of security metrics for evaluation"""
    VULNERABILITY_DETECTION = "vulnerability_detection"
    RISK_ASSESSMENT = "risk_assessment" 
    COMPLIANCE_CHECK = "compliance_check"
    INCIDENT_RESPONSE = "incident_response"
    SECURITY_ACCURACY = "security_accuracy"


@dataclass
class SecurityFinding:
    """Represents a security finding for comparison"""
    vulnerability_type: str
    severity: str
    location: str
    description: str
    remediation: Optional[str] = None
    confidence: float = 1.0


@dataclass 
class SecurityEvaluationContext:
    """Context for security evaluation"""
    evaluation_type: SecurityMetricType
    expected_findings: List[SecurityFinding]
    risk_threshold: str = "medium"
    compliance_framework: Optional[str] = None


class SecurityEvaluator(Evaluator):
    """
    Security-focused evaluator extending ADK Evaluator base class.
    
    Evaluates agent performance on security-related tasks including:
    - Vulnerability detection accuracy
    - Risk assessment quality 
    - Security response completeness
    - Compliance coverage
    """
    
    def __init__(self, threshold: float = 0.8, metric_type: SecurityMetricType = SecurityMetricType.SECURITY_ACCURACY):
        self.threshold = threshold
        self.metric_type = metric_type
        self.vulnerability_keywords = {
            'sql_injection': ['sql injection', 'sqli', 'union select', 'drop table'],
            'xss': ['cross-site scripting', 'xss', 'script tag', 'javascript injection'],
            'authentication_bypass': ['auth bypass', 'authentication', 'login bypass'],
            'privilege_escalation': ['privilege escalation', 'escalation', 'elevated privileges'],
            'data_exposure': ['data exposure', 'sensitive data', 'data leak'],
            'configuration_weakness': ['misconfiguration', 'weak configuration', 'default settings']
        }
        
    def evaluate_invocations(
        self,
        actual_invocations: List[Invocation], 
        expected_invocations: List[Invocation]
    ) -> EvaluationResult:
        """
        Evaluate security agent invocations against expected results.
        
        Args:
            actual_invocations: Agent's actual responses
            expected_invocations: Expected/reference responses
            
        Returns:
            EvaluationResult with security-specific scoring
        """
        logger.info(f"Starting security evaluation with {len(actual_invocations)} invocations")
        
        if len(actual_invocations) != len(expected_invocations):
            logger.warning(f"Invocation count mismatch: {len(actual_invocations)} vs {len(expected_invocations)}")
        
        per_invocation_results = []
        total_score = 0.0
        
        for i, (actual, expected) in enumerate(zip(actual_invocations, expected_invocations)):
            result = self._evaluate_single_invocation(actual, expected)
            per_invocation_results.append(result)
            total_score += result.score or 0.0
            
            logger.debug(f"Invocation {i}: score={result.score}, status={result.eval_status}")
        
        # Calculate overall score and status
        overall_score = total_score / len(actual_invocations) if actual_invocations else 0.0
        overall_status = EvalStatus.PASSED if overall_score >= self.threshold else EvalStatus.FAILED
        
        logger.info(f"Security evaluation complete: score={overall_score:.3f}, status={overall_status}")
        
        return EvaluationResult(
            overall_score=overall_score,
            overall_eval_status=overall_status, 
            per_invocation_results=per_invocation_results
        )
    
    def _evaluate_single_invocation(self, actual: Invocation, expected: Invocation) -> PerInvocationResult:
        """Evaluate a single invocation pair for security metrics"""
        
        try:
            actual_response = self._extract_response_text(actual.final_response)
            expected_response = self._extract_response_text(expected.final_response)
            
            # Extract security context from expected response
            security_context = self._extract_security_context(expected.user_content)
            
            # Compute security-specific score based on metric type
            if self.metric_type == SecurityMetricType.VULNERABILITY_DETECTION:
                score = self._evaluate_vulnerability_detection(actual_response, expected_response, security_context)
            elif self.metric_type == SecurityMetricType.RISK_ASSESSMENT: 
                score = self._evaluate_risk_assessment(actual_response, expected_response)
            elif self.metric_type == SecurityMetricType.COMPLIANCE_CHECK:
                score = self._evaluate_compliance_coverage(actual_response, expected_response, security_context)
            else:
                score = self._evaluate_general_security_accuracy(actual_response, expected_response)
            
            status = EvalStatus.PASSED if score >= self.threshold else EvalStatus.FAILED
            
        except Exception as e:
            logger.error(f"Error evaluating invocation: {e}")
            score = 0.0
            status = EvalStatus.FAILED
        
        return PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=score,
            eval_status=status
        )
    
    def _extract_response_text(self, content: Optional[genai_types.Content]) -> str:
        """Extract text from genai Content object"""
        if not content or not content.parts:
            return ""
        
        text_parts = []
        for part in content.parts:
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)
        
        return " ".join(text_parts)
    
    def _extract_security_context(self, user_content: genai_types.Content) -> Dict[str, Any]:
        """Extract security-related context from user query"""
        query_text = self._extract_response_text(user_content).lower()
        
        context = {
            'vulnerability_types': [],
            'compliance_frameworks': [],
            'risk_indicators': []
        }
        
        # Detect vulnerability types mentioned in query
        for vuln_type, keywords in self.vulnerability_keywords.items():
            if any(keyword in query_text for keyword in keywords):
                context['vulnerability_types'].append(vuln_type)
        
        # Detect compliance frameworks
        compliance_keywords = ['soc2', 'pci', 'gdpr', 'hipaa', 'compliance']
        for framework in compliance_keywords:
            if framework in query_text:
                context['compliance_frameworks'].append(framework)
        
        # Detect risk indicators
        risk_keywords = ['critical', 'high risk', 'severe', 'urgent']
        for keyword in risk_keywords:
            if keyword in query_text:
                context['risk_indicators'].append(keyword)
                
        return context
    
    def _evaluate_vulnerability_detection(self, actual: str, expected: str, context: Dict[str, Any]) -> float:
        """Evaluate vulnerability detection accuracy"""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Extract expected vulnerabilities from context
        expected_vulns = set(context.get('vulnerability_types', []))
        
        # Find detected vulnerabilities in actual response
        detected_vulns = set()
        for vuln_type, keywords in self.vulnerability_keywords.items():
            if any(keyword in actual_lower for keyword in keywords):
                detected_vulns.add(vuln_type)
        
        if not expected_vulns:
            # If no specific vulnerabilities expected, check for general security analysis
            security_terms = ['vulnerability', 'security', 'risk', 'exploit', 'attack']
            actual_has_security = any(term in actual_lower for term in security_terms)
            expected_has_security = any(term in expected_lower for term in security_terms)
            return 1.0 if actual_has_security == expected_has_security else 0.5
        
        # Calculate precision and recall
        true_positives = len(expected_vulns.intersection(detected_vulns))
        false_positives = len(detected_vulns - expected_vulns) 
        false_negatives = len(expected_vulns - detected_vulns)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # F1 score as combination of precision and recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return min(f1_score, 1.0)
    
    def _evaluate_risk_assessment(self, actual: str, expected: str) -> float:
        """Evaluate quality of risk assessment"""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Risk level indicators
        risk_levels = ['critical', 'high', 'medium', 'low']
        
        # Extract risk levels from both responses
        actual_risks = [level for level in risk_levels if level in actual_lower]
        expected_risks = [level for level in risk_levels if level in expected_lower]
        
        # Check for risk assessment structure
        risk_structure_terms = ['severity', 'impact', 'likelihood', 'risk score', 'assessment']
        actual_structure = sum(1 for term in risk_structure_terms if term in actual_lower)
        expected_structure = sum(1 for term in risk_structure_terms if term in expected_lower)
        
        # Combine risk level accuracy and structure completeness 
        risk_accuracy = len(set(actual_risks).intersection(set(expected_risks))) / max(len(expected_risks), 1)
        structure_score = min(actual_structure / max(expected_structure, 1), 1.0)
        
        return (risk_accuracy * 0.7) + (structure_score * 0.3)
    
    def _evaluate_compliance_coverage(self, actual: str, expected: str, context: Dict[str, Any]) -> float:
        """Evaluate compliance analysis coverage"""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Expected compliance frameworks from context
        expected_frameworks = context.get('compliance_frameworks', [])
        
        # Compliance-related terms
        compliance_terms = [
            'compliance', 'regulation', 'standard', 'requirement', 'control',
            'audit', 'policy', 'procedure', 'documentation', 'evidence'
        ]
        
        actual_compliance_score = sum(1 for term in compliance_terms if term in actual_lower)
        expected_compliance_score = sum(1 for term in compliance_terms if term in expected_lower)
        
        # Framework-specific coverage
        framework_coverage = 0.0
        if expected_frameworks:
            detected_frameworks = [fw for fw in expected_frameworks if fw in actual_lower]
            framework_coverage = len(detected_frameworks) / len(expected_frameworks)
        
        # Overall compliance coverage score
        term_coverage = min(actual_compliance_score / max(expected_compliance_score, 1), 1.0)
        
        return (term_coverage * 0.6) + (framework_coverage * 0.4)
    
    def _evaluate_general_security_accuracy(self, actual: str, expected: str) -> float:
        """Evaluate general security response accuracy"""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Core security concepts
        security_concepts = [
            'confidentiality', 'integrity', 'availability', 'authentication', 'authorization',
            'encryption', 'access control', 'monitoring', 'logging', 'incident response'
        ]
        
        # Count concept coverage
        actual_concepts = sum(1 for concept in security_concepts if concept in actual_lower)
        expected_concepts = sum(1 for concept in security_concepts if concept in expected_lower)
        
        concept_score = min(actual_concepts / max(expected_concepts, 1), 1.0)
        
        # Basic text similarity (simplified version of rouge-1)
        actual_words = set(actual_lower.split())
        expected_words = set(expected_lower.split())
        
        if not expected_words:
            similarity_score = 1.0 if not actual_words else 0.0
        else:
            overlap = len(actual_words.intersection(expected_words))
            similarity_score = overlap / len(expected_words)
        
        # Combine concept coverage and similarity
        return (concept_score * 0.6) + (similarity_score * 0.4)


# Convenience functions for easy usage
def evaluate_security_response(
    query: str,
    actual_response: str, 
    expected_response: str,
    threshold: float = 0.8,
    metric_type: SecurityMetricType = SecurityMetricType.SECURITY_ACCURACY
) -> Tuple[float, EvalStatus]:
    """
    Convenience function to evaluate a single security response.
    
    Args:
        query: The user query/prompt
        actual_response: Agent's actual response
        expected_response: Expected/reference response
        threshold: Passing threshold (default 0.8)
        metric_type: Type of security metric to evaluate
        
    Returns:
        Tuple of (score, evaluation_status)
    """
    # Create mock invocations
    user_content = genai_types.Content(parts=[genai_types.Part(text=query)])
    actual_content = genai_types.Content(parts=[genai_types.Part(text=actual_response)])
    expected_content = genai_types.Content(parts=[genai_types.Part(text=expected_response)])
    
    actual_invocation = Invocation(
        user_content=user_content,
        final_response=actual_content,
        invocation_id="test_actual"
    )
    
    expected_invocation = Invocation(
        user_content=user_content,
        final_response=expected_content, 
        invocation_id="test_expected"
    )
    
    evaluator = SecurityEvaluator(threshold=threshold, metric_type=metric_type)
    result = evaluator.evaluate_invocations([actual_invocation], [expected_invocation])
    
    return result.overall_score, result.overall_eval_status