"""
Compliance Agent Evaluator

Specialized evaluator for compliance-related agent performance.
Evaluates compliance framework coverage, regulatory requirement analysis, 
and audit readiness assessment capabilities.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from google.adk.evaluation.evaluator import Evaluator, EvaluationResult, EvalStatus, PerInvocationResult
from google.adk.evaluation.eval_case import Invocation
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOC2 = "SOC2"
    PCI_DSS = "PCI_DSS"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    CIS_CONTROLS = "CIS_Controls"
    NIST_CSF = "NIST_CSF"
    ISO27001 = "ISO27001"


@dataclass
class ComplianceRequirement:
    """Represents a compliance requirement"""
    framework: ComplianceFramework
    control_id: str
    title: str
    description: str
    category: str
    mandatory: bool = True
    

class ComplianceEvaluator(Evaluator):
    """
    Compliance-focused evaluator extending ADK Evaluator base class.
    
    Evaluates agent performance on compliance-related tasks including:
    - Framework requirement coverage
    - Control mapping accuracy
    - Gap analysis completeness
    - Remediation recommendations
    """
    
    def __init__(self, threshold: float = 0.85, framework: Optional[ComplianceFramework] = None):
        self.threshold = threshold
        self.target_framework = framework
        
        # Framework-specific keywords and requirements
        self.framework_keywords = {
            ComplianceFramework.SOC2: {
                'keywords': ['soc2', 'soc 2', 'trust services', 'security', 'availability', 'processing integrity', 'confidentiality', 'privacy'],
                'controls': ['cc1', 'cc2', 'cc3', 'cc4', 'cc5', 'cc6', 'cc7', 'cc8', 'cc9'],
                'categories': ['security', 'availability', 'processing_integrity', 'confidentiality', 'privacy']
            },
            ComplianceFramework.PCI_DSS: {
                'keywords': ['pci', 'pci dss', 'payment card', 'cardholder data', 'payment processing'],
                'controls': ['req1', 'req2', 'req3', 'req4', 'req5', 'req6', 'req7', 'req8', 'req9', 'req10', 'req11', 'req12'],
                'categories': ['network_security', 'data_protection', 'vulnerability_management', 'access_control', 'monitoring', 'policy']
            },
            ComplianceFramework.GDPR: {
                'keywords': ['gdpr', 'general data protection regulation', 'personal data', 'data subject', 'consent', 'privacy'],
                'controls': ['art5', 'art6', 'art7', 'art17', 'art20', 'art25', 'art32', 'art33', 'art35'],
                'categories': ['lawfulness', 'consent', 'data_subject_rights', 'privacy_by_design', 'security', 'breach_notification', 'dpia']
            },
            ComplianceFramework.HIPAA: {
                'keywords': ['hipaa', 'phi', 'protected health information', 'healthcare', 'medical records'],
                'controls': ['administrative', 'physical', 'technical'],
                'categories': ['administrative_safeguards', 'physical_safeguards', 'technical_safeguards']
            },
            ComplianceFramework.CIS_CONTROLS: {
                'keywords': ['cis controls', 'cis', 'center for internet security', 'cybersecurity controls'],
                'controls': [f'cis{i}' for i in range(1, 19)],
                'categories': ['basic', 'foundational', 'organizational']
            }
        }
        
        # Common compliance terms across frameworks
        self.common_compliance_terms = [
            'control', 'requirement', 'policy', 'procedure', 'standard', 'guideline',
            'audit', 'assessment', 'compliance', 'regulatory', 'framework', 'governance',
            'risk management', 'security measures', 'documentation', 'evidence',
            'remediation', 'corrective action', 'monitoring', 'review'
        ]
    
    def evaluate_invocations(
        self,
        actual_invocations: List[Invocation],
        expected_invocations: List[Invocation]
    ) -> EvaluationResult:
        """Evaluate compliance agent invocations against expected results"""
        
        logger.info(f"Starting compliance evaluation with {len(actual_invocations)} invocations")
        
        per_invocation_results = []
        total_score = 0.0
        
        for i, (actual, expected) in enumerate(zip(actual_invocations, expected_invocations)):
            result = self._evaluate_single_invocation(actual, expected)
            per_invocation_results.append(result)
            total_score += result.score or 0.0
            
            logger.debug(f"Invocation {i}: score={result.score}, status={result.eval_status}")
        
        overall_score = total_score / len(actual_invocations) if actual_invocations else 0.0
        overall_status = EvalStatus.PASSED if overall_score >= self.threshold else EvalStatus.FAILED
        
        logger.info(f"Compliance evaluation complete: score={overall_score:.3f}, status={overall_status}")
        
        return EvaluationResult(
            overall_score=overall_score,
            overall_eval_status=overall_status,
            per_invocation_results=per_invocation_results
        )
    
    def _evaluate_single_invocation(self, actual: Invocation, expected: Invocation) -> PerInvocationResult:
        """Evaluate a single invocation for compliance coverage"""
        
        try:
            actual_response = self._extract_response_text(actual.final_response)
            expected_response = self._extract_response_text(expected.final_response)
            query = self._extract_response_text(actual.user_content)
            
            # Identify target framework from query and expected response
            detected_framework = self._detect_compliance_framework(query + " " + expected_response)
            
            # Calculate compliance-specific scores
            framework_coverage = self._evaluate_framework_coverage(actual_response, expected_response, detected_framework)
            control_mapping = self._evaluate_control_mapping(actual_response, expected_response, detected_framework)
            requirement_analysis = self._evaluate_requirement_analysis(actual_response, expected_response)
            gap_analysis = self._evaluate_gap_analysis(actual_response, expected_response)
            
            # Weighted combination of scores
            score = (
                framework_coverage * 0.3 +
                control_mapping * 0.25 +
                requirement_analysis * 0.25 +
                gap_analysis * 0.2
            )
            
            status = EvalStatus.PASSED if score >= self.threshold else EvalStatus.FAILED
            
        except Exception as e:
            logger.error(f"Error evaluating compliance invocation: {e}")
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
    
    def _detect_compliance_framework(self, text: str) -> Optional[ComplianceFramework]:
        """Detect which compliance framework is being discussed"""
        text_lower = text.lower()
        
        framework_scores = {}
        for framework, config in self.framework_keywords.items():
            score = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            if score > 0:
                framework_scores[framework] = score
        
        if framework_scores:
            return max(framework_scores, key=framework_scores.get)
        
        return None
    
    def _evaluate_framework_coverage(self, actual: str, expected: str, framework: Optional[ComplianceFramework]) -> float:
        """Evaluate coverage of compliance framework requirements"""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        if not framework:
            # General compliance coverage without specific framework
            actual_terms = sum(1 for term in self.common_compliance_terms if term in actual_lower)
            expected_terms = sum(1 for term in self.common_compliance_terms if term in expected_lower)
            return min(actual_terms / max(expected_terms, 1), 1.0)
        
        # Framework-specific coverage
        config = self.framework_keywords.get(framework, {})
        keywords = config.get('keywords', [])
        
        actual_coverage = sum(1 for keyword in keywords if keyword in actual_lower)
        expected_coverage = sum(1 for keyword in keywords if keyword in expected_lower)
        
        if expected_coverage == 0:
            return 1.0 if actual_coverage == 0 else 0.5
        
        return min(actual_coverage / expected_coverage, 1.0)
    
    def _evaluate_control_mapping(self, actual: str, expected: str, framework: Optional[ComplianceFramework]) -> float:
        """Evaluate accuracy of control identification and mapping"""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        if not framework:
            # Look for general control references
            control_patterns = [r'control\s+\w+', r'requirement\s+\w+', r'section\s+\d+']
            actual_controls = set()
            expected_controls = set()
            
            for pattern in control_patterns:
                actual_controls.update(re.findall(pattern, actual_lower))
                expected_controls.update(re.findall(pattern, expected_lower))
        else:
            # Framework-specific control mapping
            config = self.framework_keywords.get(framework, {})
            controls = config.get('controls', [])
            
            actual_controls = set(control for control in controls if control in actual_lower)
            expected_controls = set(control for control in controls if control in expected_lower)
        
        if not expected_controls:
            return 1.0 if not actual_controls else 0.5
        
        # Calculate precision and recall for control mapping
        intersection = actual_controls.intersection(expected_controls)
        precision = len(intersection) / len(actual_controls) if actual_controls else 0
        recall = len(intersection) / len(expected_controls) if expected_controls else 0
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _evaluate_requirement_analysis(self, actual: str, expected: str) -> float:
        """Evaluate depth and accuracy of requirement analysis"""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Analysis indicators
        analysis_terms = [
            'analysis', 'assessment', 'evaluation', 'review', 'examination',
            'implementation', 'compliance status', 'current state', 'gaps',
            'evidence', 'documentation', 'processes', 'procedures'
        ]
        
        actual_analysis = sum(1 for term in analysis_terms if term in actual_lower)
        expected_analysis = sum(1 for term in analysis_terms if term in expected_lower)
        
        analysis_score = min(actual_analysis / max(expected_analysis, 1), 1.0)
        
        # Structure indicators (shows systematic approach)
        structure_patterns = [
            r'\d+\.\s', r'[a-z]\)\s', r'[-•]\s',  # Numbered/bulleted lists
            r'overview', r'summary', r'findings', r'recommendations'
        ]
        
        actual_structure = sum(1 for pattern in structure_patterns 
                             if len(re.findall(pattern, actual_lower)) > 0)
        expected_structure = sum(1 for pattern in structure_patterns 
                               if len(re.findall(pattern, expected_lower)) > 0)
        
        structure_score = min(actual_structure / max(expected_structure, 1), 1.0)
        
        return (analysis_score * 0.7) + (structure_score * 0.3)
    
    def _evaluate_gap_analysis(self, actual: str, expected: str) -> float:
        """Evaluate quality of gap analysis and remediation recommendations"""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Gap analysis indicators
        gap_terms = [
            'gap', 'deficiency', 'missing', 'lacking', 'insufficient',
            'non-compliant', 'violation', 'weakness', 'shortcoming'
        ]
        
        # Remediation indicators  
        remediation_terms = [
            'recommendation', 'remediation', 'corrective action', 'improvement',
            'implement', 'establish', 'enhance', 'strengthen', 'update',
            'action plan', 'next steps', 'priority', 'timeline'
        ]
        
        actual_gaps = sum(1 for term in gap_terms if term in actual_lower)
        expected_gaps = sum(1 for term in gap_terms if term in expected_lower)
        gap_score = min(actual_gaps / max(expected_gaps, 1), 1.0)
        
        actual_remediation = sum(1 for term in remediation_terms if term in actual_lower)
        expected_remediation = sum(1 for term in remediation_terms if term in expected_lower)
        remediation_score = min(actual_remediation / max(expected_remediation, 1), 1.0)
        
        return (gap_score * 0.4) + (remediation_score * 0.6)


def evaluate_compliance_response(
    query: str,
    actual_response: str,
    expected_response: str,
    threshold: float = 0.85,
    framework: Optional[ComplianceFramework] = None
) -> tuple[float, EvalStatus]:
    """
    Convenience function to evaluate a single compliance response.
    
    Args:
        query: The user query/prompt
        actual_response: Agent's actual response
        expected_response: Expected/reference response
        threshold: Passing threshold (default 0.85)
        framework: Specific compliance framework to focus on
        
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
    
    evaluator = ComplianceEvaluator(threshold=threshold, framework=framework)
    result = evaluator.evaluate_invocations([actual_invocation], [expected_invocation])
    
    return result.overall_score, result.overall_eval_status