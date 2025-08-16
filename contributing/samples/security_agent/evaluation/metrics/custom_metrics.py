"""
Custom evaluation metrics

Additional metrics for evaluating agent performance beyond standard ADK metrics.
Includes compliance coverage, tool efficiency, and response completeness metrics.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

from google.adk.evaluation.evaluator import Evaluator, EvaluationResult, EvalStatus, PerInvocationResult
from google.adk.evaluation.eval_case import Invocation

logger = logging.getLogger(__name__)


class CustomMetricEvaluator(Evaluator):
    """Base class for custom metric evaluators"""
    
    def __init__(self, threshold: float = 0.8, metric_name: str = "custom_metric"):
        self.threshold = threshold
        self.metric_name = metric_name
    
    def evaluate_invocations(
        self,
        actual_invocations: List[Invocation],
        expected_invocations: List[Invocation]
    ) -> EvaluationResult:
        """Base implementation for custom metrics"""
        
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
        """Override this method in subclasses"""
        return PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=1.0,
            eval_status=EvalStatus.PASSED
        )


class ComplianceCoverageMetric(CustomMetricEvaluator):
    """
    Evaluates coverage of compliance requirements and frameworks.
    Measures how well the agent addresses compliance-related queries.
    """
    
    def __init__(self, threshold: float = 0.9):
        super().__init__(threshold, "compliance_coverage")
        
        self.compliance_frameworks = [
            'soc2', 'pci', 'gdpr', 'hipaa', 'iso27001', 'nist', 'cis'
        ]
        
        self.compliance_terms = [
            'compliance', 'audit', 'control', 'requirement', 'policy',
            'procedure', 'documentation', 'evidence', 'assessment',
            'governance', 'risk management', 'monitoring'
        ]
    
    def _evaluate_single_invocation(self, actual: Invocation, expected: Invocation) -> PerInvocationResult:
        """Evaluate compliance coverage for a single invocation"""
        
        actual_text = self._extract_text(actual.final_response).lower()
        expected_text = self._extract_text(expected.final_response).lower()
        
        # Framework coverage
        expected_frameworks = set(fw for fw in self.compliance_frameworks if fw in expected_text)
        actual_frameworks = set(fw for fw in self.compliance_frameworks if fw in actual_text)
        
        framework_coverage = (len(actual_frameworks.intersection(expected_frameworks)) / 
                            len(expected_frameworks)) if expected_frameworks else 1.0
        
        # Terms coverage
        expected_terms = set(term for term in self.compliance_terms if term in expected_text)
        actual_terms = set(term for term in self.compliance_terms if term in actual_text)
        
        terms_coverage = (len(actual_terms.intersection(expected_terms)) / 
                         len(expected_terms)) if expected_terms else 1.0
        
        # Combined score
        score = (framework_coverage * 0.6) + (terms_coverage * 0.4)
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


class ToolEfficiencyMetric(CustomMetricEvaluator):
    """
    Evaluates efficiency of tool usage by the agent.
    Measures appropriate tool selection and execution patterns.
    """
    
    def __init__(self, threshold: float = 0.8):
        super().__init__(threshold, "tool_efficiency")
    
    def _evaluate_single_invocation(self, actual: Invocation, expected: Invocation) -> PerInvocationResult:
        """Evaluate tool usage efficiency"""
        
        actual_tools = self._extract_tool_usage(actual)
        expected_tools = self._extract_tool_usage(expected)
        
        if not expected_tools and not actual_tools:
            # No tools expected or used - perfect efficiency
            score = 1.0
        elif not expected_tools:
            # No tools expected but some used - less efficient
            score = 0.7
        elif not actual_tools:
            # Tools expected but none used - poor efficiency
            score = 0.2
        else:
            # Compare tool usage patterns
            tool_accuracy = len(set(actual_tools).intersection(set(expected_tools))) / len(expected_tools)
            tool_efficiency = min(len(expected_tools) / len(actual_tools), 1.0) if actual_tools else 0
            score = (tool_accuracy * 0.7) + (tool_efficiency * 0.3)
        
        status = EvalStatus.PASSED if score >= self.threshold else EvalStatus.FAILED
        
        return PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=score,
            eval_status=status
        )
    
    def _extract_tool_usage(self, invocation: Invocation) -> List[str]:
        """Extract tool usage from invocation"""
        tools = []
        
        if invocation.intermediate_data and invocation.intermediate_data.tool_uses:
            for tool_use in invocation.intermediate_data.tool_uses:
                if hasattr(tool_use, 'name'):
                    tools.append(tool_use.name)
        
        return tools


class ResponseCompletenessMetric(CustomMetricEvaluator):
    """
    Evaluates completeness of agent responses.
    Measures coverage of key topics and thoroughness of analysis.
    """
    
    def __init__(self, threshold: float = 0.8):
        super().__init__(threshold, "response_completeness")
    
    def _evaluate_single_invocation(self, actual: Invocation, expected: Invocation) -> PerInvocationResult:
        """Evaluate response completeness"""
        
        actual_text = self._extract_text(actual.final_response)
        expected_text = self._extract_text(expected.final_response)
        
        # Extract key topics from expected response
        expected_topics = self._extract_topics(expected_text)
        actual_topics = self._extract_topics(actual_text)
        
        # Calculate topic coverage
        if not expected_topics:
            topic_coverage = 1.0
        else:
            covered_topics = set(actual_topics).intersection(set(expected_topics))
            topic_coverage = len(covered_topics) / len(expected_topics)
        
        # Calculate response depth (approximate by length and structure)
        expected_depth = self._calculate_response_depth(expected_text)
        actual_depth = self._calculate_response_depth(actual_text)
        
        depth_ratio = min(actual_depth / expected_depth, 1.0) if expected_depth > 0 else 1.0
        
        # Combined completeness score
        score = (topic_coverage * 0.7) + (depth_ratio * 0.3)
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
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text"""
        # Simple topic extraction based on common patterns
        topics = []
        
        # Headers and bullet points
        header_pattern = r'^#+\s+(.+)$'
        bullet_pattern = r'^[-*]\s+(.+)$'
        
        for line in text.split('\n'):
            line = line.strip()
            
            # Check for headers
            header_match = re.match(header_pattern, line)
            if header_match:
                topics.append(header_match.group(1).lower())
            
            # Check for bullet points
            bullet_match = re.match(bullet_pattern, line)
            if bullet_match:
                topics.append(bullet_match.group(1).lower())
        
        return list(set(topics))  # Remove duplicates
    
    def _calculate_response_depth(self, text: str) -> float:
        """Calculate approximate response depth"""
        if not text:
            return 0.0
        
        # Simple metrics for response depth
        line_count = len([line for line in text.split('\n') if line.strip()])
        word_count = len(text.split())
        structure_elements = len(re.findall(r'[#*-]\s+', text))
        
        # Weighted combination
        depth_score = (line_count * 0.3) + (word_count * 0.01) + (structure_elements * 0.5)
        
        return depth_score