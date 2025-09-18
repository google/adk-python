"""
Response Quality Assessment Framework for LLM Analysis Detection

This module provides comprehensive tools to distinguish between raw JSON data responses
and true LLM-generated analysis responses from the ADK security agent.

Created for Task 1: Build framework to measure analysis depth and detect LLM reasoning.
"""

import json
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ResponseType(Enum):
    """Types of responses the agent can produce"""
    RAW_DATA = "raw_data"
    LLM_ANALYSIS = "llm_analysis"
    MIXED = "mixed"
    UNCLEAR = "unclear"


@dataclass
class QualityMetrics:
    """Quality metrics for response analysis"""
    analysis_depth_score: float  # 0-100 scale
    reasoning_indicators: int
    recommendation_count: int
    insight_density: float  # insights per 100 words
    raw_data_ratio: float  # 0-1 scale
    response_type: ResponseType
    confidence: float  # 0-1 scale


class ResponseQualityAssessor:
    """
    Comprehensive framework for assessing response quality and detecting LLM analysis
    """

    def __init__(self):
        # Analysis keywords that indicate LLM reasoning
        self.analysis_keywords = {
            'high_value': ['recommend', 'prioritize', 'assess', 'conclude', 'analyze', 'evaluate'],
            'medium_value': ['consider', 'suggest', 'indicate', 'implies', 'therefore', 'however'],
            'reasoning': ['because', 'since', 'due to', 'as a result', 'consequently', 'thus'],
            'comparative': ['compared to', 'versus', 'in contrast', 'relative to', 'better than'],
            'security_analysis': ['vulnerability', 'risk level', 'threat', 'remediation', 'mitigation']
        }

        # Raw data patterns that indicate unprocessed tool responses
        self.raw_data_patterns = [
            r'\{"success":\s*true',
            r'\{"data":\s*\[',
            r'\{"row_count":\s*\d+',
            r'"id":\s*\d+,\s*"name":',
            r'\[{"id":\s*\d+',
            r'"created_at":\s*"\d{4}-\d{2}-\d{2}',
            r'"asset_type":\s*"[^"]+",',
            r'"location":\s*"[^"]+",\s*"state":'
        ]

        # Template indicators (signs of formatted but not analyzed responses)
        self.template_patterns = [
            r'Here are the .+ findings:',
            r'Found \d+ .+ in your',
            r'Your .+ resources:',
            r'The following .+ were found:',
            r'Summary of .+ buckets:'
        ]

    def assess_response_quality(self, response: str) -> QualityMetrics:
        """
        Comprehensive assessment of response quality

        Args:
            response: The agent's response text

        Returns:
            QualityMetrics object with detailed analysis
        """
        # Basic metrics
        word_count = len(response.split())

        # Detect reasoning indicators
        reasoning_count = self._count_reasoning_indicators(response)

        # Count recommendations
        recommendation_count = self._count_recommendations(response)

        # Detect raw data patterns
        raw_data_score = self._detect_raw_data_patterns(response)

        # Calculate insight density
        insight_density = self._calculate_insight_density(response, word_count)

        # Determine response type
        response_type, confidence = self._classify_response_type(
            response, reasoning_count, raw_data_score, insight_density
        )

        # Calculate overall analysis depth score
        analysis_depth_score = self._calculate_analysis_depth_score(
            reasoning_count, recommendation_count, insight_density, raw_data_score
        )

        return QualityMetrics(
            analysis_depth_score=analysis_depth_score,
            reasoning_indicators=reasoning_count,
            recommendation_count=recommendation_count,
            insight_density=insight_density,
            raw_data_ratio=raw_data_score,
            response_type=response_type,
            confidence=confidence
        )

    def _count_reasoning_indicators(self, response: str) -> int:
        """Count indicators of LLM reasoning in the response"""
        response_lower = response.lower()
        total_count = 0

        for category, keywords in self.analysis_keywords.items():
            for keyword in keywords:
                # Count occurrences with word boundaries to avoid false positives
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, response_lower))
                total_count += matches

        return total_count

    def _count_recommendations(self, response: str) -> int:
        """Count explicit recommendations in the response"""
        recommendation_patterns = [
            r'recommend[a-z]*',
            r'should [a-z]+ to',
            r'you should',
            r'consider [a-z]+ing',
            r'action[s]? to take',
            r'next step[s]?',
            r'suggestion[s]?'
        ]

        response_lower = response.lower()
        total_recommendations = 0

        for pattern in recommendation_patterns:
            matches = len(re.findall(pattern, response_lower))
            total_recommendations += matches

        return total_recommendations

    def _detect_raw_data_patterns(self, response: str) -> float:
        """
        Detect raw JSON data patterns and return ratio (0-1)
        Higher values indicate more raw data content
        """
        total_matches = 0

        for pattern in self.raw_data_patterns:
            matches = len(re.findall(pattern, response))
            total_matches += matches

        # Normalize based on response length and typical data structure density
        response_length = len(response)
        if response_length == 0:
            return 0.0

        # Calculate raw data ratio (0-1 scale)
        raw_data_ratio = min(1.0, total_matches / max(1, response_length / 100))

        return raw_data_ratio

    def _calculate_insight_density(self, response: str, word_count: int) -> float:
        """Calculate insights per 100 words"""
        if word_count == 0:
            return 0.0

        # Look for insight indicators
        insight_patterns = [
            r'this indicates',
            r'this suggests',
            r'this means',
            r'the risk is',
            r'priority should be',
            r'critical because',
            r'important to note',
            r'key finding',
            r'main concern'
        ]

        response_lower = response.lower()
        insight_count = 0

        for pattern in insight_patterns:
            matches = len(re.findall(pattern, response_lower))
            insight_count += matches

        # Calculate insights per 100 words
        insight_density = (insight_count / word_count) * 100

        return insight_density

    def _classify_response_type(self, response: str, reasoning_count: int,
                              raw_data_score: float, insight_density: float) -> Tuple[ResponseType, float]:
        """
        Classify the response type and return confidence level

        Returns:
            Tuple of (ResponseType, confidence_score)
        """
        # Check if it's primarily raw JSON data
        if raw_data_score > 0.2 and reasoning_count < 2 and insight_density < 0.5:
            confidence = min(0.95, 0.7 + raw_data_score * 0.5)
            return ResponseType.RAW_DATA, confidence

        # Check if it's true LLM analysis (more lenient thresholds)
        if reasoning_count >= 2 and insight_density >= 0.5 and raw_data_score < 0.1:
            confidence = min(0.95, 0.6 + (reasoning_count / 15) + (insight_density / 5))
            return ResponseType.LLM_ANALYSIS, confidence

        # Check for mixed responses (has some analysis but also some raw data)
        if raw_data_score > 0.05 and reasoning_count >= 1:
            confidence = 0.7
            return ResponseType.MIXED, confidence

        # If we have reasoning indicators but low insight density
        if reasoning_count >= 1 and insight_density >= 0.2:
            confidence = 0.6
            return ResponseType.LLM_ANALYSIS, confidence

        # Default to unclear if we can't classify confidently
        confidence = 0.5
        return ResponseType.UNCLEAR, confidence

    def _calculate_analysis_depth_score(self, reasoning_count: int, recommendation_count: int,
                                      insight_density: float, raw_data_score: float) -> float:
        """
        Calculate overall analysis depth score (0-100)

        Scoring components:
        - Reasoning indicators: 30% weight (more lenient)
        - Recommendations: 30% weight
        - Insight density: 30% weight (more lenient)
        - Raw data penalty: -10% weight
        """
        # Reasoning score (0-30 points) - more lenient scoring
        reasoning_score = min(30, reasoning_count * 8)

        # Recommendation score (0-30 points) - more lenient scoring
        recommendation_score = min(30, recommendation_count * 8)

        # Insight density score (0-30 points) - more lenient scoring
        insight_score = min(30, insight_density * 10)

        # Raw data penalty (0-10 points deducted)
        raw_data_penalty = raw_data_score * 15

        # Base score for having any analysis at all
        base_score = 10 if reasoning_count > 0 or recommendation_count > 0 else 0

        # Calculate final score
        total_score = base_score + reasoning_score + recommendation_score + insight_score - raw_data_penalty

        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, total_score))

    def is_llm_analysis(self, response: str, threshold: float = 60.0) -> bool:
        """
        Simple boolean check if response contains LLM analysis

        Args:
            response: Response text to analyze
            threshold: Minimum analysis depth score to consider as LLM analysis

        Returns:
            True if response contains sufficient LLM analysis
        """
        metrics = self.assess_response_quality(response)
        return (metrics.analysis_depth_score >= threshold and
                metrics.response_type in [ResponseType.LLM_ANALYSIS, ResponseType.MIXED])

    def generate_quality_report(self, response: str) -> str:
        """Generate a human-readable quality assessment report"""
        metrics = self.assess_response_quality(response)

        report = f"""
Response Quality Assessment Report
================================

Analysis Depth Score: {metrics.analysis_depth_score:.1f}/100
Response Type: {metrics.response_type.value}
Confidence: {metrics.confidence:.2f}

Detailed Metrics:
- Reasoning Indicators: {metrics.reasoning_indicators}
- Recommendations: {metrics.recommendation_count}
- Insight Density: {metrics.insight_density:.2f} insights per 100 words
- Raw Data Ratio: {metrics.raw_data_ratio:.2f}

Assessment: {'PASS' if metrics.analysis_depth_score >= 60 else 'FAIL'}
(Threshold: 60+ for LLM analysis)
"""
        return report


# Test cases for validation
def test_raw_json_response():
    """Test detection of raw JSON responses"""
    assessor = ResponseQualityAssessor()

    raw_response = '{"success": true, "data": [{"id": 1, "name": "bucket1", "location": "us-central1"}], "row_count": 14}'

    metrics = assessor.assess_response_quality(raw_response)

    assert metrics.response_type == ResponseType.RAW_DATA
    assert metrics.analysis_depth_score < 30
    assert not assessor.is_llm_analysis(raw_response)


def test_llm_analysis_response():
    """Test detection of LLM analysis responses"""
    assessor = ResponseQualityAssessor()

    analysis_response = """
    Based on the analysis of your 14 storage buckets, I recommend prioritizing the following security concerns:

    1. **Critical Risk**: Three buckets lack encryption at rest, which means sensitive data could be compromised.
       You should immediately enable encryption for buckets containing PII or financial data.

    2. **High Priority**: Five buckets have public read access, which indicates potential data exposure risk.
       Consider restricting access to authenticated users only.

    The main concern is that your most critical data assets are in the least secure buckets.
    This suggests a need for a comprehensive data classification and security policy review.
    """

    metrics = assessor.assess_response_quality(analysis_response)

    assert metrics.response_type == ResponseType.LLM_ANALYSIS
    assert metrics.analysis_depth_score >= 60
    assert assessor.is_llm_analysis(analysis_response)


if __name__ == "__main__":
    # Run basic tests
    test_raw_json_response()
    test_llm_analysis_response()
    print("✅ All Response Quality Assessment tests passed!")