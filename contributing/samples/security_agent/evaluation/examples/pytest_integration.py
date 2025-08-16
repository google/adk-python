#!/usr/bin/env python3
"""
Example: Pytest Integration for ADK Evaluation

Shows how to integrate ADK evaluation with pytest for automated testing.
This follows the patterns recommended in Google ADK documentation.
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from adk_evaluator import ADKEvaluator, EvaluationCriteria, evaluate_agent


class TestSecurityAgent:
    """Test suite for security agent using ADK patterns"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator with standard criteria"""
        return ADKEvaluator(EvaluationCriteria(
            tool_trajectory_avg_score=0.9,
            response_match_score=0.8
        ))
    
    @pytest.mark.asyncio
    async def test_vulnerability_detection(self, evaluator):
        """Test agent's vulnerability detection capabilities"""
        results = await evaluator.evaluate(
            agent_module="security_agent",
            eval_dataset_file_path_or_dir="datasets/vulnerability_assessment.test.json"
        )
        
        assert len(results) > 0, "No evaluation results returned"
        assert all(r.passed for r in results), "Some tests failed"
        
        # Check specific metrics
        for result in results:
            assert result.scores.get('tool_trajectory_avg_score', 0) >= 0.9
            assert result.scores.get('response_match_score', 0) >= 0.8
    
    @pytest.mark.asyncio
    async def test_compliance_checking(self, evaluator):
        """Test agent's compliance checking abilities"""
        results = await evaluator.evaluate(
            agent_module="security_agent",
            eval_dataset_file_path_or_dir="datasets/compliance_check.test.json"
        )
        
        assert len(results) > 0
        
        # Allow some flexibility in compliance tests
        passed_count = sum(1 for r in results if r.passed)
        assert passed_count / len(results) >= 0.8, "Pass rate below 80%"
    
    @pytest.mark.asyncio
    async def test_incident_response(self):
        """Test incident response capabilities with custom criteria"""
        # More lenient criteria for complex incident scenarios
        criteria = EvaluationCriteria(
            tool_trajectory_avg_score=0.85,
            response_match_score=0.75
        )
        
        evaluator = ADKEvaluator(criteria)
        results = await evaluator.evaluate(
            agent_module="security_agent",
            eval_dataset_file_path_or_dir="datasets/incident_response.test.json"
        )
        
        # Check that agent handles incidents appropriately
        for result in results:
            assert result.scores.get('response_match_score', 0) >= 0.75
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("test_file", [
        "vulnerability_assessment.test.json",
        "compliance_check.test.json",
        "incident_response.test.json"
    ])
    async def test_individual_datasets(self, test_file):
        """Parameterized test for each dataset"""
        passed = await evaluate_agent(
            agent_module="security_agent",
            test_file=f"datasets/{test_file}"
        )
        
        assert passed, f"Evaluation failed for {test_file}"
    
    @pytest.mark.asyncio
    async def test_comprehensive_evaluation(self):
        """Run comprehensive evaluation across all datasets"""
        evaluator = ADKEvaluator()
        
        results = await evaluator.evaluate(
            agent_module="security_agent",
            eval_dataset_file_path_or_dir="datasets/",
            num_runs=3  # Multiple runs for statistical significance
        )
        
        # Calculate statistics
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        
        # Require 85% pass rate for comprehensive test
        pass_rate = passed / total if total > 0 else 0
        assert pass_rate >= 0.85, f"Pass rate {pass_rate:.1%} below required 85%"
        
        # Check average scores
        avg_tool_score = sum(
            r.scores.get('tool_trajectory_avg_score', 0) for r in results
        ) / total
        
        avg_response_score = sum(
            r.scores.get('response_match_score', 0) for r in results
        ) / total
        
        assert avg_tool_score >= 0.85, f"Avg tool score {avg_tool_score:.2f} too low"
        assert avg_response_score >= 0.75, f"Avg response score {avg_response_score:.2f} too low"


class TestEvaluationFramework:
    """Test the evaluation framework itself"""
    
    @pytest.mark.asyncio
    async def test_criteria_enforcement(self):
        """Test that criteria are properly enforced"""
        strict_criteria = EvaluationCriteria(
            tool_trajectory_avg_score=1.0,  # Require perfect tool usage
            response_match_score=1.0         # Require exact match
        )
        
        evaluator = ADKEvaluator(strict_criteria)
        
        # This should likely fail with strict criteria
        results = await evaluator.evaluate(
            agent_module="security_agent",
            eval_dataset_file_path_or_dir="datasets/vulnerability_assessment.test.json"
        )
        
        # Verify criteria are being checked
        for result in results:
            if not result.passed:
                # At least one score should be below threshold
                assert (
                    result.scores.get('tool_trajectory_avg_score', 0) < 1.0 or
                    result.scores.get('response_match_score', 0) < 1.0
                )
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling for invalid inputs"""
        evaluator = ADKEvaluator()
        
        # Test with non-existent file
        with pytest.raises(ValueError):
            await evaluator.evaluate(
                agent_module="security_agent",
                eval_dataset_file_path_or_dir="non_existent.test.json"
            )
        
        # Test with invalid agent module
        with pytest.raises(Exception):
            await evaluator.evaluate(
                agent_module="invalid_agent_module",
                eval_dataset_file_path_or_dir="datasets/vulnerability_assessment.test.json"
            )


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for evaluation"""
    
    @pytest.mark.asyncio
    async def test_evaluation_speed(self, benchmark):
        """Benchmark evaluation speed"""
        evaluator = ADKEvaluator()
        
        async def run_evaluation():
            return await evaluator.evaluate(
                agent_module="security_agent",
                eval_dataset_file_path_or_dir="datasets/vulnerability_assessment.test.json"
            )
        
        # Run benchmark
        results = benchmark(lambda: asyncio.run(run_evaluation()))
        
        assert len(results) > 0
        assert benchmark.stats['mean'] < 5.0, "Evaluation taking too long"


# Fixtures for shared test data
@pytest.fixture(scope="module")
def test_datasets_dir():
    """Path to test datasets directory"""
    return Path(__file__).parent.parent / "datasets"


@pytest.fixture(scope="module")
def sample_test_file(test_datasets_dir):
    """Path to a sample test file"""
    return test_datasets_dir / "vulnerability_assessment.test.json"


# Helper functions for test utilities
def assert_metric_above_threshold(results, metric_name, threshold):
    """Assert that a metric is above threshold for all results"""
    for result in results:
        score = result.scores.get(metric_name, 0)
        assert score >= threshold, (
            f"{metric_name} score {score:.2f} below threshold {threshold}"
        )


def calculate_aggregate_scores(results):
    """Calculate aggregate scores across all results"""
    if not results:
        return {}
    
    metrics = {}
    for result in results:
        for metric, score in result.scores.items():
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(score)
    
    return {
        metric: sum(scores) / len(scores)
        for metric, scores in metrics.items()
    }


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])