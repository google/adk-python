"""
ADK-Compliant Agent Evaluator

This module provides the main evaluation interface following Google ADK patterns.
Supports both test files (.test.json) and evalsets (.evalset.json) formats.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Standard ADK evaluation metrics"""
    TOOL_TRAJECTORY_AVG_SCORE = "tool_trajectory_avg_score"
    RESPONSE_MATCH_SCORE = "response_match_score"
    RESPONSE_EVALUATION_SCORE = "response_evaluation_score"


@dataclass
class EvaluationCriteria:
    """Evaluation criteria configuration"""
    tool_trajectory_avg_score: float = 1.0
    response_match_score: float = 0.8
    response_evaluation_score: float = 0.75
    custom_metrics: Optional[Dict[str, float]] = None


@dataclass
class EvaluationResult:
    """Result of an evaluation run"""
    eval_id: str
    passed: bool
    scores: Dict[str, float]
    details: Dict[str, Any]
    errors: List[str] = None


class ADKEvaluator:
    """
    Main evaluator class following ADK patterns.
    
    Provides methods for evaluating agents against test files and evalsets,
    computing standard metrics, and generating reports.
    """
    
    def __init__(self, criteria: Optional[EvaluationCriteria] = None):
        """
        Initialize evaluator with criteria.
        
        Args:
            criteria: Evaluation criteria with thresholds
        """
        self.criteria = criteria or EvaluationCriteria()
        
    async def evaluate(
        self,
        agent_module: str,
        eval_dataset_file_path_or_dir: str,
        num_runs: int = 1
    ) -> List[EvaluationResult]:
        """
        Evaluate an agent against test files or evalsets.
        
        This is the main entry point following the ADK pattern:
        await AgentEvaluator.evaluate(
            agent_module="my_agent",
            eval_dataset_file_path_or_dir="tests/my_test.test.json"
        )
        
        Args:
            agent_module: Python module path to the agent
            eval_dataset_file_path_or_dir: Path to test file(s) or directory
            num_runs: Number of evaluation runs for statistical significance
            
        Returns:
            List of EvaluationResult objects
        """
        logger.info(f"Starting ADK evaluation for {agent_module}")
        
        # Determine if path is file or directory
        path = Path(eval_dataset_file_path_or_dir)
        
        if path.is_file():
            test_files = [path]
        elif path.is_dir():
            # Find all test files in directory
            test_files = list(path.glob("*.test.json")) + list(path.glob("*.evalset.json"))
        else:
            raise ValueError(f"Path does not exist: {eval_dataset_file_path_or_dir}")
        
        if not test_files:
            raise ValueError(f"No test files found in {eval_dataset_file_path_or_dir}")
        
        logger.info(f"Found {len(test_files)} test files")
        
        # Run evaluations
        results = []
        for test_file in test_files:
            for run_idx in range(num_runs):
                result = await self._evaluate_single_file(
                    agent_module=agent_module,
                    test_file=test_file,
                    run_idx=run_idx
                )
                results.append(result)
        
        # Log summary
        passed_count = sum(1 for r in results if r.passed)
        logger.info(f"Evaluation complete: {passed_count}/{len(results)} passed")
        
        return results
    
    async def _evaluate_single_file(
        self,
        agent_module: str,
        test_file: Path,
        run_idx: int
    ) -> EvaluationResult:
        """Evaluate agent against a single test file"""
        
        logger.debug(f"Evaluating {test_file.name} (run {run_idx + 1})")
        
        # Load test data
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        # Determine file type and evaluate accordingly
        if test_file.suffix == '.test.json' or 'eval_cases' not in test_data:
            # Single test file format
            return await self._evaluate_test_file(agent_module, test_data, test_file.stem)
        else:
            # Evalset format with multiple cases
            return await self._evaluate_evalset(agent_module, test_data, test_file.stem)
    
    async def _evaluate_test_file(
        self,
        agent_module: str,
        test_data: Dict[str, Any],
        test_name: str
    ) -> EvaluationResult:
        """Evaluate a single test file"""
        
        eval_id = f"{test_name}_test"
        scores = {}
        errors = []
        
        try:
            # Extract expected data
            user_content = test_data.get('user_content', {})
            expected_response = test_data.get('final_response', {})
            expected_tools = test_data.get('expected_tool_use', [])
            
            # Run agent (in real implementation, would actually invoke the agent)
            # For now, we'll simulate with expected response
            actual_response = expected_response  # Placeholder
            actual_tools = expected_tools  # Placeholder
            
            # Calculate metrics
            scores['tool_trajectory_avg_score'] = self._calculate_tool_trajectory_score(
                actual_tools, expected_tools
            )
            
            scores['response_match_score'] = self._calculate_response_match_score(
                actual_response, expected_response
            )
            
            # Check against criteria
            passed = all(
                scores.get(metric.value, 0) >= getattr(self.criteria, metric.value)
                for metric in EvaluationMetric
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {test_name}: {e}")
            errors.append(str(e))
            passed = False
        
        return EvaluationResult(
            eval_id=eval_id,
            passed=passed,
            scores=scores,
            details={'test_name': test_name},
            errors=errors
        )
    
    async def _evaluate_evalset(
        self,
        agent_module: str,
        evalset_data: Dict[str, Any],
        evalset_name: str
    ) -> EvaluationResult:
        """Evaluate an evalset with multiple cases"""
        
        eval_id = evalset_data.get('eval_set_id', evalset_name)
        eval_cases = evalset_data.get('eval_cases', [])
        
        total_scores = {metric.value: [] for metric in EvaluationMetric}
        errors = []
        
        for case in eval_cases:
            case_id = case.get('eval_id', 'unknown')
            
            try:
                # Process each conversation in the case
                for conv in case.get('conversation', []):
                    user_content = conv.get('user_content', {})
                    expected_response = conv.get('final_response', {})
                    expected_tools = conv.get('expected_tool_use', [])
                    
                    # Run agent and get actual response (placeholder)
                    actual_response = expected_response
                    actual_tools = expected_tools
                    
                    # Calculate metrics for this conversation
                    tool_score = self._calculate_tool_trajectory_score(
                        actual_tools, expected_tools
                    )
                    response_score = self._calculate_response_match_score(
                        actual_response, expected_response
                    )
                    
                    total_scores['tool_trajectory_avg_score'].append(tool_score)
                    total_scores['response_match_score'].append(response_score)
                    
            except Exception as e:
                logger.error(f"Error evaluating case {case_id}: {e}")
                errors.append(f"Case {case_id}: {str(e)}")
        
        # Calculate average scores
        avg_scores = {}
        for metric, scores_list in total_scores.items():
            if scores_list:
                avg_scores[metric] = sum(scores_list) / len(scores_list)
            else:
                avg_scores[metric] = 0.0
        
        # Check if passed based on criteria
        passed = all(
            avg_scores.get(metric.value, 0) >= getattr(self.criteria, metric.value)
            for metric in EvaluationMetric
            if hasattr(self.criteria, metric.value)
        )
        
        return EvaluationResult(
            eval_id=eval_id,
            passed=passed,
            scores=avg_scores,
            details={
                'evalset_name': evalset_data.get('name', evalset_name),
                'num_cases': len(eval_cases)
            },
            errors=errors if errors else None
        )
    
    def _calculate_tool_trajectory_score(
        self,
        actual_tools: List[Any],
        expected_tools: List[Any]
    ) -> float:
        """
        Calculate tool trajectory score.
        
        Compares actual vs expected tool usage:
        - Matching step = 1 point
        - Mismatched step = 0 points
        - Final score is average of matches
        """
        if not expected_tools:
            return 1.0 if not actual_tools else 0.0
        
        if not actual_tools:
            return 0.0
        
        matches = 0
        total = max(len(actual_tools), len(expected_tools))
        
        for i in range(min(len(actual_tools), len(expected_tools))):
            if self._tools_match(actual_tools[i], expected_tools[i]):
                matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def _calculate_response_match_score(
        self,
        actual_response: Any,
        expected_response: Any
    ) -> float:
        """
        Calculate response match score using ROUGE-like metric.
        
        Default threshold is 0.8 to allow for small variations.
        """
        # Extract text from response objects
        actual_text = self._extract_text(actual_response)
        expected_text = self._extract_text(expected_response)
        
        if not expected_text:
            return 1.0 if not actual_text else 0.0
        
        # Simple word overlap calculation (simplified ROUGE-1)
        actual_words = set(actual_text.lower().split())
        expected_words = set(expected_text.lower().split())
        
        if not expected_words:
            return 1.0
        
        overlap = len(actual_words.intersection(expected_words))
        score = overlap / len(expected_words)
        
        return min(score, 1.0)
    
    def _tools_match(self, actual_tool: Any, expected_tool: Any) -> bool:
        """Check if two tool invocations match"""
        # Simplified comparison - in real implementation would be more sophisticated
        if isinstance(actual_tool, dict) and isinstance(expected_tool, dict):
            return (
                actual_tool.get('tool_name') == expected_tool.get('tool_name') and
                actual_tool.get('action') == expected_tool.get('action')
            )
        return actual_tool == expected_tool
    
    def _extract_text(self, response: Any) -> str:
        """Extract text content from response object"""
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            # Handle nested response structure
            if 'parts' in response:
                parts = response['parts']
                if isinstance(parts, list):
                    texts = []
                    for part in parts:
                        if isinstance(part, dict) and 'text' in part:
                            texts.append(part['text'])
                    return ' '.join(texts)
            elif 'text' in response:
                return response['text']
        return ""


# Convenience function for quick evaluation
async def evaluate_agent(
    agent_module: str,
    test_file: str,
    criteria: Optional[Dict[str, float]] = None
) -> bool:
    """
    Convenience function to quickly evaluate an agent.
    
    Args:
        agent_module: Python module path to agent
        test_file: Path to test file
        criteria: Optional criteria overrides
        
    Returns:
        True if all tests passed, False otherwise
    """
    eval_criteria = EvaluationCriteria()
    if criteria:
        for key, value in criteria.items():
            if hasattr(eval_criteria, key):
                setattr(eval_criteria, key, value)
    
    evaluator = ADKEvaluator(eval_criteria)
    results = await evaluator.evaluate(agent_module, test_file)
    
    return all(r.passed for r in results)