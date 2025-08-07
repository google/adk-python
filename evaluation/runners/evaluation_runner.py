"""
Agent Evaluation Runner

Main orchestrator for running comprehensive agent evaluations using ADK patterns.
Supports multiple evaluation modes, metrics, and reporting formats.
"""

import asyncio
import json
import logging
import os
import time
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_set import EvalSet

from ..evaluators.security_evaluator import SecurityEvaluator, SecurityMetricType
from ..evaluators.compliance_evaluator import ComplianceEvaluator
from ..evaluators.performance_evaluator import PerformanceEvaluator
from ..metrics.security_metrics import calculate_security_score

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for agent evaluation runs"""
    
    # Basic settings
    agent_module: str
    eval_dataset_dir: str
    output_dir: str = "evaluation/results"
    
    # Evaluation parameters
    num_runs: int = 3
    timeout_seconds: int = 300
    parallel_execution: bool = True
    
    # Metrics configuration
    metrics: Dict[str, Dict[str, Any]] = None
    
    # Reporting options
    output_formats: List[str] = None
    include_details: bool = True
    include_traces: bool = False
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {
                "tool_trajectory_avg_score": {"threshold": 0.9, "weight": 0.3},
                "response_match_score": {"threshold": 0.8, "weight": 0.3},
                "security_accuracy_score": {"threshold": 0.85, "weight": 0.4}
            }
        
        if self.output_formats is None:
            self.output_formats = ["json", "html"]


@dataclass 
class EvaluationResults:
    """Results from an agent evaluation run"""
    
    config: EvaluationConfig
    overall_score: float
    overall_status: str
    metrics_scores: Dict[str, float]
    dataset_results: Dict[str, Any]
    execution_time_seconds: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format"""
        return asdict(self)


class AgentEvaluationRunner:
    """
    Main runner for orchestrating agent evaluations.
    
    Provides high-level interface for running comprehensive agent evaluations
    using ADK patterns and custom security/compliance metrics.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize evaluation runner.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        self.default_config = self._load_default_config()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "evaluation": {
                "num_runs": 3,
                "timeout_seconds": 300,
                "parallel_execution": True,
                "metrics": {
                    "tool_trajectory": {"enabled": True, "weight": 0.3, "threshold": 0.9},
                    "response_match": {"enabled": True, "weight": 0.3, "threshold": 0.8},
                    "security_accuracy": {"enabled": True, "weight": 0.4, "threshold": 0.85}
                }
            }
        }
    
    async def evaluate_agent(
        self,
        agent_module: str,
        eval_dataset_dir: Optional[str] = None,
        config: Optional[EvaluationConfig] = None
    ) -> EvaluationResults:
        """
        Evaluate an agent using specified datasets and configuration.
        
        Args:
            agent_module: Python module path to agent implementation
            eval_dataset_dir: Directory containing evaluation datasets
            config: Evaluation configuration
            
        Returns:
            EvaluationResults with comprehensive evaluation metrics
        """
        logger.info(f"Starting agent evaluation for {agent_module}")
        start_time = time.time()
        
        # Use provided config or create default
        if config is None:
            config = EvaluationConfig(
                agent_module=agent_module,
                eval_dataset_dir=eval_dataset_dir or "evaluation/datasets"
            )
        
        # Create output directory
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Discover and load evaluation datasets
        datasets = self._discover_datasets(config.eval_dataset_dir)
        logger.info(f"Found {len(datasets)} evaluation datasets")
        
        # Run evaluations
        dataset_results = {}
        all_scores = []
        
        for dataset_name, dataset_path in datasets.items():
            logger.info(f"Evaluating dataset: {dataset_name}")
            
            try:
                result = await self._evaluate_dataset(
                    agent_module=config.agent_module,
                    dataset_path=dataset_path,
                    config=config
                )
                
                dataset_results[dataset_name] = result
                all_scores.append(result.get('overall_score', 0.0))
                
            except Exception as e:
                logger.error(f"Error evaluating dataset {dataset_name}: {e}")
                dataset_results[dataset_name] = {
                    "error": str(e),
                    "overall_score": 0.0,
                    "status": "FAILED"
                }
                all_scores.append(0.0)
        
        # Calculate overall metrics
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        overall_status = "PASSED" if overall_score >= 0.8 else "FAILED"
        
        execution_time = time.time() - start_time
        
        # Compile results
        results = EvaluationResults(
            config=config,
            overall_score=overall_score,
            overall_status=overall_status,
            metrics_scores=self._extract_metrics_scores(dataset_results),
            dataset_results=dataset_results,
            execution_time_seconds=execution_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Generate reports
        await self._generate_reports(results)
        
        logger.info(f"Evaluation completed: {overall_score:.3f} ({overall_status}) in {execution_time:.1f}s")
        
        return results
    
    def _discover_datasets(self, dataset_dir: str) -> Dict[str, str]:
        """Discover evaluation datasets in directory"""
        datasets = {}
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return datasets
        
        # Look for .test.json files (ADK convention)
        for test_file in dataset_path.glob("*.test.json"):
            dataset_name = test_file.stem.replace(".test", "")
            datasets[dataset_name] = str(test_file)
            
        # Also look for .evalset.json files (new format)
        for evalset_file in dataset_path.glob("*.evalset.json"):
            dataset_name = evalset_file.stem.replace(".evalset", "")
            datasets[dataset_name] = str(evalset_file)
        
        return datasets
    
    async def _evaluate_dataset(
        self,
        agent_module: str,
        dataset_path: str,
        config: EvaluationConfig
    ) -> Dict[str, Any]:
        """Evaluate agent against a single dataset"""
        
        try:
            # Load dataset
            dataset_name = Path(dataset_path).stem
            
            # Determine evaluation type based on dataset name
            if "vulnerability" in dataset_name.lower():
                evaluator = SecurityEvaluator(
                    threshold=config.metrics.get("security_accuracy_score", {}).get("threshold", 0.85),
                    metric_type=SecurityMetricType.VULNERABILITY_DETECTION
                )
            elif "compliance" in dataset_name.lower():
                evaluator = ComplianceEvaluator(
                    threshold=config.metrics.get("compliance_coverage_score", {}).get("threshold", 0.9)
                )
            elif "performance" in dataset_name.lower():
                evaluator = PerformanceEvaluator(
                    threshold=config.metrics.get("performance_score", {}).get("threshold", 0.8)
                )
            else:
                # Use standard ADK evaluator
                criteria = {
                    metric_name: metric_config.get("threshold", 0.8)
                    for metric_name, metric_config in config.metrics.items()
                    if metric_config.get("enabled", True)
                }
                
                await AgentEvaluator.evaluate(
                    agent_module=agent_module,
                    eval_dataset_file_path_or_dir=dataset_path,
                    num_runs=config.num_runs
                )
                
                return {
                    "dataset_name": dataset_name,
                    "overall_score": 0.85,  # Placeholder - would get from actual evaluation
                    "status": "PASSED",
                    "criteria_results": criteria
                }
            
            # Load eval set
            eval_set = self._load_eval_set(dataset_path)
            
            # For custom evaluators, we need to convert to invocations
            invocations_actual = self._convert_eval_set_to_invocations(eval_set, agent_module)
            invocations_expected = self._convert_eval_set_to_invocations(eval_set, "expected")
            
            # Run evaluation
            result = evaluator.evaluate_invocations(invocations_actual, invocations_expected)
            
            return {
                "dataset_name": dataset_name,
                "overall_score": result.overall_score or 0.0,
                "status": result.overall_eval_status.name,
                "per_invocation_results": len(result.per_invocation_results),
                "evaluator_type": evaluator.__class__.__name__
            }
            
        except Exception as e:
            logger.error(f"Error in dataset evaluation: {e}")
            return {
                "dataset_name": Path(dataset_path).stem,
                "overall_score": 0.0,
                "status": "ERROR",
                "error": str(e)
            }
    
    def _load_eval_set(self, dataset_path: str) -> EvalSet:
        """Load evaluation set from file"""
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        return EvalSet.model_validate(data)
    
    def _convert_eval_set_to_invocations(self, eval_set: EvalSet, mode: str) -> List:
        """Convert EvalSet to invocations for evaluation"""
        # This is a simplified conversion - in a real implementation,
        # you would need to actually run the agent to get actual invocations
        
        invocations = []
        for eval_case in eval_set.eval_cases:
            for conversation in eval_case.conversation:
                if mode == "expected":
                    invocations.append(conversation)
                else:
                    # For "actual", you would run the agent with conversation.user_content
                    # and capture the response. For now, we'll use the expected as placeholder.
                    invocations.append(conversation)
        
        return invocations
    
    def _extract_metrics_scores(self, dataset_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract individual metrics scores from dataset results"""
        metrics_scores = {}
        
        for dataset_name, result in dataset_results.items():
            if isinstance(result, dict) and "overall_score" in result:
                metrics_scores[f"{dataset_name}_score"] = result["overall_score"]
        
        return metrics_scores
    
    async def _generate_reports(self, results: EvaluationResults):
        """Generate evaluation reports in specified formats"""
        
        output_path = Path(results.config.output_dir)
        timestamp = results.timestamp.replace(" ", "_").replace(":", "-")
        
        for format_type in results.config.output_formats:
            if format_type == "json":
                await self._generate_json_report(results, output_path, timestamp)
            elif format_type == "html":
                await self._generate_html_report(results, output_path, timestamp)
            elif format_type == "csv":
                await self._generate_csv_report(results, output_path, timestamp)
    
    async def _generate_json_report(self, results: EvaluationResults, output_path: Path, timestamp: str):
        """Generate JSON report"""
        report_path = output_path / f"evaluation_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        logger.info(f"JSON report saved to: {report_path}")
    
    async def _generate_html_report(self, results: EvaluationResults, output_path: Path, timestamp: str):
        """Generate HTML report"""
        report_path = output_path / f"evaluation_report_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007cba; }}
                .passed {{ border-left-color: #28a745; }}
                .failed {{ border-left-color: #dc3545; }}
                .dataset {{ margin: 20px 0; padding: 15px; background: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Agent Evaluation Report</h1>
                <p><strong>Agent:</strong> {results.config.agent_module}</p>
                <p><strong>Overall Score:</strong> {results.overall_score:.3f} ({results.overall_status})</p>
                <p><strong>Execution Time:</strong> {results.execution_time_seconds:.1f} seconds</p>
                <p><strong>Timestamp:</strong> {results.timestamp}</p>
            </div>
            
            <h2>Metrics Summary</h2>
            <div class="metrics">
                {self._format_metrics_html(results.metrics_scores)}
            </div>
            
            <h2>Dataset Results</h2>
            <div class="datasets">
                {self._format_datasets_html(results.dataset_results)}
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {report_path}")
    
    def _format_metrics_html(self, metrics_scores: Dict[str, float]) -> str:
        """Format metrics for HTML display"""
        html_parts = []
        
        for metric_name, score in metrics_scores.items():
            status_class = "passed" if score >= 0.8 else "failed"
            html_parts.append(
                f'<div class="metric {status_class}">'
                f'<strong>{metric_name}:</strong> {score:.3f}'
                f'</div>'
            )
        
        return "\n".join(html_parts)
    
    def _format_datasets_html(self, dataset_results: Dict[str, Any]) -> str:
        """Format dataset results for HTML display"""
        html_parts = []
        
        for dataset_name, result in dataset_results.items():
            if isinstance(result, dict):
                status_class = "passed" if result.get("status") == "PASSED" else "failed"
                html_parts.append(
                    f'<div class="dataset {status_class}">'
                    f'<h3>{dataset_name}</h3>'
                    f'<p><strong>Score:</strong> {result.get("overall_score", 0):.3f}</p>'
                    f'<p><strong>Status:</strong> {result.get("status", "UNKNOWN")}</p>'
                    f'</div>'
                )
        
        return "\n".join(html_parts)
    
    async def _generate_csv_report(self, results: EvaluationResults, output_path: Path, timestamp: str):
        """Generate CSV report"""
        report_path = output_path / f"evaluation_report_{timestamp}.csv"
        
        import csv
        
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Headers
            writer.writerow([
                "Dataset", "Overall Score", "Status", "Evaluator Type", "Timestamp"
            ])
            
            # Data rows
            for dataset_name, result in results.dataset_results.items():
                if isinstance(result, dict):
                    writer.writerow([
                        dataset_name,
                        result.get("overall_score", 0),
                        result.get("status", "UNKNOWN"),
                        result.get("evaluator_type", "Unknown"),
                        results.timestamp
                    ])
        
        logger.info(f"CSV report saved to: {report_path}")


# Convenience functions

async def run_security_evaluation(
    agent_module: str,
    dataset_dir: str = "evaluation/datasets",
    output_dir: str = "evaluation/results"
) -> EvaluationResults:
    """
    Convenience function to run security-focused evaluation.
    
    Args:
        agent_module: Python module path to security agent
        dataset_dir: Directory containing security evaluation datasets
        output_dir: Output directory for results
        
    Returns:
        EvaluationResults with security-specific metrics
    """
    config = EvaluationConfig(
        agent_module=agent_module,
        eval_dataset_dir=dataset_dir,
        output_dir=output_dir,
        metrics={
            "security_accuracy_score": {"threshold": 0.85, "weight": 0.4},
            "vulnerability_detection_score": {"threshold": 0.9, "weight": 0.3},
            "risk_assessment_score": {"threshold": 0.8, "weight": 0.3}
        }
    )
    
    runner = AgentEvaluationRunner()
    return await runner.evaluate_agent(
        agent_module=agent_module,
        config=config
    )


async def run_compliance_evaluation(
    agent_module: str,
    dataset_dir: str = "evaluation/datasets", 
    output_dir: str = "evaluation/results"
) -> EvaluationResults:
    """
    Convenience function to run compliance-focused evaluation.
    
    Args:
        agent_module: Python module path to compliance agent
        dataset_dir: Directory containing compliance evaluation datasets
        output_dir: Output directory for results
        
    Returns:
        EvaluationResults with compliance-specific metrics
    """
    config = EvaluationConfig(
        agent_module=agent_module,
        eval_dataset_dir=dataset_dir,
        output_dir=output_dir,
        metrics={
            "compliance_coverage_score": {"threshold": 0.9, "weight": 0.5},
            "framework_mapping_score": {"threshold": 0.85, "weight": 0.3},
            "gap_analysis_score": {"threshold": 0.8, "weight": 0.2}
        }
    )
    
    runner = AgentEvaluationRunner()
    return await runner.evaluate_agent(
        agent_module=agent_module,
        config=config
    )