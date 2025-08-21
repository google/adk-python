#!/usr/bin/env python3
"""
Security Agent Evaluation Runner

Run comprehensive evaluations of the security agent using ADK evaluation framework.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add the agents directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "agents" / "gcp_security"))

from evaluation.runners.evaluation_runner import (
    AgentEvaluationRunner,
    EvaluationConfig,
    run_security_evaluation
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main evaluation runner"""
    parser = argparse.ArgumentParser(description="Run security agent evaluations")
    parser.add_argument(
        "--agent-module", 
        default="vertex_sqlite_agent",
        help="Agent module to evaluate"
    )
    parser.add_argument(
        "--dataset-dir",
        default="evaluation/datasets", 
        help="Directory containing evaluation datasets"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config-file",
        default="evaluation/config/evaluation_config.yaml",
        help="Evaluation configuration file"
    )
    parser.add_argument(
        "--evaluation-type",
        choices=["security", "compliance", "full"],
        default="full",
        help="Type of evaluation to run"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting {args.evaluation_type} evaluation for {args.agent_module}")
    
    try:
        if args.evaluation_type == "security":
            results = await run_security_evaluation(
                agent_module=args.agent_module,
                dataset_dir=args.dataset_dir,
                output_dir=args.output_dir
            )
        else:
            # Create custom config
            config = EvaluationConfig(
                agent_module=args.agent_module,
                eval_dataset_dir=args.dataset_dir,
                output_dir=args.output_dir,
                num_runs=3,
                timeout_seconds=300,
                metrics={
                    "tool_trajectory_avg_score": {"threshold": 0.9, "weight": 0.2},
                    "response_match_score": {"threshold": 0.8, "weight": 0.2},
                    "security_accuracy_score": {"threshold": 0.85, "weight": 0.3},
                    "vulnerability_detection_score": {"threshold": 0.9, "weight": 0.3}
                },
                output_formats=["json", "html", "csv"]
            )
            
            runner = AgentEvaluationRunner(args.config_file)
            results = await runner.evaluate_agent(
                agent_module=args.agent_module,
                config=config
            )
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Agent: {results.config.agent_module}")
        logger.info(f"Overall Score: {results.overall_score:.3f}")
        logger.info(f"Status: {results.overall_status}")
        logger.info(f"Execution Time: {results.execution_time_seconds:.1f}s")
        logger.info(f"Results saved to: {results.config.output_dir}")
        
        # Print per-dataset results
        logger.info("\nDataset Results:")
        for dataset_name, result in results.dataset_results.items():
            if isinstance(result, dict):
                score = result.get("overall_score", 0)
                status = result.get("status", "UNKNOWN")
                logger.info(f"  {dataset_name}: {score:.3f} ({status})")
        
        # Exit with appropriate code
        sys.exit(0 if results.overall_status == "PASSED" else 1)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())