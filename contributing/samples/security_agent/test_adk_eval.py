#!/usr/bin/env python3
"""
Test ADK evaluation with the security agent
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_adk_evaluation():
    """Test basic ADK evaluation functionality"""
    
    try:
        from google.adk.evaluation.agent_evaluator import AgentEvaluator
        
        # Test with the existing evaluation dataset
        eval_dataset = "agents/gcp_security/vertex_sqlite/Evaluation.evalset.json"
        agent_module = "agents.gcp_security.vertex_sqlite_agent"
        
        logger.info(f"Testing evaluation with dataset: {eval_dataset}")
        logger.info(f"Agent module: {agent_module}")
        
        # Check if files exist
        if not Path(eval_dataset).exists():
            logger.error(f"Evaluation dataset not found: {eval_dataset}")
            return False
            
        # Run evaluation
        results = await AgentEvaluator.evaluate(
            agent_module=agent_module,
            eval_dataset_file_path_or_dir=eval_dataset,
            num_runs=1  # Start with 1 run for testing
        )
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results type: {type(results)}")
        
        # Try to extract some basic info
        if hasattr(results, 'overall_score'):
            logger.info(f"Overall score: {results.overall_score}")
        if hasattr(results, 'overall_eval_status'):
            logger.info(f"Overall status: {results.overall_eval_status}")
            
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import ADK evaluation: {e}")
        logger.info("Make sure ADK[eval] is installed: pip install google-adk[eval]")
        return False
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        return False

async def main():
    """Main test function"""
    logger.info("Testing ADK evaluation setup...")
    
    success = await test_adk_evaluation()
    
    if success:
        logger.info("✅ ADK evaluation test passed!")
        logger.info("\nNext steps:")
        logger.info("1. Run: python run_evaluation.py --evaluation-type security")
        logger.info("2. Check results in evaluation/results/")
        logger.info("3. Add more test cases to evaluation/datasets/")
    else:
        logger.error("❌ ADK evaluation test failed!")
        logger.info("\nTroubleshooting:")
        logger.info("1. Install: pip install google-adk[eval]")
        logger.info("2. Check agent module path")
        logger.info("3. Verify evaluation dataset format")

if __name__ == "__main__":
    asyncio.run(main())