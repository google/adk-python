"""
Demonstration script for the AVG Financial Expert agent.

This script initializes the FinancialAdvisorMasterAgent, loads configurations,
and processes sample queries from a text file to showcase the agent's
capabilities in selecting and executing financial expertise pipelines.
"""

import asyncio
import os
import json
import logging
from typing import Dict, Any, Callable # BaseExpertWrapper needs to be imported for type hinting
from functools import partial

# Attempt to load environment variables
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not found. .env file will not be loaded. pip install python-dotenv")
    load_dotenv = None # type: ignore

# AVG Toolkit and Financial Agent Components
# Assuming this script is run from the repository root (e.g., using `python -m avg.demo.run_demo`)
# or the `avg` directory is in PYTHONPATH.
try:
    from ..adk_avg_toolkit import (
        KnowledgeConfigLoader,
        BaseExpertWrapper,
        ADKLlmExpertWrapper, # For the factory
        ExpertisePipelineManagerError,
        ConfigLoadError
    )
    from ..financial_expert_agent import FinancialAdvisorMasterAgent
    from ..financial_expert_agent.expert_wrappers import (
        CompanyDataRetriever,
        MarketNewsAnalyzer,
        ReportGenerator,
        ProfileRetriever # Added ProfileRetriever
    )
except ImportError as e:
    print(f"ImportError: {e}. Ensure that the script is run as part of the 'avg' package, "
          "e.g., `python -m avg.demo.run_demo` from the repository root, "
          "or that 'avg' parent directory is in PYTHONPATH.")
    # As a fallback for simpler direct execution for very basic tests,
    # try a different relative path strategy IF this script is in avg/demo/
    # This is not robust and module execution is preferred.
    if __package__ is None or __package__ == '': # Direct execution
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Add repo root to path
        from avg.adk_avg_toolkit import (
            KnowledgeConfigLoader, BaseExpertWrapper, ADKLlmExpertWrapper,
            ExpertisePipelineManagerError, ConfigLoadError
        )
        from avg.financial_expert_agent import FinancialAdvisorMasterAgent
        from avg.financial_expert_agent.expert_wrappers import (
            CompanyDataRetriever, MarketNewsAnalyzer, ReportGenerator, ProfileRetriever
        )
    else: # Re-raise if it's meant to be a package and still fails
        raise

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_expert_wrapper_factory(
    expert_id: str, # As per ExpertisePipelineManager's factory signature
    expert_step_config: Dict[str, Any], # Config from the pipeline step for this expert
    knowledge_loader: KnowledgeConfigLoader # Passed via functools.partial
) -> BaseExpertWrapper:
    """
    Factory function to create expert wrapper instances based on configuration.

    Args:
        expert_id: The unique identifier for the expert instance (from pipeline step).
        expert_step_config: Configuration dictionary from the pipeline step.
                            This should contain 'expert_type' and 'init_params'
                            which are considered the 'avg_expert_config' for the wrapper.
        knowledge_loader: Instance of KnowledgeConfigLoader (available for richer expert init if needed).

    Returns:
        An instance of a BaseExpertWrapper subclass.
    """
    expert_type = expert_step_config.get("expert_type")
    # The 'init_params' from the YAML step configuration maps to 'avg_expert_config'
    # that BaseExpertWrapper and its children expect.
    avg_expert_config_for_wrapper = expert_step_config.get("init_params", {})

    logger.info(f"Factory: Creating expert '{expert_id}' of type '{expert_type}' "
                f"with avg_config: {json.dumps(avg_expert_config_for_wrapper, indent=2)}")

    if expert_type == "CompanyDataRetriever":
        return CompanyDataRetriever(expert_id=expert_id, avg_expert_config=avg_expert_config_for_wrapper)
    elif expert_type == "MarketNewsAnalyzer":
        return MarketNewsAnalyzer(expert_id=expert_id, avg_expert_config=avg_expert_config_for_wrapper)
    elif expert_type == "ReportGenerator":
        return ReportGenerator(expert_id=expert_id, avg_expert_config=avg_expert_config_for_wrapper)
    elif expert_type == "ADKLlmExpertWrapper": # For generic LLM tasks defined in pipeline
        return ADKLlmExpertWrapper(expert_id=expert_id, avg_expert_config=avg_expert_config_for_wrapper)
    elif expert_type == "ProfileRetriever": # Ensure this uses the actual ProfileRetriever class
        logger.info(f"Factory: Creating ProfileRetriever for expert_id '{expert_id}'.")
        return ProfileRetriever(expert_id=expert_id, avg_expert_config=avg_expert_config_for_wrapper)
    elif expert_type == "DecisionModelExecutor":
        logger.warning(f"Factory: Hypothetical expert type '{expert_type}' requested. Using ADKLlmExpertWrapper as fallback.")
        return ADKLlmExpertWrapper(expert_id=expert_id, avg_expert_config=avg_expert_config_for_wrapper)
    else:
        error_msg = f"Factory: Unknown expert_type '{expert_type}' requested for expert_id '{expert_id}'."
        logger.error(error_msg)
        raise ValueError(error_msg)


async def main():
    """
    Main asynchronous function to run the AVG Financial Expert demo.
    """
    logger.info("--- Starting AVG Financial Expert Demo ---")

    # Load environment variables from .env file (if it exists)
    # Looks for .env in the directory *above* avg/ (i.e. repo root)
    if load_dotenv:
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        example_dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env.example')
        if os.path.exists(dotenv_path):
            logger.info(f"Loading environment variables from: {dotenv_path}")
            load_dotenv(dotenv_path=dotenv_path)
        elif os.path.exists(example_dotenv_path):
            logger.info(f"Loading environment variables from example: {example_dotenv_path}")
            load_dotenv(dotenv_path=example_dotenv_path) # Load example for defaults if .env missing
        else:
            logger.warning(".env file or .env.example not found in repository root. API keys might be missing.")


    # Define paths relative to this script's location (avg/demo/run_demo.py)
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "..", "financial_expert_agent", "configs", "financial_knowledge_config.yaml")
    queries_file_path = os.path.join(script_dir, "sample_queries.txt")

    if not os.path.exists(config_path):
        logger.error(f"FATAL: Knowledge configuration file not found: {config_path}")
        return
    if not os.path.exists(queries_file_path):
        logger.error(f"FATAL: Sample queries file not found: {queries_file_path}")
        # Create a dummy queries file for basic run
        with open(queries_file_path, 'w') as f:
            f.write("# Sample Queries (fallback)\nWhat is the company overview for MSFT?\n")
        logger.info(f"Created a dummy sample_queries.txt at {queries_file_path}")


    try:
        knowledge_loader = KnowledgeConfigLoader(config_path)
        logger.info("KnowledgeConfigLoader initialized.")
    except (ConfigLoadError, FileNotFoundError) as e:
        logger.error(f"Failed to load knowledge configuration: {e}")
        return

    pipeline_definitions = knowledge_loader.get_config_section("expertise_pipelines")
    if not pipeline_definitions:
        logger.error("No 'expertise_pipelines' found in the knowledge configuration. Cannot proceed.")
        return
    logger.info(f"Loaded {len(pipeline_definitions)} pipeline definitions.")

    # Create the factory with knowledge_loader partially applied
    # The factory signature for ExpertisePipelineManager is (expert_id: str, expert_step_config: Dict)
    # Our demo_expert_wrapper_factory also needs knowledge_loader, so we use partial.
    # Note: The current wrappers (CompanyDataRetriever etc.) don't use knowledge_loader in their constructor,
    # but a real factory might need it to configure experts more deeply (e.g. resolving tool configs).
    bound_expert_factory = partial(demo_expert_wrapper_factory, knowledge_loader=knowledge_loader)


    try:
        agent = FinancialAdvisorMasterAgent(
            knowledge_config_path=config_path, # Passed for agent's own KCL instance
            avg_pipeline_configs=pipeline_definitions,
            expert_factory=bound_expert_factory
        )
        logger.info("FinancialAdvisorMasterAgent initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize FinancialAdvisorMasterAgent: {e}", exc_info=True)
        return

    queries = []
    try:
        with open(queries_file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        logger.info(f"Loaded {len(queries)} sample queries from: {queries_file_path}")
    except FileNotFoundError: # Should be caught above, but as safety
        logger.error(f"Sample queries file not found at: {queries_file_path}")
        return

    for query in queries:
        logger.info(f"\n--- Processing Query: {query} ---")
        try:
            response = await agent.handle_user_request(query)
            # Pretty print JSON response
            response_str = json.dumps(response, indent=2, ensure_ascii=False)
            logger.info(f"Agent Response for '{query}':\n{response_str}")
        except ExpertisePipelineManagerError as e_pipe:
            logger.error(f"Error executing pipeline for query '{query}': {e_pipe}", exc_info=True)
        except Exception as e_gen: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred while processing query '{query}': {e_gen}", exc_info=True)

    logger.info("\n--- AVG Financial Expert Demo Finished ---")


if __name__ == "__main__":
    # This allows the script to be run directly, e.g., `python avg/demo/run_demo.py`
    # For proper package resolution with relative imports, it's often better to run as a module:
    # `python -m avg.demo.run_demo` from the repository root.
    # The try-except for imports at the top attempts to handle both.
    asyncio.run(main())
