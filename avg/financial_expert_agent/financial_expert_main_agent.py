"""
Main logic for the Financial Advisor Master Agent.
This agent uses AVG components (KnowledgeConfigLoader, ExpertisePipelineManager)
to understand user queries and orchestrate financial expertise pipelines.
"""
import asyncio
import os
from typing import Dict, Any, List, Callable, Optional

# AVG Toolkit Components - using relative imports
try:
    from ..adk_avg_toolkit import (
        KnowledgeConfigLoader,
        ExpertisePipelineManager,
        BaseExpertWrapper,
        ADKLlmExpertWrapper, # For the factory in __main__
        ExpertisePipelineManagerError,
        ConfigLoadError
    )
except ImportError:
    # Fallback for cases where the script might be run directly for testing (not recommended for pkg structure)
    # This assumes adk_avg_toolkit is in python path or at same level for such direct test.
    print("Warning: Relative import failed. Attempting direct import for adk_avg_toolkit components (might fail if not in PYTHONPATH).")
    from adk_avg_toolkit import (
        KnowledgeConfigLoader,
        ExpertisePipelineManager,
        BaseExpertWrapper,
        ADKLlmExpertWrapper,
        ExpertisePipelineManagerError,
        ConfigLoadError
    )


class FinancialAdvisorMasterAgent:
    """
    The FinancialAdvisorMasterAgent orchestrates financial analysis tasks
    by selecting and executing relevant expertise pipelines.
    """

    def __init__(
        self,
        knowledge_config_path: str,
        avg_pipeline_configs: List[Dict[str, Any]],
        expert_factory: Callable[[str, Dict[str, Any]], BaseExpertWrapper]
    ):
        """
        Initializes the FinancialAdvisorMasterAgent.

        Args:
            knowledge_config_path: Path to the knowledge configuration YAML/JSON file.
            avg_pipeline_configs: A list of pipeline definition dictionaries.
            expert_factory: A callable that instantiates expert wrappers.
                           It takes (expert_id: str, expert_step_config: Dict)
                           and returns a BaseExpertWrapper instance.
        """
        print(f"FinancialAdvisorMasterAgent: Initializing with knowledge config: {knowledge_config_path}")
        try:
            self.knowledge_loader = KnowledgeConfigLoader(knowledge_config_path)
        except ConfigLoadError as e:
            print(f"Error loading knowledge configuration: {e}")
            raise
        except FileNotFoundError as e:
            print(f"Knowledge configuration file not found: {e}")
            raise

        self.pipeline_manager = ExpertisePipelineManager(
            pipeline_configs=avg_pipeline_configs,
            knowledge_loader=self.knowledge_loader,
            expert_wrapper_factory=expert_factory
        )
        self.avg_pipeline_configs = avg_pipeline_configs # Store for reference if needed

        # Conceptual: An LLM agent could be used here for more advanced query understanding
        # and pipeline selection. For now, we use simple rule-based selection.
        # self.main_llm_agent_for_understanding = expert_factory(
        #     "master_understanding_llm",
        #     {"avg_expert_config": {"init_params": {"model_name": "gemini-pro-for-understanding"}}}
        # )
        print("FinancialAdvisorMasterAgent: Initialized successfully.")


    def _select_pipeline_for_query(self, query: str) -> Optional[str]:
        """
        Selects an appropriate pipeline name based on the user query (simplified logic).

        Args:
            query: The user's query string.

        Returns:
            The name of the selected pipeline, or None if no specific match.
        """
        query_lower = query.lower()
        selected_pipeline: Optional[str] = None

        if "overview" in query_lower and ("company" in query_lower or "stock" in query_lower):
            selected_pipeline = "company_overview_pipeline"
        elif "sentiment" in query_lower and "market" in query_lower:
            selected_pipeline = "market_sentiment_pipeline"
        elif "report" in query_lower and "investment" in query_lower:
            selected_pipeline = "investment_suitability_report_pipeline" # Example for future

        print(f"FinancialAdvisorMasterAgent: For query '{query}', selected pipeline: {selected_pipeline}")
        return selected_pipeline

    async def handle_user_request(self, user_query: str) -> Any:
        """
        Handles a user request by selecting and executing the appropriate pipeline.

        Args:
            user_query: The user's query string.

        Returns:
            The result from the executed pipeline, or an error/default message.
        """
        print(f"\nFinancialAdvisorMasterAgent: Handling user request: '{user_query}'")
        pipeline_name = self._select_pipeline_for_query(user_query)

        if pipeline_name is None:
            return {"status": "clarification_needed", "message": "Sorry, I'm not sure how to handle that specific request. Could you please rephrase or provide more details?"}

        # Prepare initial input for the pipeline.
        # Pipelines might expect specific keys, e.g., 'query', 'user_profile', etc.
        initial_input = {
            "user_query": user_query,
            "query": user_query,
        }

        # Demo-specific hack: Add dummy data if this specific pipeline is selected
        if pipeline_name == "investment_suitability_report_pipeline":
            # Using print for demo visibility; logger would be for production
            print("FinancialAdvisorMasterAgent: Augmenting initial_input with dummy client_id and product_ticker for demo pipeline.")
            initial_input["client_id"] = "DEMO_CLIENT_789"
            initial_input["product_ticker"] = "DEMO_XYZ_TECH"
            # In a real scenario, these would be parsed from the user_query or other context.

        try:
            pipeline_result = await self.pipeline_manager.execute_pipeline(
                pipeline_name,
                initial_input
            )
            return pipeline_result
        except ExpertisePipelineManagerError as e:
            print(f"FinancialAdvisorMasterAgent: Error executing pipeline '{pipeline_name}': {e}")
            return {"status": "error", "message": f"An error occurred while processing your request: {e}"}
        except Exception as e:
            print(f"FinancialAdvisorMasterAgent: Unexpected error for query '{user_query}': {e}")
            # Log the full error for debugging (import traceback if needed)
            return {"status": "error", "message": "An unexpected internal error occurred."}

    # --- Conceptual ADK Integration Comments ---
    # This FinancialAdvisorMasterAgent class could be integrated with Google ADK
    # in several ways:
    #
    # 1. As a System in `adk.System`:
    #    The `handle_user_request` method could be the entry point for the system.
    #    An ADK Runner would invoke this method.
    #    ```python
    #    # Example:
    #    # from google.adk import runtime
    #    #
    #    # def main():
    #    #     # Setup configs, factory
    #    #     agent = FinancialAdvisorMasterAgent(...)
    #    #     sys_obj = adk.System(agent.handle_user_request)
    #    #     runtime.serve(sys_obj) # If using ADK serving
    #    #
    #    # if __name__ == '__main__':
    #    #     main()
    #    ```
    #
    # 2. Wrapped by an ADK Agent (e.g., LlmAgent or WorkflowAgent):
    #    An ADK LlmAgent could use this master agent as a "tool" or delegate
    #    complex reasoning to it.
    #    A WorkflowAgent could call `handle_user_request` as one of its steps.
    #
    # 3. Individual Pipelines as ADK Agents/Tools:
    #    Alternatively, each pipeline managed by ExpertisePipelineManager could
    #    itself be exposed as a separate ADK Agent or Tool, and another
    #    top-level ADK agent could decide which tool/pipeline-agent to call.
    #    The FinancialAdvisorMasterAgent's selection logic would then reside
    #    in that top-level ADK agent.
    #
    # The choice depends on the desired level of granularity and control within
    # the ADK framework. The current AVG structure aims to provide flexibility.


if __name__ == '__main__':

    # --- Dummy Expert Wrapper Factory for __main__ testing ---
    def main_dummy_expert_wrapper_factory(expert_id: str, expert_step_config: Dict[str, Any]) -> BaseExpertWrapper:
        print(f"MainFactory: Creating/getting Expert ID '{expert_id}' "
              f"with step_config: {expert_step_config}")
        base_wrapper_config = expert_step_config.get("avg_expert_config", {})
        # Example: If expert_id implies a type, could return different dummies.
        # For now, all are ADKLlmExpertWrapper.
        return ADKLlmExpertWrapper(
            expert_id=expert_id,
            avg_expert_config=base_wrapper_config
        )

    # --- Sample AVG Pipeline Configurations for __main__ testing ---
    # These define the structure of pipelines that the FinancialAdvisorMasterAgent can choose from.
    main_avg_pipeline_configs = [
        {
            "name": "company_overview_pipeline",
            "description": "Provides an overview of a specified company.",
            "steps": [
                {
                    "expert_id": "company_data_retriever_expert", # Needs to be created by factory
                    "expert_config": { # Config for this step, passed to factory
                        "avg_expert_config": { # Config for the BaseExpertWrapper
                           "init_params": {"model_name": "company-data-model-v1"},
                           "capabilities": {"can_fetch_company_data": True}
                        }
                    },
                    # Output of this step becomes the input for the next, or final output
                },
                # Could add more steps, e.g., summarization expert
            ]
        },
        {
            "name": "market_sentiment_pipeline",
            "description": "Analyzes market sentiment for a given topic or sector.",
            "steps": [
                {
                    "expert_id": "market_news_analyzer_expert",
                    "expert_config": {
                        "avg_expert_config": {
                           "init_params": {"model_name": "sentiment-analysis-model-v3"},
                           "capabilities": {"can_analyze_sentiment": True}
                        }
                    },
                    # Input: query (e.g., "semiconductor industry")
                    # Output: sentiment analysis report
                }
            ]
        }
    ]

    # --- Main Test Loop ---
    async def main_test_loop():
        print("--- Starting FinancialAdvisorMasterAgent Test Loop ---")

        # Define path to the dummy knowledge config created earlier
        # Ensure CWD is the repository root when running this.
        # For `python -m ...` execution, paths relative to repo root are fine.
        dummy_config_file_path = os.path.join(
            "avg", "financial_expert_agent", "configs", "dummy_financial_expert_config.yaml"
        )

        if not os.path.exists(dummy_config_file_path):
            print(f"FATAL: Dummy config file for testing not found at {dummy_config_file_path}")
            print("Please ensure it was created in the previous step or create a placeholder.")
            # As a fallback, create a minimal one if missing, so __main__ can run
            # This is mostly for robustness if steps are run non-sequentially outside the agent
            os.makedirs(os.path.dirname(dummy_config_file_path), exist_ok=True)
            with open(dummy_config_file_path, 'w') as f:
                f.write("version: '1.0'\nminimal_for_test: true\n")
            print(f"Created minimal dummy config: {dummy_config_file_path}")


        try:
            master_agent = FinancialAdvisorMasterAgent(
                knowledge_config_path=dummy_config_file_path,
                avg_pipeline_configs=main_avg_pipeline_configs,
                expert_factory=main_dummy_expert_wrapper_factory
            )
        except Exception as e:
            print(f"Failed to initialize FinancialAdvisorMasterAgent: {e}")
            print("Ensure all AVG toolkit components (KnowledgeConfigLoader, etc.) are correctly implemented.")
            return # Cannot proceed with tests

        test_queries = [
            "Can I get a company overview for MSFT?",
            "What's the market sentiment for renewable energy?",
            "What are today's winning lottery numbers?" # Should not match a pipeline
        ]

        for query in test_queries:
            result = await master_agent.handle_user_request(query)
            print(f"Response for query '{query}':\n{result}\n" + "-"*30)

        print("--- FinancialAdvisorMasterAgent Test Loop Finished ---")

    # Run the async main test loop
    # This structure (`if __name__ == '__main__': asyncio.run(...)`) is standard.
    asyncio.run(main_test_loop())
