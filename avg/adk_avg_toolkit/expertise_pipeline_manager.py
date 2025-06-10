"""
Manages the definition, loading, and execution of expertise pipelines.
Orchestrates BaseExpertWrapper instances to perform complex tasks.
"""
import asyncio
from typing import Dict, Any, List, Callable, Optional

from .knowledge_config_loader import KnowledgeConfigLoader # Assuming relative import
from .base_expert_wrapper import BaseExpertWrapper, ADKLlmExpertWrapper # For main block

class ExpertisePipelineManagerError(Exception):
    """Custom exception for errors within the ExpertisePipelineManager."""
    pass

class ExpertisePipelineManager:
    """
    Manages and executes predefined sequences of expert tasks (pipelines).
    """

    def __init__(
        self,
        pipeline_configs: List[Dict[str, Any]],
        knowledge_loader: KnowledgeConfigLoader, # Can be used to enrich expert_config if needed
        expert_wrapper_factory: Callable[[str, Dict[str, Any]], BaseExpertWrapper]
    ):
        """
        Initializes the ExpertisePipelineManager.

        Args:
            pipeline_configs: A list of pipeline configuration dictionaries.
                              Each dict should have a 'name' key.
            knowledge_loader: An instance of KnowledgeConfigLoader.
            expert_wrapper_factory: A callable that takes an expert_id (str) and
                                     an expert_config (Dict) and returns an
                                     instance of BaseExpertWrapper.
        """
        self.pipeline_configs_map: Dict[str, Dict[str, Any]] = {}
        for pc in pipeline_configs:
            if 'name' not in pc:
                raise ExpertisePipelineManagerError("Pipeline configuration missing 'name' key.")
            self.pipeline_configs_map[pc['name']] = pc

        self.knowledge_loader = knowledge_loader # Stored for potential future use
        self.expert_wrapper_factory = expert_wrapper_factory
        self.experts_cache: Dict[str, BaseExpertWrapper] = {}

    def _get_pipeline_config(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Retrieves a pipeline's configuration by its name.

        Args:
            pipeline_name: The name of the pipeline.

        Returns:
            The configuration dictionary for the pipeline.

        Raises:
            ExpertisePipelineManagerError: If the pipeline name is not found.
        """
        config = self.pipeline_configs_map.get(pipeline_name)
        if not config:
            raise ExpertisePipelineManagerError(f"Pipeline configuration '{pipeline_name}' not found.")
        if 'steps' not in config or not isinstance(config['steps'], list):
            raise ExpertisePipelineManagerError(
                f"Pipeline '{pipeline_name}' has no 'steps' list or it's not a list."
            )
        return config

    def _get_expert_wrapper(self, expert_id: str, step_expert_config: Dict[str, Any]) -> BaseExpertWrapper:
        """
        Retrieves or creates an expert wrapper.

        Args:
            expert_id: The unique identifier for the expert.
            step_expert_config: The configuration for the expert from the pipeline step.
                                This config is passed to the factory.

        Returns:
            An instance of BaseExpertWrapper.

        Raises:
            ExpertisePipelineManagerError: If expert_id is not provided in step_expert_config.
        """
        if not expert_id:
            raise ExpertisePipelineManagerError("Expert ID not provided in step configuration.")

        if expert_id in self.experts_cache:
            # print(f"Cache hit for expert: {expert_id}")
            return self.experts_cache[expert_id]

        # print(f"Cache miss for expert: {expert_id}. Creating new instance.")
        # The factory is responsible for using the step_expert_config to create the expert.
        # This config might include type, model, instructions, specific tools, etc.
        expert_instance = self.expert_wrapper_factory(expert_id, step_expert_config)
        self.experts_cache[expert_id] = expert_instance
        return expert_instance

    async def execute_pipeline(
        self,
        pipeline_name: str,
        initial_input: Dict[str, Any],
        **kwargs  # Passed to each expert's process method
    ) -> Any:
        """
        Executes a defined pipeline.

        Args:
            pipeline_name: The name of the pipeline to execute.
            initial_input: The initial data to start the pipeline with.
            **kwargs: Additional arguments to pass to each expert's process method.

        Returns:
            The final output of the pipeline.

        Raises:
            ExpertisePipelineManagerError: For configuration issues or runtime errors in pipeline execution.
        """
        pipeline_config = self._get_pipeline_config(pipeline_name)
        print(f"\n--- Executing Pipeline: {pipeline_name} ---")

        current_data: Any = initial_input.copy()
        shared_context: Dict[str, Any] = {"initial_input": initial_input.copy()}

        for i, step in enumerate(pipeline_config['steps']):
            expert_id = step.get("expert_id")
            if not expert_id:
                raise ExpertisePipelineManagerError(f"Step {i} in pipeline '{pipeline_name}' missing 'expert_id'.")

            # The 'step' dictionary itself contains all configuration for this expert invocation
            # including 'expert_id', 'expert_type', 'init_params', 'input_mapping', etc.
            # It's passed as step_expert_config to _get_expert_wrapper.
            print(f"Step {i+1}/{len(pipeline_config['steps'])}: Using Expert ID '{expert_id}' with step config: {step}")

            expert_wrapper = self._get_expert_wrapper(expert_id, step) # Pass the whole step as config

            # Prepare input for the expert's process method
            process_input_data = {}

            # 1. Handle explicit input from shared_context (takes precedence for the primary data)
            input_from_sc_key = step.get("input_from_shared_context")
            if input_from_sc_key:
                if input_from_sc_key not in shared_context:
                    raise ExpertisePipelineManagerError(
                        f"Pipeline '{pipeline_name}', Step {i+1} ('{expert_id}'): "
                        f"Specified 'input_from_shared_context' key '{input_from_sc_key}' not found in shared_context."
                    )
                process_input_data = shared_context[input_from_sc_key]
                # Ensure it's a dict if we are to merge other things into it
                if not isinstance(process_input_data, dict):
                    # If it's not a dict, it becomes the sole input, usually under a specific key
                    # For simplicity, let's assume if it's not a dict, it's a simple value assigned to a default key like 'data'
                    # Or, the expert must be designed to handle non-dict input directly.
                    # For now, let's wrap it if it's not a dict, and the step needs to expect that.
                    # A better approach is for the step to define how this data is structured.
                    # For this version, we will assume the expert process method can handle it or
                    # input_mapping (if more advanced) would structure it.
                    # Let's simplify: if input_from_shared_context is used, it's the primary input.
                    # If it's a dict, other things can be merged. If not, it's up to the expert.
                    pass


            # 2. Merge current_data if specified or if process_input_data is still empty
            #    This allows carrying over data or using current_data as default.
            if step.get("merge_current_data", True) or not input_from_sc_key:
                if isinstance(current_data, dict):
                    if isinstance(process_input_data, dict):
                        process_input_data.update(current_data) # current_data can augment or be augmented
                    elif not input_from_sc_key : # only use current_data if no shared_context input was primary
                        process_input_data = current_data.copy()
                elif not input_from_sc_key : # current_data is not a dict, use as is if no shared_context input
                     process_input_data = current_data


            # 3. Simple input_mapping from shared_context or initial_input
            #    (e.g., {"target_key_for_expert": "$shared_context.source_key"})
            #    This is a very simplified version of input mapping.
            input_mapping = step.get("input_mapping", {})
            if not isinstance(process_input_data, dict) and input_mapping:
                 # If base input is not a dict, we can't easily map into it.
                 # We'll create a new dict for the input.
                 process_input_data = {}

            for target_key, source_path in input_mapping.items():
                if not isinstance(source_path, str): continue # simple validation

                value_source = None
                actual_source_key = source_path
                if source_path.startswith("$shared_context."):
                    actual_source_key = source_path.split('.', 1)[1]
                    value_source = shared_context
                elif source_path.startswith("$initial_input."):
                    actual_source_key = source_path.split('.', 1)[1]
                    value_source = shared_context.get("initial_input", {}) # initial_input is in shared_context

                if value_source is not None and actual_source_key in value_source:
                    process_input_data[target_key] = value_source[actual_source_key]
                else:
                    print(f"Warning: In pipeline '{pipeline_name}', step '{expert_id}', "
                          f"input_mapping key '{source_path}' not found or source invalid.")

            print(f"  - Input to expert '{expert_id}': {process_input_data}")
            step_output = await expert_wrapper.process(process_input_data, **kwargs)
            print(f"  - Output from expert '{expert_id}': {step_output}")

            # Output Handling
            output_to_sc_key = step.get("output_to_shared_context")
            if output_to_sc_key:
                shared_context[output_to_sc_key] = step_output
                print(f"  - Saved output to shared_context['{output_to_sc_key}']")
            else:
                # Default: pass output as current_data for the next step
                current_data = step_output
                print(f"  - Output becomes current_data for next step.")

            # Allow step to also update current_data even if output went to shared_context
            if step.get("update_current_data_with_output", False) and output_to_sc_key:
                if isinstance(current_data, dict) and isinstance(step_output, dict):
                    current_data.update(step_output)
                else: # Overwrite if types are not dicts or if current_data is not dict
                    current_data = step_output


        print(f"--- Pipeline {pipeline_name} execution finished ---")
        return current_data


if __name__ == '__main__':
    # --- Dummy/Mock Components for Testing ---
    class MockKnowledgeConfigLoader:
        def __init__(self, dummy_data: Optional[Dict[str, Any]] = None):
            self._dummy_data = dummy_data if dummy_data is not None else {}
            print("MockKnowledgeConfigLoader initialized.")

        def get_config_section(self, section_name: str) -> Any:
            print(f"MockKnowledgeConfigLoader: get_config_section('{section_name}') called.")
            return self._dummy_data.get(section_name)

    def dummy_expert_wrapper_factory(expert_id: str, expert_config_from_step: Dict[str, Any]) -> BaseExpertWrapper:
        """
        A dummy factory that creates ADKLlmExpertWrapper instances.
        The expert_config_from_step is what's defined in the pipeline step.
        It should contain 'avg_expert_config' for the BaseExpertWrapper.
        """
        print(f"DummyFactory: Received request for Expert ID '{expert_id}' "
              f"with step config: {expert_config_from_step}")

        # The config for the BaseExpertWrapper itself (init_params, capabilities)
        # should be nested under 'avg_expert_config' within the step's expert_config.
        base_wrapper_config = expert_config_from_step.get("avg_expert_config", {})

        # Example: if type is specified, could return different dummy experts
        # expert_type = expert_config_from_step.get("type", "ADKLlmExpertWrapper")

        return ADKLlmExpertWrapper(
            expert_id=expert_id,
            avg_expert_config=base_wrapper_config # This includes init_params, capabilities
        )

    # --- Sample Pipeline Configurations ---
    sample_pipelines = [
        {
            "name": "direct_expert_interaction",
            "description": "A simple pipeline with a single expert.",
            "steps": [
                {
                    "expert_id": "GreetingExpert",
                    "expert_config": { # Config for this step, passed to factory
                        "avg_expert_config": { # Config for the BaseExpertWrapper
                           "init_params": {"model_name": "friendly-model"},
                           "capabilities": {"can_greet": True}
                        }
                    },
                    # No output_to_shared_context, so output becomes current_data
                }
            ]
        },
        {
            "name": "simple_rag_pipeline",
            "description": "A pipeline that retrieves documents then generates an answer.",
            "steps": [
                {
                    "expert_id": "DocumentRetriever",
                     "expert_config": {
                        "avg_expert_config": {
                           "init_params": {"model_name": "retrieval-model-v1"},
                           "capabilities": {"can_retrieve": True}
                        }
                    },
                    "output_to_shared_context": "retrieved_documents"
                    # Input for this step will be the initial_input (current_data)
                },
                {
                    "expert_id": "AnswerGenerator",
                    "expert_config": {
                        "avg_expert_config": {
                           "init_params": {"model_name": "generation-model-v2"},
                           "capabilities": {"can_generate_answers": True}
                        }
                    },
                    # This expert needs the original query (from initial_input/current_data)
                    # and the retrieved_documents (from shared_context).
                    # The simplified input handling will merge current_data (containing the query)
                    # with mappings from shared_context.
                    "input_mapping": {
                        "documents": "$shared_context.retrieved_documents",
                        # Assumes the query is still in current_data or passed via initial_input
                        # "original_query": "$initial_input.query" # More explicit mapping
                    },
                    # No output_to_shared_context, final answer is pipeline output
                }
            ]
        }
    ]

    async def run_tests():
        mock_loader = MockKnowledgeConfigLoader()
        manager = ExpertisePipelineManager(
            pipeline_configs=sample_pipelines,
            knowledge_loader=mock_loader,
            expert_wrapper_factory=dummy_expert_wrapper_factory
        )

        # Test 1: Direct Expert Interaction
        print("\n--- Test 1: Direct Expert Interaction ---")
        direct_input = {"query": "Hello there!"}
        try:
            direct_output = await manager.execute_pipeline("direct_expert_interaction", direct_input)
            print(f"Final output from 'direct_expert_interaction': {direct_output}")
        except ExpertisePipelineManagerError as e:
            print(f"Error in Test 1: {e}")

        # Test 2: Simple RAG Pipeline
        print("\n--- Test 2: Simple RAG Pipeline ---")
        rag_input = {"user_query": "What is AVG?"} # Changed key to avoid clash with ADKLlmExpertWrapper's 'query'
        try:
            rag_output = await manager.execute_pipeline("simple_rag_pipeline", rag_input)
            print(f"Final output from 'simple_rag_pipeline': {rag_output}")
        except ExpertisePipelineManagerError as e:
            print(f"Error in Test 2: {e}")

        # Test 3: Non-existent pipeline
        print("\n--- Test 3: Non-existent pipeline ---")
        try:
            await manager.execute_pipeline("non_existent_pipeline", {"query": "test"})
        except ExpertisePipelineManagerError as e:
            print(f"Caught expected error for non-existent pipeline: {e}")

        # Test 4: Pipeline step missing expert_id
        print("\n--- Test 4: Pipeline step missing expert_id ---")
        broken_pipeline_config = [{
            "name": "broken_pipeline",
            "steps": [{"expert_config": {}}] # Missing expert_id
        }]
        manager_broken = ExpertisePipelineManager(broken_pipeline_config, mock_loader, dummy_expert_wrapper_factory)
        try:
            await manager_broken.execute_pipeline("broken_pipeline", {"query": "test"})
        except ExpertisePipelineManagerError as e:
            print(f"Caught expected error for broken pipeline: {e}")


    if __name__ == '__main__':
        asyncio.run(run_tests())
