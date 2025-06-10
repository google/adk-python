"""
Manages loading and accessing configuration for various knowledge sources,
ingestion pipelines, knowledge graphs, vector stores, decision models,
and expertise pipelines.
Supports YAML and JSON file formats.
"""
import yaml
import json
import os

class ConfigLoadError(Exception):
    """Custom exception for errors during configuration loading or parsing."""
    pass

class KnowledgeConfigLoader:
    """
    Loads, validates, and provides access to AVG toolkit configurations.
    """
    def __init__(self, config_path: str):
        """
        Initializes the KnowledgeConfigLoader.

        Args:
            config_path: Path to the configuration file (YAML or JSON).
        """
        self.config_path = config_path
        self.config: dict = self._load_config()

    def _load_config(self) -> dict:
        """
        Loads the configuration file based on its extension.

        Returns:
            A dictionary containing the loaded configuration.

        Raises:
            FileNotFoundError: If the config_path does not exist.
            ConfigLoadError: If there's an error parsing the file or if validation fails.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        _, file_extension = os.path.splitext(self.config_path)
        config_data = {}

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if file_extension in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif file_extension == '.json':
                    config_data = json.load(f)
                else:
                    raise ConfigLoadError(
                        f"Unsupported configuration file type: {file_extension}. "
                        "Supported types are .yaml, .yml, .json."
                    )
            if not isinstance(config_data, dict):
                raise ConfigLoadError("Configuration content is not a valid dictionary.")

        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Error parsing YAML file {self.config_path}: {e}") from e
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"Error parsing JSON file {self.config_path}: {e}") from e
        except Exception as e: # Catch any other unexpected errors during loading
            raise ConfigLoadError(f"An unexpected error occurred while loading {self.config_path}: {e}") from e

        if not self._validate_config(config_data):
            # _validate_config will print its own messages, but we still raise an error
            raise ConfigLoadError(f"Configuration validation failed for {self.config_path}.")

        return config_data

    def _validate_config(self, config_data: dict) -> bool:
        """
        Validates the loaded configuration data.
        Placeholder for more robust validation logic.

        Args:
            config_data: The configuration data (dictionary) to validate.

        Returns:
            True if validation passes (for now), False otherwise.
        """
        if not isinstance(config_data, dict):
            print(f"Validation Error: Configuration data is not a dictionary. Found type: {type(config_data)}")
            return False

        print(f"Notice: '{self.__class__.__name__}._validate_config' is currently a placeholder. "
              "Basic check (is dict) passed. Implement more specific validation rules as needed.")
        # Add more specific validation rules here, e.g., schema validation
        # For example, check for presence of key sections:
        # expected_sections = ['knowledge_sources', 'ingestion_pipelines', 'expertise_pipelines']
        # for section in expected_sections:
        #     if section not in config_data:
        #         print(f"Validation Warning: Expected section '{section}' not found in config.")
        return True

    def get_knowledge_sources(self) -> list:
        """Returns the 'knowledge_sources' section of the config, or an empty list."""
        return self.config.get('knowledge_sources', [])

    def get_ingestion_pipelines(self) -> list:
        """Returns the 'ingestion_pipelines' section of the config, or an empty list."""
        return self.config.get('ingestion_pipelines', [])

    def get_knowledge_graphs(self) -> list:
        """Returns the 'knowledge_graphs' section of the config, or an empty list."""
        return self.config.get('knowledge_graphs', [])

    def get_vector_stores(self) -> list:
        """Returns the 'vector_stores' section of the config, or an empty list."""
        return self.config.get('vector_stores', [])

    def get_decision_models(self) -> list:
        """Returns the 'decision_models' section of the config, or an empty list."""
        return self.config.get('decision_models', [])

    def get_expertise_pipelines(self) -> list:
        """Returns the 'expertise_pipelines' section of the config, or an empty list."""
        return self.config.get('expertise_pipelines', [])

    def get_config_section(self, section_name: str) -> any:
        """
        Returns a specific top-level section from the configuration.

        Args:
            section_name: The name of the configuration section to retrieve.

        Returns:
            The configuration section if found, otherwise None.
        """
        return self.config.get(section_name)

if __name__ == '__main__':
    # Assuming this script is run from the repository root,
    # or paths are adjusted accordingly.
    # For subtask execution, this path should be relative to the workspace root.
    DUMMY_CONFIG_PATH = "avg/adk_avg_toolkit/dummy_config.yaml"
    DUMMY_MALFORMED_JSON_PATH = "avg/adk_avg_toolkit/dummy_malformed_config.json"
    NON_EXISTENT_CONFIG_PATH = "avg/adk_avg_toolkit/non_existent_config.yaml"

    print(f"--- Testing with valid config: {DUMMY_CONFIG_PATH} ---")
    try:
        if not os.path.exists(DUMMY_CONFIG_PATH):
            print(f"Error: Dummy config file {DUMMY_CONFIG_PATH} not found. "
                  "Make sure it was created in the previous steps.")
            # Create a minimal dummy if it's missing, for robustness in testing.
            with open(DUMMY_CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump({'knowledge_sources': [{'name': 'fallback_dummy'}]}, f)
            print(f"Created a minimal {DUMMY_CONFIG_PATH} for the test run.")


        loader = KnowledgeConfigLoader(DUMMY_CONFIG_PATH)
        print("Config loaded successfully.")
        print(f"Knowledge Sources: {loader.get_knowledge_sources()}")
        print(f"Ingestion Pipelines: {loader.get_ingestion_pipelines()}")
        print(f"Expertise Pipelines: {loader.get_expertise_pipelines()}")
        print(f"Vector Stores: {loader.get_vector_stores()}")
        print(f"Decision Models (should be empty): {loader.get_decision_models()}")
        print(f"Full 'knowledge_sources' section: {loader.get_config_section('knowledge_sources')}")

    except FileNotFoundError as e:
        print(f"Handled Error: {e}")
    except ConfigLoadError as e:
        print(f"Handled Error: {e}")
    except Exception as e:
        print(f"Unexpected Error during valid config test: {e}")

    print(f"\n--- Testing with non-existent config: {NON_EXISTENT_CONFIG_PATH} ---")
    try:
        loader_non_existent = KnowledgeConfigLoader(NON_EXISTENT_CONFIG_PATH)
    except FileNotFoundError as e:
        print(f"Handled Error: {e}")
    except ConfigLoadError as e: # Should ideally be FileNotFoundError
        print(f"Handled Error (expected FileNotFoundError, got ConfigLoadError): {e}")
    except Exception as e:
        print(f"Unexpected Error during non-existent config test: {e}")

    print(f"\n--- Testing with malformed JSON config: {DUMMY_MALFORMED_JSON_PATH} ---")
    try:
        if not os.path.exists(DUMMY_MALFORMED_JSON_PATH):
            print(f"Error: Dummy malformed config file {DUMMY_MALFORMED_JSON_PATH} not found. "
                  "Make sure it was created in the previous steps.")
            with open(DUMMY_MALFORMED_JSON_PATH, 'w', encoding='utf-8') as f:
                f.write("{'invalid_json': ") # Intentionally malformed
            print(f"Created a minimal {DUMMY_MALFORMED_JSON_PATH} for the test run.")

        loader_malformed = KnowledgeConfigLoader(DUMMY_MALFORMED_JSON_PATH)
    except ConfigLoadError as e:
        print(f"Handled Error: {e}")
    except FileNotFoundError as e: # Should ideally be ConfigLoadError
         print(f"Handled Error (expected ConfigLoadError, got FileNotFoundError): {e}")
    except Exception as e:
        print(f"Unexpected Error during malformed config test: {e}")

    # Example of loading a non-yaml/json file
    print("\n--- Testing with unsupported file type (.txt) ---")
    DUMMY_TXT_CONFIG_PATH = "avg/adk_avg_toolkit/dummy_config.txt"
    try:
        with open(DUMMY_TXT_CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write("this is not yaml or json")
        loader_txt = KnowledgeConfigLoader(DUMMY_TXT_CONFIG_PATH)
    except ConfigLoadError as e:
        print(f"Handled Error: {e}")
    except Exception as e:
        print(f"Unexpected Error during unsupported file type test: {e}")
    finally:
        # Clean up dummy files created just for this test section
        if os.path.exists(DUMMY_TXT_CONFIG_PATH):
            os.remove(DUMMY_TXT_CONFIG_PATH)

    print("\n--- KnowledgeConfigLoader tests finished ---")
