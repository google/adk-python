# Threat Intelligence Support Agent Sample

## Overview

This sample demonstrates a multi-agent system designed for threat intelligence support. It showcases how a `ClassifierAgent` can analyze user input and route it to a specialized `ResponderAgent`. The `ResponderAgent` then uses dynamically loaded instructions from text files based on the classified category to generate a relevant (though currently placeholder) response.

This example highlights:
-   Agent collaboration (classification followed by specialized response).
-   Dynamic instruction loading for agents based on context.
-   Basic structure for creating and testing agents within the ADK.

## How to Run

### Prerequisites

1.  **Python Environment**: Ensure you have a Python 3.8+ environment.
2.  **ADK Installation**: The Application Development Kit (ADK) for building agents must be installed. If you are running this sample from within the ADK repository, the necessary paths should be configured by the environment or the `main.py` script's path adjustments. If running standalone, ensure the ADK is in your `PYTHONPATH`.

### Running the Sample

To run the main demonstration script, navigate to the root directory of the ADK repository and execute the following command:

```bash
python -m contributing.samples.threat_intelligence_support.main
```

This will:
1.  Simulate a user query.
2.  Use `ClassifierAgent` to determine a category for the query.
3.  Use `ResponderAgent` to generate a response based on the category and its specific instruction file.
4.  Print the category and the final response to the console.

## File Descriptions

-   `main.py`:
    Orchestrates the sample application. It initializes the agents, passes a sample user query to the `ClassifierAgent`, and then uses the resulting category to invoke the `ResponderAgent` for a final response.

-   `classifier_agent.py`:
    Contains the `ClassifierAgent`. This agent is responsible for analyzing the initial user input and determining a category (e.g., "Malware Analysis", "Phishing Report"). It loads its own set of instructions from `classifier.txt`.

-   `responder_agent.py`:
    Contains the `ResponderAgent`. This agent takes the user input and the category determined by the `ClassifierAgent`. It then dynamically loads specific instructions from a corresponding `.txt` file (e.g., `malware_analysis.txt` for the "Malware Analysis" category) to tailor its response.

-   `guidelines.txt`:
    A placeholder file containing general guidelines. This is used by the `ResponderAgent` as a fallback if a category-specific instruction file is not found.

-   `classifier.txt`:
    Contains instructions specifically for the `ClassifierAgent` to guide its classification process.

-   `malware_analysis.txt`, `phishing_report.txt` (etc.):
    These `.txt` files (e.g., `malware_analysis.txt`) provide category-specific instructions for the `ResponderAgent`. The `ResponderAgent` loads the content of the relevant file based on the category it receives.

-   `tests/test_agents.py`:
    Contains basic unit tests for `ClassifierAgent` and `ResponderAgent`. These tests verify instantiation, basic invocation, and (given the placeholder nature of current agents) expected outputs based on simple logic and file access. To run tests, you can use a test runner like `pytest` from the repository root or run the file directly:
    ```bash
    python -m unittest contributing.samples.threat_intelligence_support.tests.test_agents
    ```

-   `.gitkeep`:
    An empty file in the `tests/` directory to ensure Git tracks the directory.

## Placeholder LLM Calls

Currently, both the `ClassifierAgent` and `ResponderAgent` use placeholder logic instead of making actual calls to a Large Language Model (LLM).
-   The `ClassifierAgent` uses simple keyword matching to determine the category.
-   The `ResponderAgent` generates a canned response that includes the category and parts of the input/instructions.

To make these agents fully functional, the placeholder sections in their `invoke` (or `run`) methods would need to be replaced with actual LLM API interactions, using the prompts constructed from user input and loaded instructions.

This sample provides the structural foundation for such an integration.
