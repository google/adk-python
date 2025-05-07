# Application Integration Agent Sample

## Introduction

This sample demonstrates how to use the `ApplicationIntegrationToolset` within an ADK agent to interact with external applications, specifically Jira in this case. The agent (`agent.py`) is configured to manage Jira issues using a pre-configured Application Integration connection.

## Prerequisites

1.  **Set up Integration Connection:**
    *   You need an existing [Integration connection](https://cloud.google.com/integration-connectors/docs/overview) configured to interact with your Jira instance. Follow the [documentation](https://google.github.io/adk-docs/tools/google-cloud-tools/#use-integration-connectors) to provision the Integration Connector in Google Cloud and then use this [documentation](https://cloud.google.com/integration-connectors/docs/connectors/jiracloud/configure) to create an JIRA connection. Note the `Connection Name`, `Project ID`, and `Location` of your connection.
    * 

2.  **Configure Environment Variables:**
    *   Create a `.env` file in the same directory as `agent.py` (or add to your existing one).
    *   Add the following variables to the `.env` file, replacing the placeholder values with your actual connection details:

      ```dotenv
      CONNECTION_NAME=<YOUR_JIRA_CONNECTION_NAME>
      CONNECTION_PROJECT=<YOUR_GOOGLE_CLOUD_PROJECT_ID>
      CONNECTION_LOCATION=<YOUR_CONNECTION_LOCATION>
      ```

## How to Use

1.  **Install Dependencies:** Ensure you have the necessary libraries installed (e.g., `google-adk`, `python-dotenv`).
2.  **Run the Agent:** Execute the agent script from your terminal:
    ```bash
    python agent.py
    ```
3.  **Interact:** Once the agent starts, you can interact with it by typing prompts related to Jira issue management.

## Sample Prompts

Here are some examples of how you can interact with the agent:

*   `Can you list me all the issues ?`
*   `Can you list me all the projects ?`
*   `Can you create an issue: "Bug in product XYZ" in project ABC ?`

