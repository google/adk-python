# BigQuery Data Agent Sample

This sample agent demonstrates ADK's first-party tools for interacting with
Data Agents based on BigQuery's Conversational Analytics API, distributed via
the `google.adk.tools.bigquery` module. These tools allow you to list,
inspect, and
chat with BigQuery Data Agents using natural language.

These tools leverage stateful conversations, meaning you can ask follow-up
questions in the same session, and the agent will maintain context.

## Prerequisites

1.  An active Google Cloud project with BigQuery and Gemini APIs enabled.
2.  Google Cloud authentication configured for Application Default Credentials:
    ```bash
    gcloud auth application-default login
    ```
3.  At least one Data Agent created in BigQuery Studio (Data Canvas). These
    agents are created and configured in the Google Cloud console and point to
    your BigQuery tables or other data sources.

## Tools Used

*   `list_accessible_data_agents`: Lists Data Agents you have permission to
    access in the configured GCP project.
*   `get_data_agent_info`: Retrieves details about a specific Data Agent given
    its full resource name.
*   `ask_data_agent`: Chats with a specific Data Agent using natural language.
    This tool maintains conversation state: if you ask multiple
    questions to the same agent in one session, it will use the same
    conversation, allowing for follow-ups. If you switch agents, a new
    conversation will be started for the new agent.

## How to Run

1.  Navigate to the root of the ADK repository.
2.  Run the agent using the ADK CLI:
    ```bash
    adk run --agent-path contributing/samples/bigquery_data_agent
    ```
3.  The CLI will prompt you for input. You can ask questions like the examples
    below.

## Sample prompts

*   "List accessible data agents."
*   "Using agent
    `projects/my-project/locations/global/dataAgents/sales-agent-123`, who were
    my top 3 customers last quarter?"
*   "How does that compare to the quarter before?"
