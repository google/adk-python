# BigQuery MCP Toolset Sample

## Introduction

This sample agent demonstrates using ADK's `McpToolset` to interact with
BigQuery's official MCP endpoint, allowing an agent to access and execute
tools by leveraging the Model Context Protocol (MCP). These tools include:


1. `list_dataset_ids`

  Fetches BigQuery dataset ids present in a GCP project.

2. `get_dataset_info`

  Fetches metadata about a BigQuery dataset.

3. `list_table_ids`

  Fetches table ids present in a BigQuery dataset.

4. `get_table_info`

  Fetches metadata about a BigQuery table.

5. `execute_sql`

  Runs or dry-runs a SQL query in BigQuery.

## How to use

Set up your project and local authentication by following the guide
[Use the BigQuery remote MCP server](https://docs.cloud.google.com/bigquery/docs/use-bigquery-mcp).
This agent uses Application Default Credentials (ADC) to authenticate with the
BigQuery MCP endpoint.

Set up environment variables in your `.env` file for using
[Google AI Studio](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-ai-studio)
or
[Google Cloud Vertex AI](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-cloud-vertex-ai)
for the LLM service for your agent. For example, for using Google AI Studio you
would set:

* GOOGLE_GENAI_USE_VERTEXAI=FALSE
* GOOGLE_API_KEY={your api key}

Then run the agent using `adk run .` or `adk web .` in this directory.

## Sample prompts

* which weather datasets exist in bigquery public data?
* tell me more about noaa_lightning
* which tables exist in the ml_datasets dataset?
* show more details about the penguins table
* compute penguins population per island.
