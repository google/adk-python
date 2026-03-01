# BigQuery Tools Sample

## Introduction

This sample agent demonstrates the BigQuery first-party tools in ADK,
distributed via the `google.adk.tools.bigquery` module. These tools include:

1. `list_dataset_ids`

  Fetches BigQuery dataset ids present in a GCP project.

2. `get_dataset_info`

  Fetches metadata about a BigQuery dataset.

3. `list_table_ids`

  Fetches table ids present in a BigQuery dataset.

4. `get_table_info`

  Fetches metadata about a BigQuery table.

5. `get_job_info`
  Fetches metadata about a BigQuery job.

6. `execute_sql`

  Runs or dry-runs a SQL query in BigQuery.

7. `ask_data_insights`

  Natural language-in, natural language-out tool that answers questions
  about structured data in BigQuery. Provides a one-stop solution for generating
  insights from data.

  **Note**: This tool requires additional setup in your project. Please refer to
  the official [Conversational Analytics API documentation](https://cloud.google.com/gemini/docs/conversational-analytics-api/overview)
  for instructions.

8. `forecast`

  Perform time series forecasting using BigQuery's `AI.FORECAST` function,
  leveraging the TimesFM 2.0 model.

9. `analyze_contribution`

  Perform contribution analysis in BigQuery by creating a temporary
  `CONTRIBUTION_ANALYSIS` model and then querying it with
  `ML.GET_INSIGHTS` to find top contributors for a given metric.

10. `detect_anomalies`

  Perform time series anomaly detection in BigQuery by creating a temporary
  `ARIMA_PLUS` model and then querying it with
  `ML.DETECT_ANOMALIES` to detect time series data anomalies.

## How to use

Set up environment variables in your `.env` file for using
[Google AI Studio](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-ai-studio)
or
[Google Cloud Vertex AI](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-cloud-vertex-ai)
for the LLM service for your agent. For example, for using Google AI Studio you
would set:

* GOOGLE_GENAI_USE_VERTEXAI=FALSE
* GOOGLE_API_KEY={your api key}

### With Application Default Credentials

This mode is useful for quick development when the agent builder is the only
user interacting with the agent. The tools are run with these credentials.

1. Create application default credentials on the machine where the agent would
be running by following https://cloud.google.com/docs/authentication/provide-credentials-adc.

1. Set `CREDENTIALS_TYPE=None` in `agent.py`

1. Run the agent

### With Service Account Keys

This mode is useful for quick development when the agent builder wants to run
the agent with service account credentials. The tools are run with these
credentials.

1. Create service account key by following https://cloud.google.com/iam/docs/service-account-creds#user-managed-keys.

1. Set `CREDENTIALS_TYPE=AuthCredentialTypes.SERVICE_ACCOUNT` in `agent.py`

1. Download the key file and replace `"service_account_key.json"` with the path

1. Run the agent

### With Interactive OAuth

1. Follow
https://developers.google.com/identity/protocols/oauth2#1.-obtain-oauth-2.0-credentials-from-the-dynamic_data.setvar.console_name.
to get your client id and client secret. Be sure to choose "web" as your client
type.

1. Follow https://developers.google.com/workspace/guides/configure-oauth-consent to add scope "https://www.googleapis.com/auth/bigquery".

1. Follow https://developers.google.com/identity/protocols/oauth2/web-server#creatingcred to add http://localhost/dev-ui/ to "Authorized redirect URIs".

  Note: localhost here is just a hostname that you use to access the dev ui,
  replace it with the actual hostname you use to access the dev ui.

1. For 1st run, allow popup for localhost in Chrome.

1. Configure your `.env` file to add two more variables before running the agent:

  * OAUTH_CLIENT_ID={your client id}
  * OAUTH_CLIENT_SECRET={your client secret}

  Note: don't create a separate .env, instead put it to the same .env file that
  stores your Vertex AI or Dev ML credentials

1. Set `CREDENTIALS_TYPE=AuthCredentialTypes.OAUTH2` in `agent.py` and run the agent

### With Agent Engine and Gemini Enterprise

This mode is useful when you deploy the agent to Vertex AI Agent Engine and
want to make it available in Gemini Enterprise, allowing the agent to access
BigQuery on behalf of the end-user. This setup uses OAuth 2.0 managed by
Gemini Enterprise.

1. Create an Authorization resource in Gemini Enterprise by following the guide at
[Register and manage ADK agents hosted on Vertex AI Agent Engine](https://docs.cloud.google.com/gemini/enterprise/docs/register-and-manage-an-adk-agent) to:
  * Create OAuth 2.0 credentials in your Google Cloud project.
  * Create an Authorization resource in Gemini Enterprise, linking it to your
    OAuth 2.0 credentials. When creating this resource, you will define a
    unique identifier (`AUTH_ID`).

2. Prepare the sample agent for consuming the access token provided by Gemini
Enterprise and deploy to Vertex AI Agent Engine.
  * Set `CREDENTIALS_TYPE=AuthCredentialTypes.HTTP` in `agent.py`. This
configures the agent to use access tokens provided by Gemini Enterprise and
provided by Agent Engine via the tool context.
  * Replace `AUTH_ID` in `agent.py` with your authorization resource identifier
    from step 1.
  * [Deploy your agent to Vertex AI Agent Engine](https://google.github.io/adk-docs/deploy/agent-engine/).

3. [Register your deployed agent with Gemini Enterprise](https://docs.cloud.google.com/gemini/enterprise/docs/register-and-manage-an-adk-agent#register-an-adk-agent), attaching the
Authorization resource `AUTH_ID`. When this agent is invoked through Gemini
Enterprise, an access token obtained using these OAuth credentials will be
passed to the agent and made available in the ADK `tool_context` under the key
`AUTH_ID`, which `agent.py` is configured to use.

Once registered, users interacting with your agent via Gemini Enterprise will
go through an OAuth consent flow, and Agent Engine will provide the agent with
the necessary access tokens to call BigQuery APIs on their behalf.

## Sample prompts

* which weather datasets exist in bigquery public data?
* tell me more about noaa_lightning
* which tables exist in the ml_datasets dataset?
* show more details about the penguins table
* compute penguins population per island.
