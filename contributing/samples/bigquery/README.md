# BigQuery Tools Sample

## Introduction

This sample agent demonstrates the BigQuery first-party tools in ADK,
distributed via the `google.adk.tools.bigquery` module. These tools include:

1. `list_dataset_ids`

  Fetches BigQuery dataset ids present in a GCP project.

1. `get_dataset_info`

  Fetches metadata about a BigQuery dataset.

1. `list_table_ids`

  Fetches table ids present in a BigQuery dataset.

1. `get_table_info`

  Fetches metadata about a BigQuery table.

1. `execute_sql`

  Runs a SQL query in BigQuery.

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

## Sample prompts

* which weather datasets exist in bigquery public data?
* tell me more about noaa_lightning
* which tables exist in the ml_datasets dataset?
* show more details about the penguins table
* compute penguins population per island.
