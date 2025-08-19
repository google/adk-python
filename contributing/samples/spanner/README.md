# Spanner Tools Sample

## Introduction

This sample agent demonstrates the Spanner first-party tools in ADK,
distributed via the `google.adk.tools.spanner` module. These tools include:

1. `list_table_names`

  Fetches Spanner table names present in a GCP Spanner database.

1. `list_table_indexes`

  Fetches Spanner table indexes present in a GCP Spanner database.

1. `list_table_index_columns`

  Fetches Spanner table index columns present in a GCP Spanner database.

1. `list_named_schemas`

  Fetches named schema for a Spanner database.

1. `get_table_schema`

  Fetches Spanner database table schema.

1. `execute_sql`

  Runs a SQL query in Spanner database.

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

1.  Follow https://developers.google.com/workspace/guides/configure-oauth-consent
    to add scope "https://www.googleapis.com/auth/spanner.data" and
    "https://www.googleapis.com/auth/spanner.admin" as declaration, this is used
    for review purpose.

1.  Follow
    https://developers.google.com/identity/protocols/oauth2/web-server#creatingcred
    to add http://localhost/dev-ui/ to "Authorized redirect URIs".

    Note: localhost here is just a hostname that you use to access the dev ui,
    replace it with the actual hostname you use to access the dev ui.

1.  For 1st run, allow popup for localhost in Chrome.

1.  Configure your `.env` file to add two more variables before running the
    agent:

    *   OAUTH_CLIENT_ID={your client id}
    *   OAUTH_CLIENT_SECRET={your client secret}

    Note: don't create a separate .env, instead put it to the same .env file that
    stores your Vertex AI or Dev ML credentials

1.  Set `CREDENTIALS_TYPE=AuthCredentialTypes.OAUTH2` in `agent.py` and run the
    agent

## Sample prompts

* Show me all tables in the product_db Spanner database.
* Describe the schema of the product_table table.
* List all indexes on the product_table table.
* Show me the first 10 rows of data from the product_table table.
* Write a query to find the most popular product by joining the product_table and sales_table tables.
