# GCS Tools Sample

## Introduction

This sample agent demonstrates the Google Cloud Storage (GCS) first-party tools in ADK,
distributed via the `google.adk.integrations.gcs` module. These tools include:

1. `gcs_get_bucket`

Get metadata information about a GCS bucket.

1. `gcs_list_objects`

List object names in a GCS bucket.

1. `gcs_get_object_metadata`

Get metadata information about a GCS object (blob).

## How to use

Set up environment variables in your `.env` file for using
[Google AI Studio](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-ai-studio)
or
[Google Cloud Vertex AI](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-cloud-vertex-ai)
for the LLM service for your agent. For example, for using Google AI Studio you
would set:

- GOOGLE_GENAI_USE_ENTERPRISE=FALSE
- GOOGLE_API_KEY={your api key}

### With Application Default Credentials (gcloud)

This is the easiest way to use your own Google Cloud identity for both the tools AND the LLM.

1. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
1. Run `gcloud auth application-default login` in your terminal.
1. Configure your environment to use Vertex AI (which supports ADC) instead of AI Studio:
   - `export GOOGLE_GENAI_USE_ENTERPRISE=TRUE`
   - `export GOOGLE_CLOUD_PROJECT={your-project-id}`
1. Ensure the Vertex AI API is enabled and you have the correct permissions:
   - Enable API: `gcloud services enable aiplatform.googleapis.com`
   - Grant Role: `gcloud projects add-iam-policy-binding {your-project-id} --member="user:{your-email}" --role="roles/aiplatform.user"`
1. Set `CREDENTIALS_TYPE = None` in `agent.py`.
1. Run the agent.

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

1. Follow https://developers.google.com/workspace/guides/configure-oauth-consent
   to add scope "https://www.googleapis.com/auth/cloud-platform" and
   "https://www.googleapis.com/auth/devstorage.full_control" as a declaration, this is used
   for review purpose.

1. Follow
   https://developers.google.com/identity/protocols/oauth2/web-server#creatingcred
   to add http://localhost/dev-ui/ to "Authorized redirect URIs".

   Note: localhost here is just a hostname that you use to access the dev ui,
   replace it with the actual hostname you use to access the dev ui.

1. For 1st run, allow popup for localhost in Chrome.

1. Configure your `.env` file to add two more variables before running the
   agent:

   - OAUTH_CLIENT_ID={your client id}
   - OAUTH_CLIENT_SECRET={your client secret}

   Note: don't create a separate .env, instead put it to the same .env file that
   stores your Vertex AI or Dev ML credentials

1. Set `CREDENTIALS_TYPE=AuthCredentialTypes.OAUTH2` in `agent.py` and run the
   agent

## Sample prompts

- Show me metadata for the my-bucket bucket.
- List all objects in the my-bucket bucket.
- Get metadata for the my-object.txt object in my-bucket.
- Download the GCS object my-object.txt in my-bucket to a local file ~/Downloads/downloaded.txt.
- Upload my local file /tmp/local_report.pdf to my-bucket as report.pdf.
