# Eventarc Tools Sample

## Introduction

This sample agent demonstrates the Eventarc first-party tool in ADK,
distributed via the `google.adk.integrations.eventarc` module. This tool suite currently includes:

1. `publish_message`

Publishes a structured event in CloudEvents format to a Google Cloud Eventarc message bus.

## How to use

Set up environment variables in your `.env` file for using
[Google AI Studio](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-ai-studio)
or
[Google Cloud Vertex AI](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-cloud-vertex-ai)
for the LLM service for your agent. For example, for using Google AI Studio you
would set:

- GOOGLE_GENAI_USE_VERTEXAI=FALSE
- GOOGLE_API_KEY={your api key}

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

1. Follow https://developers.google.com/workspace/guides/configure-oauth-consent to add scope "https://www.googleapis.com/auth/cloud-platform".

1. Follow https://developers.google.com/identity/protocols/oauth2/web-server#creatingcred to add http://localhost/dev-ui/ to "Authorized redirect URIs".

Note: localhost here is just a hostname that you use to access the dev ui,
replace it with the actual hostname you use to access the dev ui.

1. For 1st run, allow popup for localhost in Chrome.

1. Configure your `.env` file to add two more variables before running the agent:

- OAUTH_CLIENT_ID={your client id}
- OAUTH_CLIENT_SECRET={your client secret}

Note: don't create a separate .env, instead put it to the same .env file that
stores your Vertex AI or Dev ML credentials

1. Set `CREDENTIALS_TYPE=AuthCredentialTypes.OAUTH2` in `agent.py` and run the agent

### With Agent Identity (in Agent Runtime / Vertex AI Reasoning Engine)

When deploying this agent to Agent Runtime, it can use its unique SPIFFE-based Agent Identity to authenticate. This is the recommended security best practice.

1. **Configure Deployment**: Create a `.agent_engine_config.json` file in this directory to specify the identity type:

   ```json
   {
     "identity_type": "AGENT_IDENTITY"
   }
   ```

1. **Use Default Credentials**: Leave `CREDENTIALS_TYPE = None` in `agent.py` (which is the default). This configures the agent to use Application Default Credentials (ADC), which automatically resolves to the Agent Identity in the container runtime environment.

1. **Deploy the Agent**: Deploy your agent using the ADK CLI:

   ```bash
   uv run adk deploy agent_engine \
     --project=YOUR_PROJECT_ID \
     --region=YOUR_REGION \
     --display_name=eventarc-agent-test \
     contributing/samples/integrations/eventarc
   ```

   Take note of the generated **Reasoning Engine ID** (e.g., `1234567890`) and the **Project Number** of your project.

1. **Grant IAM Permissions**: Grant the Eventarc Message Bus User role (`roles/eventarc.messageBusUser`) to the Agent Identity principal at the project level:

   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="principal://agents.global.org-171145599760.system.id.goog/resources/aiplatform/projects/YOUR_PROJECT_NUMBER/locations/YOUR_REGION/reasoningEngines/YOUR_REASONING_ENGINE_ID" \
     --role="roles/eventarc.messageBusUser"
   ```

   *Note: Eventarc Advanced message buses require `roles/eventarc.messageBusUser` for publishing, rather than `roles/eventarc.publisher`.*

## Sample prompts

- "Publish an event of type 'com.example.hello' to bus 'projects/my-project/locations/global/messageBuses/my-bus' with data 'Hello World' and source '//my/agent'"
- "Send a JSON payload to Eventarc bus 'projects/my-project/locations/global/messageBuses/my-bus' representing a user sign-up event"
