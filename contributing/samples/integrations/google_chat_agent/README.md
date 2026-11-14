# Google Chat Agent Sample

## Introduction

This sample runs an ADK agent as a Google Chat app using `GoogleChatRunner`.
Users can chat with the agent in direct messages, and in spaces and group
conversations where the app is mentioned. The agent's responses are posted
back to the same conversation, in the same thread when the space is organized
by thread.

Like the [Teams sample](../teams_agent/), this one is a standalone script
rather than an `adk run` / `adk web` package: it builds its own `Runner` and
serves its own HTTP endpoint, so it is started with `python` directly.

## Prerequisites

Install ADK with Google Chat support:

```bash
pip install "google-adk[google-chat]"
```

The script does not read a `.env` file, so export the LLM credentials. For
example, for using Google AI Studio:

```bash
export GOOGLE_GENAI_USE_ENTERPRISE=FALSE
export GOOGLE_API_KEY="{your api key}"
```

## Run it in Google Chat

1. In a Google Cloud project, enable the Google Chat API and create a service
   account for the app. Replies are posted as this service account with the
   `https://www.googleapis.com/auth/chat.bot` scope.

1. Deploy the script at a public HTTPS URL, running as that service account.
   On Cloud Run, keep CPU allocated outside requests, because replies are sent
   after the request is acknowledged:

   ```bash
   gcloud run deploy google-chat-agent --source . \
     --service-account "{service account email}" \
     --no-cpu-throttling --allow-unauthenticated \
     --set-env-vars "HOST=0.0.0.0,CHAT_AUDIENCE=https://{your host}/"
   ```

1. On the Google Chat API **Configuration** page, set **Connection settings**
   to **HTTP endpoint URL** with `https://{your host}/`, and set
   **Authentication Audience** to **HTTP endpoint URL**. To use the project
   number as the audience instead, set `CHAT_AUDIENCE` to the project number.

1. If you build the app as a Google Workspace add-on, Chat signs requests with
   a per-project account. Copy it from **Connection settings** and export it:

   ```bash
   export CHAT_SERVICE_ACCOUNT_EMAIL="service-{project number}@gcp-sa-gsuiteaddons.iam.gserviceaccount.com"
   ```

1. Make the app visible to yourself, then open Google Chat, start a direct
   message with the app, and send it a message.

To run the script on your machine instead, leave `CHAT_AUDIENCE` unset, which
turns off request verification, and expose the port with a tunnel. Do not run
without verification on a server that is reachable from the internet, because
anyone could send messages to the agent.

See the
[GoogleChatRunner guide](../../../../docs/guides/integrations/google_chat/google_chat_runner/index.md)
for how sessions, threads, and errors are handled.
