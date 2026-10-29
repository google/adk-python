# Microsoft Teams Notification Sample

## Introduction

This sample shows how an ADK agent can send notifications to a Microsoft Teams
channel through an incoming webhook. The `send_teams_notification` function
tool posts an [Adaptive Card](https://adaptivecards.io/) with a title, a
message, and optional key/value facts.

This is one-way: the agent posts to Teams, but users can't chat with the agent
from Teams. For an interactive bot, see the [Slack sample](../slack_agent/) for
the equivalent pattern on Slack.

## Create a Teams webhook

Microsoft 365 Connectors (the classic "Incoming Webhook" connector) are being
retired, so create the webhook with the Teams **Workflows** app:

1. In Teams, open the channel, select **More options (...)** > **Workflows**.
1. Choose the **Send webhook alerts to a channel** template (or another
   template triggered by "When a Teams webhook request is received").
1. Pick the team and channel, then save the workflow.
1. Copy the generated webhook URL.

See
[Create an Incoming Webhook](https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-incoming-webhook)
for details. The same payload also works with an existing Connector webhook.

The webhook URL contains a signature that lets anyone holding it post to the
channel. Treat it as a secret: keep it out of source control, and in production
load it from a secret store such as Secret Manager.

## How to use

Set up environment variables in your `.env` file for using
[Google AI Studio](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-ai-studio)
or
[Google Cloud Vertex AI](https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-cloud-vertex-ai)
for the LLM service, plus the webhook URL. For example:

```
GOOGLE_GENAI_USE_ENTERPRISE=FALSE
GOOGLE_API_KEY={your api key}
TEAMS_WEBHOOK_URL={your webhook url}
```

Then run the agent:

```bash
adk web contributing/samples/integrations
```

and select `teams_notification_agent`.

## Sample prompts

- Post to Teams that the nightly build for `payments-service` failed in
  staging, owner is Alex.
- Send a Teams notification that the Q3 report is ready for review.

## Sending notifications without the LLM

To notify on every run regardless of what the model decides, call
`send_teams_notification` from an `after_agent_callback` instead of exposing it
as a tool.
