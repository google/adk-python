# Slack Agent Sample

## Introduction

This sample connects an ADK agent to Slack using `SlackRunner`, which bridges an
ADK `Runner` to a [Slack Bolt](https://tools.slack.dev/bolt-python/) app running
in [Socket Mode](https://api.slack.com/apis/connections/socket). Messages that
mention the bot, and direct messages to it, are handled by the agent and its
responses are posted back to the same conversation.

Unlike the other samples in this directory, this one is a standalone script
rather than an `adk run` / `adk web` package: it builds its own `Runner` and
owns the event loop, so it is started with `python` directly.

## Prerequisites

Install ADK with Slack support:

```bash
pip install "google-adk[slack]"
```

Create and configure a Slack app (Socket Mode, bot token scopes, and event
subscriptions) by following
[the Slack integration guide](../../../../src/google/adk/integrations/slack/README.md).
That gives you the two tokens this sample reads from the environment:
`SLACK_BOT_TOKEN` (starts with `xoxb-`) and `SLACK_APP_TOKEN` (starts with
`xapp-`).

## How to Use

This script does not read a `.env` file, so export both the LLM credentials and
the Slack tokens. For example, for using Google AI Studio:

```bash
export GOOGLE_GENAI_USE_ENTERPRISE=FALSE
export GOOGLE_API_KEY="{your api key}"
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_APP_TOKEN="xapp-..."
```

Then run the script from the root of the ADK repository:

```bash
python contributing/samples/integrations/slack_agent/agent.py
```

The bot stays connected until you interrupt it. Mention it in a channel it has
been invited to, or send it a direct message, to talk to the agent.
