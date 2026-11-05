# Microsoft Teams Agent Sample

## Introduction

This sample runs an ADK agent as a Microsoft Teams bot using `TeamsRunner`,
which bridges an ADK `Runner` to a
[Microsoft 365 Agents SDK](https://learn.microsoft.com/en-us/microsoft-365/agents-sdk/)
app. Users can chat with the agent in personal chats, group chats, and channels
where the bot is mentioned, and the agent's responses are posted back to the
same conversation.

Like the [Slack sample](../slack_agent/), this one is a standalone script
rather than an `adk run` / `adk web` package: it builds its own `Runner` and
serves its own HTTP endpoint, so it is started with `python` directly.

## Prerequisites

Install ADK with Teams support:

```bash
pip install "google-adk[teams]"
```

The script does not read a `.env` file, so export the LLM credentials. For
example, for using Google AI Studio:

```bash
export GOOGLE_GENAI_USE_ENTERPRISE=FALSE
export GOOGLE_API_KEY="{your api key}"
```

## Try it locally

The [Microsoft 365 Agents Playground](https://www.npmjs.com/package/@microsoft/teams-app-test-tool)
sends Teams-style messages to a local endpoint without a bot registration.
Allow unauthenticated requests for local testing only, then run the script from
the root of the ADK repository:

```bash
export CONNECTIONS__SERVICE_CONNECTION__SETTINGS__ANONYMOUS_ALLOWED=True
python contributing/samples/integrations/teams_agent/agent.py
```

In another terminal, start the playground and chat with the agent in the
browser window it opens:

```bash
npm install -g @microsoft/teams-app-test-tool
teamsapptester
```

## Run it in Teams

1. Create an Azure Bot resource with a single tenant app registration, and
   create a client secret for it.

1. Host the script at a public HTTPS URL. For a container, set `HOST=0.0.0.0`;
   the port defaults to `3978` and can be changed with `PORT`.

1. Set the Azure Bot's messaging endpoint to `https://{your host}/api/messages`
   and enable the Microsoft Teams channel.

1. Start the script with the bot's credentials instead of anonymous access:

   ```bash
   export CONNECTIONS__SERVICE_CONNECTION__SETTINGS__CLIENTID="{app id}"
   export CONNECTIONS__SERVICE_CONNECTION__SETTINGS__CLIENTSECRET="{client secret}"
   export CONNECTIONS__SERVICE_CONNECTION__SETTINGS__TENANTID="{tenant id}"
   python contributing/samples/integrations/teams_agent/agent.py
   ```

1. Add the bot to Teams from the Azure Bot's Teams channel page, and send it a
   message.

See the
[TeamsRunner guide](../../../../docs/guides/integrations/teams/teams_runner/index.md)
for how sessions, retries, and errors are handled.
