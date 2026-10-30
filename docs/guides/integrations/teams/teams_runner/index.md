# TeamsRunner

TeamsRunner connects an ADK agent to Microsoft Teams, so that users can chat
with the agent in personal chats, group chats, and channels.

## Introduction

TeamsRunner bridges an ADK `Runner` to an `AgentApplication` from the
[Microsoft 365 Agents SDK](https://learn.microsoft.com/en-us/microsoft-365/agents-sdk/),
which is the successor to the Bot Framework SDK. Teams delivers each message to
an HTTPS endpoint that you host. TeamsRunner serves that endpoint, passes the
message text to your agent, and posts the agent's text responses back to the
same conversation.

Use TeamsRunner when users should talk to the agent inside Teams. To send
one-way notifications to a channel, a Teams Workflows webhook is simpler and
does not need a bot registration. The `teams_notification_agent` sample shows
that approach.

## Get started

The following example builds an agent, wraps it in an `AgentApplication`
configured from environment variables, and serves the Teams endpoint at
`http://localhost:3978/api/messages`.

```python
import asyncio
import os

from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.teams import TeamsRunner
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from microsoft_agents.activity import load_configuration_from_env
from microsoft_agents.authentication.msal import MsalConnectionManager
from microsoft_agents.hosting.aiohttp import CloudAdapter
from microsoft_agents.hosting.core import AgentApplication
from microsoft_agents.hosting.core import MemoryStorage
from microsoft_agents.hosting.core import TurnState


async def main():
  runner = Runner(
      agent=LlmAgent(name="teams_agent", instruction="Be concise."),
      app_name="teams_app",
      session_service=InMemorySessionService(),
      auto_create_session=True,
  )

  sdk_config = load_configuration_from_env(os.environ)
  connection_manager = MsalConnectionManager(**sdk_config)
  agent_app = AgentApplication[TurnState](
      storage=MemoryStorage(),
      adapter=CloudAdapter(connection_manager=connection_manager),
      connection_manager=connection_manager,
      **sdk_config,
  )

  teams_runner = TeamsRunner(runner=runner, agent_app=agent_app)
  await teams_runner.start(
      connection_manager.get_default_connection_configuration()
  )


asyncio.run(main())
```

Install the dependencies with `pip install "google-adk[teams]"`. The
`Runner` needs `auto_create_session=True`, because TeamsRunner picks the session
id for each conversation and does not create sessions itself.

## How it works

TeamsRunner registers a handler for message activities on the
`AgentApplication` you pass in. When a message arrives, the handler reads the
text, which the `AgentApplication` has already stripped of the bot's
`@mention`, and runs the agent with that text as the user message.

Each run uses two identifiers from the Teams activity:

- The session id is derived from the Teams conversation id. A personal chat
  and a group chat each have one conversation id, and each thread in a channel
  has its own, so a channel thread maps to its own session. The derived id is
  a hash that contains only letters, digits, and hyphens, because Teams
  conversation ids contain characters such as `:` and `;` that some session
  services reject.
- The user id is the sender's Microsoft Entra object id, which stays the same
  across chats. Senders without one fall back to their Teams channel account
  id.

Because ADK keys sessions by both user and session id, two people talking to
the bot in the same channel thread each get their own session.

Teams retries any request that takes longer than 15 seconds to answer, and an
agent turn with model and tool calls can easily take that long. To avoid
duplicate runs, TeamsRunner acknowledges each request right away and runs the
agent in the background. The replies are sent through a proactive turn that
reuses the identity of the incoming request, and a typing indicator shows while
the agent works. Within one server process, messages for the same session wait
for the previous run to finish, so that the agent always sees them in order.

Every non-thought text part of a final event is posted as its own message, in
the order the agent produces them. If the run raises an exception, the user
receives a generic error message and the exception is logged. The exception
text is not sent to Teams, because it can contain internal details.

## Configuration options

TeamsRunner takes two constructor arguments:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `runner` | `Runner` | Required | The ADK runner that executes the agent. |
| `agent_app` | `AgentApplication[TurnState]` | Required | The Microsoft 365 Agents SDK application that receives Teams activities. |

The `start` method serves the endpoint until it is cancelled:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `auth_configuration` | `AgentAuthConfiguration` | Required | Settings used to validate the token on each incoming request. |
| `host` | `str` | `"localhost"` | The interface to listen on. Use `"0.0.0.0"` in a container. |
| `port` | `int` | `3978` | The port to listen on. |

The `AgentApplication` stays under your control, so its own options apply
unchanged. For example, `remove_recipient_mention` and `start_typing_timer`
both default to `True`.

## Advanced applications

### Register a bot for Teams

Teams only delivers messages to a registered bot. The registration gives the
bot an identity in Microsoft Entra, and the SDK uses that identity both to
verify incoming requests and to authenticate its replies.

1. Create an Azure Bot resource in the Azure portal. Choose a single tenant
   app registration and create a client secret for it.
1. Set the messaging endpoint of the Azure Bot to the public HTTPS URL of your
   server, followed by `/api/messages`.
1. Enable the Microsoft Teams channel on the Azure Bot, then add the bot to
   Teams, either from the channel page or through a Teams app manifest.

The `load_configuration_from_env` call in the Get started example reads the
registration from these environment variables:

```bash
export CONNECTIONS__SERVICE_CONNECTION__SETTINGS__CLIENTID="{app id}"
export CONNECTIONS__SERVICE_CONNECTION__SETTINGS__CLIENTSECRET="{client secret}"
export CONNECTIONS__SERVICE_CONNECTION__SETTINGS__TENANTID="{tenant id}"
```

The client secret grants the ability to send messages as the bot, so load it
from a secret store such as Secret Manager in production.

### Test locally without a registration

For local development, the
[Microsoft 365 Agents Playground](https://www.npmjs.com/package/@microsoft/teams-app-test-tool)
sends Teams-style activities to your endpoint without a bot registration.
Allow unauthenticated requests by setting
`CONNECTIONS__SERVICE_CONNECTION__SETTINGS__ANONYMOUS_ALLOWED=True` and leave
the other connection settings unset, then run `teamsapptester`. Do not enable
anonymous access on a server that is reachable from the internet, because it
lets anyone send messages to the agent.

### Mount the endpoint in your own server

`create_app` returns the aiohttp application that `start` would serve. Use it
when the endpoint should share a server with other routes, or when your hosting
setup manages the event loop.

```python
from aiohttp import web

app = web.Application()
app.add_subapp(
    "/teams",
    teams_runner.create_app(
        connection_manager.get_default_connection_configuration()
    ),
)
web.run_app(app, host="0.0.0.0", port=8080)
```

With the sub-application mounted at `/teams`, the messaging endpoint to
register in Azure is `https://{your host}/teams/api/messages`.

## Limitations

TeamsRunner requires the Microsoft 365 Agents SDK, which is installed with
`pip install "google-adk[teams]"`.

Only text is exchanged. Attachments, images, and Adaptive Card submissions
sent by the user are not passed to the agent, and the agent's non-text output
is not posted to Teams.

The background run lives in the server process. If the process stops while an
agent is running, the reply is lost. Hosting platforms that scale to zero or
throttle CPU between requests, such as Cloud Run with request-based billing,
can pause the background run. Keep CPU allocated outside requests on those
platforms.

Sessions are not shared across processes unless the session service is. With
`InMemorySessionService`, conversation history is lost on restart and is not
visible to other replicas, so use a persistent service such as
`DatabaseSessionService` or `VertexAiSessionService` when you run more than one
instance.

## Related samples

- [`teams_agent`](../../../../../contributing/samples/integrations/teams_agent/)
  runs an agent as a Teams bot.
- [`teams_notification_agent`](../../../../../contributing/samples/integrations/teams_notification_agent/)
  sends one-way notifications to a Teams channel through a webhook.
- The [Slack integration](../../../../../src/google/adk/integrations/slack/README.md)
  provides the same pattern for Slack.
