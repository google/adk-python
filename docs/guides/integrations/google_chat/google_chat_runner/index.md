# GoogleChatRunner

GoogleChatRunner connects an ADK agent to Google Chat, so that users can chat
with the agent in direct messages, spaces, and group conversations.

## Introduction

GoogleChatRunner serves the HTTP endpoint of a
[Google Chat app](https://developers.google.com/workspace/chat). Google Chat
delivers each message to that endpoint, GoogleChatRunner passes the message
text to your agent, and it posts the agent's text responses back to the same
conversation with the
[Chat API](https://developers.google.com/workspace/chat/api/reference/rest).

The runner accepts events from Chat apps and from Chat apps built as Google
Workspace add-ons, so the same endpoint works for either configuration.

## Get started

The following example builds an agent and serves the Chat endpoint at
`http://localhost:8080/`.

```python
import asyncio

from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.google_chat import GoogleChatRunner
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService


async def main():
  runner = Runner(
      agent=LlmAgent(name="chat_agent", instruction="Be concise."),
      app_name="chat_app",
      session_service=InMemorySessionService(),
      auto_create_session=True,
  )

  chat_runner = GoogleChatRunner(runner=runner)
  await chat_runner.start(audience="https://example.com/")


asyncio.run(main())
```

Install the dependencies with `pip install "google-adk[google-chat]"`. The
`Runner` needs `auto_create_session=True`, because GoogleChatRunner picks the
session id for each conversation and does not create sessions itself.

The `audience` argument is the Authentication Audience configured for the Chat
app. Every request from Google Chat carries a signed bearer token, and the
runner rejects requests whose token is not signed for that audience.

## How it works

GoogleChatRunner handles message events and ignores every other event type,
such as the app being added to a space. For a message, it reads
`argumentText`, which is the message text without the app's `@mention`, and
runs the agent with that text as the user message.

Each run uses two identifiers from the Chat event:

- The session id is derived from the conversation. In a space organized by
  thread, each thread is its own conversation, so a thread maps to its own
  session, and the replies go to that thread. In a direct message or a group
  conversation, the whole conversation is one session, and the replies are
  posted as plain messages. When a direct message has
  [conversation sessions](https://developers.google.com/workspace/chat/sessions)
  turned on, each session thread maps to its own ADK session. The derived id is
  a hash that contains only letters, digits, and hyphens, because Chat resource
  names contain `/`, which some session services reject.
- The user id is the sender's Chat user name, such as `users/1234567890`,
  which stays the same across conversations.

Because ADK keys sessions by both user and session id, two people talking to
the app in the same thread each get their own session.

Google Chat shows an error if the endpoint does not answer within 30 seconds,
and an agent turn with model and tool calls can take that long. To stay within
that limit, GoogleChatRunner acknowledges each event right away with an empty
response and runs the agent in the background. The replies are posted through
the Chat API as the app. Within one server process, messages for the same
session wait for the previous run to finish, so that the agent always sees
them in order.

Every non-thought text part of a final event is posted as its own message, in
the order the agent produces them. If the run raises an exception, or the Chat
API rejects a reply, the user receives a generic error message and the
exception is logged. The exception text is not sent to Chat, because it can
contain internal details.

## Configuration options

GoogleChatRunner takes these constructor arguments:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `runner` | `Runner` | Required | The ADK runner that executes the agent. |
| `credentials` | `google.auth.credentials.Credentials \| None` | `None` | Credentials used to post replies with the `chat.bot` scope. `None` uses Application Default Credentials. |
| `api_endpoint` | `str` | `"https://chat.googleapis.com"` | Base URL of the Chat API. |

The `start` method serves the endpoint until it is cancelled:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `audience` | `str \| None` | Required | The HTTP endpoint URL or the project number that Chat signs requests for. `None` turns off request verification. |
| `service_account_email` | `str` | `"chat@system.gserviceaccount.com"` | The account that signs requests. |
| `host` | `str` | `"localhost"` | The interface to listen on. Use `"0.0.0.0"` in a container. |
| `port` | `int` | `8080` | The port to listen on. |

## Advanced applications

### Configure the Chat app

Google Chat only delivers messages to a configured Chat app, and the app posts
its replies as a service account.

1. In a Google Cloud project, enable the Google Chat API and create a service
   account. Run the server as that service account, or pass its credentials to
   the constructor, because replies are posted with its identity.
1. On the Chat API **Configuration** page, set **Connection settings** to
   **HTTP endpoint URL** and enter the public HTTPS URL of your server.
1. Choose the **Authentication Audience**, and pass the same value as
   `audience`.

The Authentication Audience decides which kind of token Chat sends:

| Authentication Audience | Token | `audience` value |
| :--- | :--- | :--- |
| HTTP endpoint URL | Google-signed ID token for `chat@system.gserviceaccount.com` | The endpoint URL, exactly as configured |
| Project number | JWT self-signed by `chat@system.gserviceaccount.com` | The project number |

A Chat app built as a Google Workspace add-on only supports the endpoint URL
audience, and its requests are signed by a per-project account. Copy that
account from **Connection settings** and pass it as `service_account_email`:

```python
await chat_runner.start(
    audience="https://example.com/",
    service_account_email=(
        "service-123456789@gcp-sa-gsuiteaddons.iam.gserviceaccount.com"
    ),
)
```

### Test locally without verification

Passing `audience=None` accepts requests without a token, so that you can send
hand-written Chat events to a local server. Do not turn off verification on a
server that is reachable from the internet, because it lets anyone send
messages to the agent.

### Mount the endpoint in your own server

`create_app` returns the aiohttp application that `start` would serve. Use it
when the endpoint should share a server with other routes, or when your hosting
setup manages the event loop.

```python
from aiohttp import web

app = web.Application()
app.add_subapp(
    "/chat",
    chat_runner.create_app(audience="https://example.com/chat/"),
)
web.run_app(app, host="0.0.0.0", port=8080)
```

With the sub-application mounted at `/chat`, the endpoint URL to configure in
the Chat API is `https://{your host}/chat/`.

## Limitations

GoogleChatRunner requires aiohttp, which is installed with
`pip install "google-adk[google-chat]"`.

Only text is exchanged. Attachments, cards, and slash commands sent by the user
are not passed to the agent, and the agent's non-text output is not posted to
Chat. A single reply is limited to the Chat message size of 32,000 bytes.

The agent only sees the messages addressed to it. When the app is mentioned
partway through a thread, the earlier messages in that thread are not part of
the session.

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

- [`google_chat_agent`](../../../../../contributing/samples/integrations/google_chat_agent/)
  runs an agent as a Google Chat app.
- The [TeamsRunner guide](../../teams/teams_runner/index.md) describes the same
  pattern for Microsoft Teams.
