# AgentCoreSessionService

`AgentCoreSessionService` is a `BaseSessionService` that stores ADK sessions in Amazon Bedrock AgentCore Memory short-term memory. Pass it to `Runner` as `session_service` when the conversation history should live in AgentCore instead of in memory, Redis, or Firestore.

## Introduction

AgentCore Memory has no session resource. An actor and a session id identify a stream of events, and each event is one turn payload. ADK sessions are finer-grained: they carry a list of `Event` objects, session state, and an `(app_name, user_id, session_id)` key.

`AgentCoreSessionService` is the compatibility layer. `Runner` depends on `BaseSessionService`, so any agent that already uses `Runner(..., session_service=...)` can swap this implementation in. The service writes each non-partial ADK event as one AgentCore event: conversational text so AgentCore can extract long-term memories from the turn, and a blob of the full ADK event JSON so tool calls, state deltas, and metadata come back on `get_session`.

Install the extra, create an AgentCore Memory resource, and point the service at its id. AWS credentials come from the boto3 default chain.

## Get started

```python
from google.adk.agents import Agent
from google.adk.integrations.agentcore import AgentCoreSessionService
from google.adk.runners import Runner

session_service = AgentCoreSessionService(
    memory_id="my-memory-abc123",
    region_name="us-east-1",
)

agent = Agent(
    name="assistant",
    instruction="You are a helpful AI assistant.",
)

runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=session_service,
)
```

```shell
pip install 'google-adk[agentcore]'
```

`create_session` / `get_session` / `append_event` work the same as on `InMemorySessionService`. `Runner.run_async` loads the session, appends user and model events, and the service flushes each complete event to AgentCore.

## How it works

ADK identifies a session by `(app_name, user_id, session_id)`. AgentCore identifies a stream by `(memoryId, actorId, sessionId)`. The service maps:

- `actorId` to `{app_name}:{user_id}`, so two ADK apps sharing one Memory resource do not mix users
- `sessionId` to the ADK session id
- each non-partial `Event` to one `create_event` call

When the event has text parts, the AgentCore payload starts with a `conversational` item whose role is `USER`, `ASSISTANT`, or `TOOL`. A `blob` of `Event.model_dump_json()` always follows, and `get_session` rebuilds history from those blobs. Streaming fragments with `partial=True` are not written, matching other persistent backends.

AgentCore has no create-session or delete-session API. `create_session` writes a bootstrap event with `extractionMode=SKIP` so the session appears in `list_sessions` before the first user turn. `delete_session` deletes every event in that stream.

`list_sessions` with a `user_id` lists that actor. Omitting `user_id` lists every actor whose id starts with `{app_name}:`. Returned sessions have empty `events` lists, as `BaseSessionService` requires.

## Configuration options

### Constructor

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `memory_id` | `str \| None` | `None` | AgentCore Memory resource id. Required when `config` is omitted. |
| `config` | `AgentCoreSessionServiceConfig \| None` | `None` | Full config object. Overrides `memory_id` and `region_name` when set. |
| `client` | boto3 client or test double | `None` | Pre-built `bedrock-agentcore` client. Built lazily from boto3 when omitted. |
| `region_name` | `str \| None` | `None` | AWS region. Ignored when `config` is set. Falls back to the boto3 default chain. |

- **`memory_id`** is the Memory resource you created in AgentCore. Every event is written under this id.
- **`config`** is useful when you already constructed `AgentCoreSessionServiceConfig`. See that type for field-level detail.
- **`client`** is how tests inject a fake, and how an application can pass a client with custom retries or endpoints.
- **`region_name`** is needed when the default AWS region is not the region where the Memory resource lives.

### `AgentCoreSessionServiceConfig` fields

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `memory_id` | `str` | required | AgentCore Memory resource id. |
| `region_name` | `str \| None` | `None` | AWS region for the `bedrock-agentcore` client. |

## Advanced applications

### Passing a boto3 client

```python
import boto3
from google.adk.integrations.agentcore import AgentCoreSessionService

client = boto3.client("bedrock-agentcore", region_name="eu-west-2")
session_service = AgentCoreSessionService(
    memory_id="my-memory-abc123",
    client=client,
)
```

Use this when credential loading, retries, or the endpoint should not follow process-wide boto3 defaults.

### Filtering history on get

```python
from google.adk.sessions.base_session_service import GetSessionConfig

session = await session_service.get_session(
    app_name="my_app",
    user_id="user-42",
    session_id="session-a",
    config=GetSessionConfig(num_recent_events=20),
)
```

`after_timestamp` keeps events whose ADK `timestamp` is at least that unix time. `num_recent_events=0` returns the session with an empty event list.

## Limitations

- **App- and user-scoped state is not shared across sessions.** Redis and Firestore keep `app:` and `user:` keys in separate documents. This backend replays `state_delta` onto the session that stored the events. A new session for the same user does not inherit `user:` keys from older sessions.

- **Empty AgentCore sessions expire.** AgentCore deletes empty sessions after about a day. `create_session` writes a bootstrap event so the session is not empty. If every event is later deleted, the session disappears.

- **Session ids must be acceptable to AgentCore.** The service does not rewrite ids. Prefer the generated UUID, or a value in the character set AgentCore accepts.

- **This is short-term memory only.** Long-term extraction is whatever the Memory resource is configured to do with conversational payloads. The ADK `BaseMemoryService` search API is a different interface.
