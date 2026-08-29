# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# AgentCore Memory Integration for ADK

This integration stores ADK sessions in
[Amazon Bedrock AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html)
short-term memory.

AgentCore has no session object. Each ADK session is an AgentCore
`(actorId, sessionId)` pair, and each ADK `Event` is one AgentCore event.

The mapping follows the approach in
[divakaivan/agentcore_session_service](https://github.com/divakaivan/agentcore_session_service)
(#6920): a conversational payload so AgentCore can extract memories from the
turn text, plus a blob of the full ADK event JSON so tool calls, state deltas,
and metadata round-trip.

## Installation

```bash
pip install "google-adk[agentcore]"
```

AWS credentials must be available to boto3 (environment, shared config, or
instance role). You also need an AgentCore Memory resource id.

## Quick Start

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

## Mapping

| ADK | AgentCore |
| :--- | :--- |
| `(app_name, user_id)` | `actorId` = `{app_name}:{user_id}` |
| `session.id` | `sessionId` |
| `Event` (non-partial) | `create_event` payload: optional `conversational` text (`USER` / `ASSISTANT` / `TOOL`) and a `blob` of `Event.model_dump_json()` |

`create_session` writes a bootstrap blob with `extractionMode=SKIP` so the
session exists in `list_sessions` / `get_session` before the first user turn.
`delete_session` deletes every event in that session (AgentCore has no
delete-session API).

App- and user-scoped state is stored on the session via event `state_delta`
replay. It is not shared across sessions the way Redis/Firestore backends
share `app:` / `user:` keys.

## Configuration

`AgentCoreSessionServiceConfig`:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `memory_id` | `str` | required | AgentCore Memory resource id. |
| `region_name` | `Optional[str]` | `None` | AWS region. Falls back to the boto3 default chain. |

You can pass a pre-built `boto3.client("bedrock-agentcore", ...)` as `client`.
