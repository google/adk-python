# AgentCoreSessionServiceConfig

`AgentCoreSessionServiceConfig` holds the AgentCore Memory resource id and optional AWS region for `AgentCoreSessionService`. Construct it when you want to pass configuration as an object rather than as `memory_id=` / `region_name=` keyword arguments.

## Introduction

`AgentCoreSessionService` needs to know which Memory resource to write to, and which AWS region hosts that resource. Those two values are the only fields on this config type. `AgentCoreSessionService(config=...)` takes precedence over the convenience `memory_id` and `region_name` constructor arguments.

Callers who only have a memory id can skip this type and pass `memory_id=` directly. The config object is the better fit when configuration is loaded from a file or built in one place and handed to the service in another.

## Get started

```python
from google.adk.integrations.agentcore import AgentCoreSessionService
from google.adk.integrations.agentcore import AgentCoreSessionServiceConfig

config = AgentCoreSessionServiceConfig(
    memory_id="my-memory-abc123",
    region_name="us-east-1",
)
session_service = AgentCoreSessionService(config=config)
```

## How it works

The service reads `config.memory_id` on every AgentCore API call as `memoryId`. When `client` is omitted, the service builds a `boto3.client("bedrock-agentcore", ...)` and passes `config.region_name` if it is set. When `region_name` is unset, boto3 uses its default credential and region chain.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `memory_id` | `str` | required | AgentCore Memory resource id sent as `memoryId`. |
| `region_name` | `str \| None` | `None` | AWS region for the lazily built boto3 client. |

- **`memory_id`** is the id of the Memory resource, not an ARN. Every `create_event`, `list_events`, `list_sessions`, `list_actors`, and `delete_event` call includes it.
- **`region_name`** is ignored when you pass a pre-built `client` to `AgentCoreSessionService`, because that client already has a region.

## Related samples

See [AgentCoreSessionService](../agentcore_session_service/index.md) for Runner wiring, event mapping, and limitations.
