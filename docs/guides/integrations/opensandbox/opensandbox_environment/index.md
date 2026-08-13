# OpenSandboxEnvironment

`OpenSandboxEnvironment` implements ADK's `BaseEnvironment` contract with an
isolated OpenSandbox workspace. It gives `EnvironmentToolset` a persistent
remote shell and byte-oriented filesystem without running model-generated code
on the ADK host.

## Introduction

ADK environment tools need four capabilities from a runtime: lifecycle
management, a working directory, command execution, and file I/O.
`OpenSandboxEnvironment` maps those operations to the asynchronous OpenSandbox
SDK. A single sandbox persists across tool calls, so commands can reuse files
and installed packages.

The environment can either create a sandbox or attach to an existing one.
Created sandboxes are owned by the environment and destroyed during `close()`.
Attached sandboxes remain caller-owned; closing the environment only releases
its local SDK resources.

The integration depends on:

- `BaseEnvironment` and `ExecutionResult` for the ADK runtime contract;
- `EnvironmentToolset` as its primary ADK consumer;
- the optional `opensandbox` package for lifecycle, command, and file APIs.

## Get started

Install the optional dependency and start an OpenSandbox lifecycle server:

```bash
pip install google-adk[opensandbox]
export OPEN_SANDBOX_DOMAIN="localhost:8080"
```

Then pass the environment to the standard toolset:

```python
from google.adk import Agent
from google.adk.integrations.opensandbox import OpenSandboxEnvironment
from google.adk.tools.environment import EnvironmentToolset

environment = OpenSandboxEnvironment(
    image="python:3.11",
    timeout=300,
)

root_agent = Agent(
    name="sandboxed_coding_agent",
    instruction=(
        "Use the environment tools to inspect files, write code, and run it."
    ),
    tools=[EnvironmentToolset(environment=environment)],
)
```

`EnvironmentToolset` initializes the sandbox before returning tools and closes
it when the toolset closes. Relative command and file paths resolve below
`/workspace` by default.

## How it works

Initialization is serialized with an async lock, so concurrent callers cannot
create duplicate sandboxes. For a new environment, initialization performs
these steps:

1. Build an OpenSandbox `ConnectionConfig` from explicit options or the
   `OPEN_SANDBOX_API_KEY` and `OPEN_SANDBOX_DOMAIN` environment variables.
1. Call `Sandbox.create()` with an image or snapshot.
1. Create the configured working directory through the filesystem API.
1. Mark the ADK environment initialized only after all setup succeeds.

If working-directory setup fails, the environment destroys a sandbox it
created. When attaching by `sandbox_id`, the same failure closes only the local
SDK client.

Before each operation on an owned sandbox, the environment renews its configured
lifetime. A finite command deadline extends that renewal past the deadline, and
a command without a deadline uses periodic renewal heartbeats. It does not renew
an attached sandbox because the caller owns that resource's expiration policy.

Operations hold a shared lifecycle lease. They can run concurrently, while
`close()` waits for in-flight operations before destroying or disconnecting the
sandbox. Once closing starts, new operations fail fast and a concurrent
`initialize()` waits until cleanup finishes before creating a new sandbox.
Concurrent `close()` calls share the same cleanup operation. If destroying an
owned sandbox fails, a later `close()` reconnects by ID and retries cleanup
before the environment can be initialized again.

Owned operations serialize renewal requests and remember the expiration time
returned by the service. A shorter concurrent operation cannot overwrite a
longer command's expiration. Operations that can run indefinitely also renew
periodically until they finish.

Commands use `RunCommandOpts` to set the working directory, command deadline,
and environment variables. OpenSandbox stdout, stderr, and exit status become
an ADK `ExecutionResult`. Nonzero exit status is returned normally rather than
raised as an exception.

File operations resolve relative paths below the working directory. Reads
return raw bytes, HTTP 404 becomes `FileNotFoundError`, and writes explicitly
use mode `644` rather than the OpenSandbox SDK's executable default.

## Configuration options

| Parameter           | Type                              | Default                         | Description                                                          |
| ------------------- | --------------------------------- | ------------------------------- | -------------------------------------------------------------------- |
| `image`             | `str \| SandboxImageSpec \| None` | `python:3.11`                   | Container image for a newly created sandbox.                         |
| `snapshot_id`       | `str \| None`                     | `None`                          | Snapshot startup source used instead of an image.                    |
| `sandbox_id`        | `str \| None`                     | `None`                          | Existing caller-owned sandbox to attach to.                          |
| `timeout`           | `int \| None`                     | `300`                           | Owned sandbox lifetime, at least 60 seconds; `None` disables expiry. |
| `ready_timeout`     | `float`                           | `30`                            | Maximum wait for create or connect readiness.                        |
| `working_dir`       | `str \| os.PathLike`              | `/workspace`                    | Absolute POSIX path used for commands and relative files.            |
| `env_vars`          | `dict[str, str] \| None`          | `None`                          | Variables applied during create and command execution.               |
| `metadata`          | `dict[str, str] \| None`          | `None`                          | Lifecycle metadata for a newly created sandbox.                      |
| `connection_config` | `ConnectionConfig \| None`        | `None`                          | Complete SDK connection configuration.                               |
| `api_key`           | `str \| None`                     | environment                     | OpenSandbox lifecycle API key.                                       |
| `domain`            | `str \| None`                     | environment or `localhost:8080` | Lifecycle API domain.                                                |
| `protocol`          | `str \| None`                     | `http`                          | Scheme used when the domain has no scheme.                           |
| `request_timeout`   | `float \| None`                   | `30`                            | HTTP request timeout in seconds.                                     |
| `use_server_proxy`  | `bool \| None`                    | `False`                         | Route sandbox service requests through the lifecycle server.         |

`image` and `snapshot_id` are mutually exclusive. Neither can be supplied with
`sandbox_id`. A complete `connection_config` also cannot be mixed with the
individual connection settings.

### Use a remote server

Basic connection options can be supplied explicitly:

```python
environment = OpenSandboxEnvironment(
    image="python:3.11",
    domain="sandbox.example.com",
    protocol="https",
    api_key="...",
    use_server_proxy=True,
)
```

Prefer environment-based secret injection in deployed applications rather than
putting an API key in source code.

## Advanced applications

### Attach without taking ownership

Use an existing workspace when another component manages its lifecycle:

```python
environment = OpenSandboxEnvironment(
    sandbox_id="existing-sandbox-id",
    env_vars={"TASK_ID": "analysis-42"},
)

await environment.initialize()
result = await environment.execute("printf '%s' \"$TASK_ID\"")
await environment.close()  # The remote sandbox remains running.
```

Command-level `env_vars` still apply to attached sandboxes. Their remote
lifetime is not renewed or destroyed by this environment.

### Start from a snapshot

Pass a snapshot when the workspace needs preinstalled dependencies:

```python
environment = OpenSandboxEnvironment(
    snapshot_id="python-data-tools",
    timeout=900,
)
```

Snapshots and images are alternative startup sources, so do not pass both.

## Limitations

- OpenSandbox SDK 0.1.x does not expose a dedicated command-timeout flag. The
  adapter marks a result timed out only when a deadline was supplied, execution
  lasted at least that long, and the SDK returned exit code `-1`. A process
  terminated by another SIGKILL near the deadline can be indistinguishable.
- SDK 0.1.15 parses command output as line-oriented SSE. Unicode separators
  U+0085, U+2028, and U+2029 can be split by the underlying HTTP parser and
  dropped. This is an SDK transport limitation, not an ADK file-I/O limitation.
- Command stream events do not preserve whether every original chunk ended in
  a newline. The adapter preserves text and ordering but reconstructs chunks
  with `\n` separators.
- Like the existing E2B and Daytona remote environments, absolute file paths
  can address locations outside the working directory. Treat the entire
  sandbox filesystem as the security boundary.
- `BaseEnvironment` currently has no endpoint discovery, snapshot creation,
  pause, or resume methods. Those OpenSandbox capabilities are outside this
  integration's public ADK contract.
- A `timeout` of `None` requires explicit cleanup and is runtime-dependent.
  Some Kubernetes workload providers can reject non-expiring sandboxes.
- If an owned sandbox expires while idle, the next operation surfaces the
  OpenSandbox error instead of silently creating an empty replacement. Call
  `close()` and then `initialize()` to create a fresh workspace.
- A command without a deadline can also delay `close()` until the command
  returns or its caller cancels it. This preserves in-flight operation safety;
  callers remain responsible for cancelling work that should not finish.

## Related samples

See
[`contributing/samples/environment_and_skills/opensandbox_environment`](../../../../../contributing/samples/environment_and_skills/opensandbox_environment/README.md)
for a runnable coding-agent configuration.

The [verification record](../verification.md) lists the automated checks and
the live-test boundary used for this integration.
