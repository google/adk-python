# PlimsollEnvironment

PlimsollEnvironment provides a persistent workspace in a sandbox you run
yourself with Plimsoll. Agents execute shell commands and read and write files
in it, and every call states the isolation tier it ran behind.

## Introduction

The PlimsollEnvironment class connects an agent to a self-hosted Plimsoll
daemon. Developers use this unit when the code an agent writes must run on
their own infrastructure (a laptop, a server, a Kubernetes cluster) and they
need to know how strongly that code is isolated from the host.

Plimsoll reports an isolation tier for every sandbox: `container` for plain
Docker, `kernel` for Docker under gVisor, and `vm` for a microVM. The
environment refuses to open a sandbox below the `minimum_isolation` you set, so
a misconfigured daemon fails before any code runs instead of running it behind
a weaker boundary. The tier is the provider's configuration and a startup check
by the daemon, not an attestation made at run time.

The environment is used as a backend for an `EnvironmentToolset` or a
`SkillToolset`. The toolset turns the agent's requests into commands and file
operations, and the environment sends each one to the same Plimsoll session.

## Get started

The following example gives an agent a Plimsoll sandbox through an
`EnvironmentToolset`. The daemon's address and token come from the
`PLIMSOLL_URL` and `PLIMSOLL_TOKEN` environment variables.

```python
from google.adk import Agent
from google.adk.integrations.plimsoll import PlimsollEnvironment
from google.adk.tools.environment import EnvironmentToolset

environment = PlimsollEnvironment(minimum_isolation="kernel")

root_agent = Agent(
    name="coder",
    description="An agent that writes and runs code in a Plimsoll sandbox.",
    instruction=(
        "You have a Linux sandbox with no network access. Write scripts there,"
        " run them, and report their output."
    ),
    tools=[EnvironmentToolset(environment=environment)],
)
```

## How it works

Constructing the environment contacts nothing. When `initialize` is called, the
unit asks the daemon whether it offers sessions, opens one at the requested
floor, checks the tier the session states, and learns the working directory
(`/work` on Docker). The `working_dir` property is `None` until then, because
the daemon's provider decides the path.

Every `execute`, `read_file` and `write_file` is a call of that one session, so
files persist between calls. Each command runs in a new `sh` in the working
directory: a `cd` or an `export` does not carry over to the next command, and a
process left running in the background is stopped when its call ends. Relative
paths resolve against the working directory; `..` out of it is refused.

`execute` takes a `timeout` in seconds, capped at the daemon's own limit.
`None` gives the daemon's limit, or the environment's `timeout` when the daemon
states none; `0` sends the shortest budget the daemon takes, one millisecond; a
negative or non-finite value raises `ValueError`. The toolset's `Execute` tool
gives every command 30 seconds. A command that runs out of time comes back as a
result with `timed_out` set, not as an error.

The client checks a run record on every answer: digests of the request and the
result that the daemon states. A call that came back without a usable answer
raises a `RuntimeError` that says the command may have run, and a refusal made
before anything ran is raised as the client's error with its not-dispatched
reason. `close` closes the session, and the sandbox and its files are gone.
A `close` that fails keeps the session, so calling `close` again retries it.
After `close`, `initialize` raises `RuntimeError` instead of opening a new,
empty sandbox, because ADK calls `initialize` before every tool call and the
agent would take an empty sandbox for the one holding its files. Pass
`reopen_after_close=True` to open a new one instead.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `url` | `str \| None` | `None` | The daemon's address; `PLIMSOLL_URL`, then `http://127.0.0.1:8746`. |
| `token` | `str \| None` | `None` | The caller token; `PLIMSOLL_TOKEN`. |
| `client` | `Client \| None` | `None` | A `plimsoll_client.Client` to use instead of `url` and `token`. |
| `minimum_isolation` | `str` | `"kernel"` | The weakest tier accepted: `process`, `container`, `kernel` or `vm`. |
| `timeout` | `float` | `120.0` | Seconds a file call may take, capped at the daemon's limit; also a command's budget when it has none and the daemon states no limit. |
| `env_vars` | `dict[str, str] \| None` | `None` | Variables added to every command's environment. |
| `lifetime` | `float \| None` | `None` | Seconds the session may live; only shortens the daemon's limit. |
| `idle_timeout` | `float \| None` | `None` | Seconds the session may sit unused; only shortens the daemon's limit. |
| `owner` | `str \| None` | `None` | The end user the session is for, which a daemon can cap. |
| `reopen_after_close` | `bool` | `False` | Whether `initialize` after `close` opens a new, empty sandbox instead of raising `RuntimeError`. |

`env_vars` enter the sandbox, so never put a credential there. The `isolation`
property returns the tier the open session states.

## Limitations

The PlimsollEnvironment requires the `plimsoll-client` package, installed with
`pip install google-adk[plimsoll]`, and a Plimsoll daemon with sessions enabled.
The sandbox has no network access, so an agent cannot install packages from the
internet at run time; put them in the daemon's project image instead.

One environment holds one sandbox for every agent run that uses it. Give each
end user an environment of their own. The sandbox is lost when the session
ends (its lifetime passes, or the daemon restarts) and when a call ends
without an answer the client can check, as a call cancelled after it was sent
does: ADK cancels the other calls of a parallel batch when one of them fails.
The call that found it raises its own error, the environment closes the
session, and every later call raises `RuntimeError` saying the sandbox was
lost, even with `reopen_after_close=True`, until `close` is called. The
environment does not replace the sandbox on its own, because an agent would
take an empty sandbox for the one holding its files. To go on, give the agent
a new environment, or call `close` and, with `reopen_after_close=True`, let
the next call open a new, empty sandbox.

Don't share one environment between Runners that close on their own. ADK
closes an environment when a Runner that holds it closes, and an `AgentTool`
runs its agent in a nested `Runner` that closes when the agent finishes: an
environment shared by an agent and the agent it wraps is closed in the middle
of the wrapping agent's turn, and its next tool call raises `RuntimeError`.
Give the wrapped agent an environment of its own.

`adk eval` runs each eval case in a Runner of its own, several at once, over
one agent, so an agent that holds one environment has it closed when the first
case ends, and the next case raises `RuntimeError`. The sample below gives each
ADK session an environment of its own and closes only the environments of
sessions whose agent run has ended; [its README](../../../../../contributing/samples/environment_and_skills/plimsoll_environment/README.md)
says what that costs.

## Related samples

- [plimsoll_environment](../../../../../contributing/samples/environment_and_skills/plimsoll_environment/agent.py) - A coding agent that runs its scripts in a Plimsoll sandbox, one for each ADK session, so `adk eval` can run it.
