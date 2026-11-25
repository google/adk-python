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

The client checks a run record on every answer: digests of the request and the
result that the daemon states. A call that came back without a usable answer
raises a `RuntimeError` that says the command may have run, and a refusal made
before anything ran is raised as the client's error with its not-dispatched
reason. `close` closes the session, and the sandbox and its files are gone.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `url` | `str \| None` | `None` | The daemon's address; `PLIMSOLL_URL`, then `http://127.0.0.1:8746`. |
| `token` | `str \| None` | `None` | The caller token; `PLIMSOLL_TOKEN`. |
| `client` | `Client \| None` | `None` | A `plimsoll_client.Client` to use instead of `url` and `token`. |
| `minimum_isolation` | `str` | `"kernel"` | The weakest tier accepted: `process`, `container`, `kernel` or `vm`. |
| `timeout` | `float` | `120.0` | Seconds a file call, or a command with no timeout of its own, may take. |
| `env_vars` | `dict[str, str] \| None` | `None` | Variables added to every command's environment. |
| `lifetime` | `float \| None` | `None` | Seconds the session may live; only shortens the daemon's limit. |
| `idle_timeout` | `float \| None` | `None` | Seconds the session may sit unused; only shortens the daemon's limit. |
| `owner` | `str \| None` | `None` | The end user the session is for, which a daemon can cap. |

`env_vars` enter the sandbox, so never put a credential there. The `isolation`
property returns the tier the open session states.

## Limitations

The PlimsollEnvironment requires the `plimsoll-client` package, installed with
`pip install google-adk[plimsoll]`, and a Plimsoll daemon with sessions enabled.
The sandbox has no network access, so an agent cannot install packages from the
internet at run time; put them in the daemon's project image instead.

One environment holds one sandbox for every agent run that uses it. Give each
end user an environment of their own. When the session ends (its lifetime
passes, or the daemon restarts), every call raises
`plimsoll_client.SessionEndedError`. The environment does not replace the
sandbox on its own, because an agent would take an empty sandbox for the one
holding its files; a `close` followed by `initialize` opens a new one.

## Related samples

- [plimsoll_environment](../../../../../contributing/samples/environment_and_skills/plimsoll_environment/agent.py) - A coding agent that runs its scripts in a Plimsoll sandbox.
