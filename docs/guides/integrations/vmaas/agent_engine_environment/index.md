# AgentEngineEnvironment

AgentEngineEnvironment gives an agent a remote shell workspace backed by an
Agent Engine sandbox, a gVisor-isolated container on GKE with a persistent
filesystem. The agent runs commands and reads and writes files in the sandbox,
and the sandbox pauses between uses instead of being deleted.

## Introduction

AgentEngineEnvironment implements `BaseEnvironment` on top of Agent Engine
sandboxes. Use it when an agent needs to run code you do not want on your own
machine, install software at runtime, or keep files between turns, and you
want that sandbox to live in your Google Cloud project next to the rest of your
Agent Engine resources.

The environment is the backend for `EnvironmentToolset`, the same way
`E2BEnvironment` and `DaytonaEnvironment` are. The toolset gives the agent its
Execute, ReadFile, EditFile, and WriteFile tools, and the environment carries
each of those calls out in the sandbox.

AgentEngineEnvironment is experimental and gated by the
`AGENT_ENGINE_SANDBOX_ENVIRONMENT` feature, which is on by default. It needs the
Agent Platform SDK, which `pip install google-adk[gcp]` installs, and
credentials that can create Agent Engine resources in your project, such as
Application Default Credentials.

## Get started

The following example gives an agent a sandbox through an `EnvironmentToolset`.
The toolset creates the sandbox the first time the agent needs it.

```python
from google.adk import Agent
from google.adk.integrations.vmaas import AgentEngineEnvironment
from google.adk.tools.environment import EnvironmentToolset

environment = AgentEngineEnvironment(
    project_id="my-project",
    location="us-central1",
    agent_engine_name=(
        "projects/my-project/locations/us-central1/reasoningEngines/1234"
    ),
)

root_agent = Agent(
    name="sandbox_worker",
    description="An agent that runs code in a remote sandbox.",
    instruction=(
        "You have a Linux sandbox. Use it to write files, install packages,"
        " and run scripts."
    ),
    tools=[EnvironmentToolset(environment=environment)],
)
```

Passing `agent_engine_name` is optional. It is shown here because the agent
engine is where the sandbox template lives, and reusing one agent engine is
what lets later runs skip creating that template. The section on configuration
options explains the trade-off.

## How it works

Every sandbox is created from a sandbox template. The first time an
environment needs a sandbox, it looks under its agent engine for an active
shell sandbox template that ADK created earlier, and creates one only if there
is none. Every later sandbox under the same agent engine, from this process or
any other, reuses that template. Creating a template takes much longer than
creating a sandbox from one, anywhere from about half a minute to more than
ten minutes, while creating a sandbox from an existing template takes a few
seconds.

`initialize()` creates one sandbox for the environment instance.
`EnvironmentToolset` calls it the first time it hands the agent its tools, so
an agent that never uses its tools creates no sandbox. Calling `initialize()`
again on an initialized environment does nothing, and concurrent calls create
a single sandbox.

`close()` pauses the sandbox rather than deleting it. A paused sandbox keeps
its files and stops using compute, and the next `initialize()` resumes it with
those files in place. `Runner.close()` closes the toolset, which closes the
environment, so a sandbox pauses when the runner that used it shuts down. If
the sandbox no longer exists by the time `initialize()` resumes it, for
example because its time to live ran out while it was paused, the environment
creates a new sandbox and logs a warning that the old files are gone.

A command sent to a paused sandbox does not fail. The sandbox service starts
resuming the sandbox when the command arrives, and the environment waits until
the sandbox is running, then sends the command again. A sandbox can be paused
behind the environment's back, by another process or by the service, and the
agent's next command still runs.

`execute()` reports a command's standard output, standard error, and exit
code as an `ExecutionResult`. A non-zero exit code is a normal result, not an
exception, so the agent can read the error and react. A command that runs past
its timeout is stopped by the sandbox and reported with `timed_out` set to
`True`. `EnvironmentToolset` gives each command a 30 second timeout.

`read_file()` and `write_file()` move file contents through shell commands in
the sandbox. A relative path resolves against the sandbox working directory,
`/workspace`, which `working_dir` returns once the environment is initialized.
`write_file()` creates any missing parent directories, and `read_file()` raises
`FileNotFoundError` for a path that does not exist.

## Configuration options

The following options select where the sandbox lives and which sandbox the
environment uses:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `project_id` | `str \| None` | `None` | Google Cloud project for the agent engine and sandbox. |
| `location` | `str` | `"us-central1"` | Agent Engine location. |
| `agent_engine_name` | `str \| None` | `None` | Agent engine to create the template and sandbox under. |
| `sandbox_name` | `str \| None` | `None` | Existing shell sandbox to use instead of creating one. |
| `sandbox_template_name` | `str \| None` | `None` | Shell sandbox template to create the sandbox from. |
| `sandbox_ttl_seconds` | `int \| None` | `None` | Time to live of a sandbox the environment creates. |
| `vertexai_client` | `agentplatform.Client \| None` | `None` | Agent Platform client to use instead of creating one. |

When `project_id` is `None`, the client uses the project of your Application
Default Credentials. The `location` must be one where Agent Engine sandboxes
are available, and the agent engine, template, and sandbox all live there.

`agent_engine_name` decides how often a template gets created. Templates belong
to an agent engine, so pointing every process at the same agent engine means
the template is created once and reused from then on. When it is `None`, each
environment instance creates a new agent engine the first time it needs one,
and with it a new template, so every new process waits for template creation
again. A fixed agent engine is worth passing for anything that starts more than
once.

`sandbox_name` attaches the environment to a sandbox that already exists, such
as one a previous process created, and `initialize()` resumes it if it is
paused. The environment never replaces a sandbox you named: if that sandbox no
longer exists, `initialize()` raises `RuntimeError` instead of creating another,
because a silent replacement would hide the loss of its files. The sandbox must
come from a shell sandbox template.

`sandbox_template_name` creates the sandbox from a template you manage, for
example one you created with more CPU or memory through the Agent Platform SDK.
The sandbox is created under the agent engine that owns the template, so the
template's agent engine takes precedence over `agent_engine_name`.

`sandbox_ttl_seconds` bounds how long a sandbox the environment creates can
exist, counting paused time. When it is `None`, the service default applies. A
shorter time to live limits how long an abandoned sandbox lingers, at the cost
of losing the files of a sandbox that stays paused past it.

`vertexai_client` lets you share one configured Agent Platform client, for
example one with custom HTTP options, across environments. When it is `None`,
the environment creates a client from `project_id` and `location` on first use.

## Advanced applications

### Continue work in a later process

The environment's `sandbox_name` property holds the resource name of its
sandbox once `initialize()` has run. Storing that name and passing it back as
`sandbox_name` lets a later process attach to the same sandbox, resume it, and
find the files the earlier process left.

```python
from google.adk.integrations.vmaas import AgentEngineEnvironment

async def run_and_remember(store: dict[str, str]) -> None:
  environment = AgentEngineEnvironment(
      project_id="my-project",
      sandbox_name=store.get("sandbox_name"),
  )
  await environment.initialize()
  await environment.write_file("notes.txt", "step one done")
  store["sandbox_name"] = environment.sandbox_name
  await environment.close()
```

Because the environment never replaces a sandbox you named, catch the
`RuntimeError` from `initialize()` if your application should start over with
a new sandbox once the old one has expired.

### Pause between bursts of work

`close()` pauses the sandbox, so calling it when an agent finishes a task stops
compute use until the next task, and the next `initialize()` resumes the
sandbox in several seconds. Commands also resume a paused sandbox on their own,
so an application can pause a sandbox it is not using without coordinating with
whatever might use it next.

## Limitations

Each environment instance owns one sandbox, so every session that uses the
instance shares that sandbox and its files. An agent served to several users
through one runner gives all of them the same sandbox, so give each user or
session its own environment when their files must stay apart.

`write_file()` sends file contents inside a single shell command, which limits
a write to about 90 KB. A larger write raises `ValueError` before anything is
sent.

The first command on a new or resumed sandbox can take half a minute or
longer, while the service starts routing to the sandbox. Later commands return
in well under a second.

A paused sandbox still counts against its time to live. A sandbox that stays
paused past that time is deleted with its files, and the environment replaces
it with an empty one the next time it initializes.

The `working_dir` property is not available until `initialize()` has run, and
accessing it earlier raises `RuntimeError`.
