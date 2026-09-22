# DaytonaEnvironment

`DaytonaEnvironment` gives an agent a remote Linux sandbox to work in: it can
run shell commands, install packages, and keep files there, none of it on your
machine. It is the second of ADK's two hosted-sandbox backends, and it differs
from the first in ways that change how you write the agent.

Before any of it works you need a Daytona account and credentials, or a
self-hosted Daytona you point it at, because this is a third-party hosted
service rather than something that runs locally. Sandbox time is billed to that
account, and an abandoned sandbox keeps costing money until Daytona's own
auto-stop fires.

## Introduction

An agent that writes and runs code needs somewhere to do it. `LocalEnvironment`
gives it your process's user account, your environment variables and your
network. That is fine while you are developing, and wrong the moment the code
being run is whatever a model has produced.

Daytona is a hosted sandbox service. `DaytonaEnvironment` implements ADK's
`BaseEnvironment` interface on top of it, so switching from local to sandboxed
is one constructor argument in the toolset. The agent gets a full machine:
a writable working directory, internet access, and enough privilege to
`pip install` and `apt install` whatever the task needs.

The other backend,
[`E2BEnvironment`](../../e2b/e2b_environment/index.md), covers the same ground.
They are close enough that you can swap one for the other by changing the
import, and different enough in three places that the swap shows up in the code
you write.

## Get started

Three steps get an agent working in a sandbox.

1.  Create a Daytona account and obtain credentials, or stand up a self-hosted
    Daytona and point the environment at it.

2.  Install the extra:

    ```bash
    pip install "google-adk[daytona]"
    ```

3.  Hand the environment to the environment toolset. If you leave `api_key`
    unset, the Daytona SDK reads its own credentials from the environment, in
    whatever way [its documentation](https://www.daytona.io/docs) describes:

    ```python
    from google.adk.agents import Agent
    from google.adk.integrations.daytona import DaytonaEnvironment
    from google.adk.tools.environment import EnvironmentToolset

    root_agent = Agent(
        name="data_analysis_agent",
        description="Downloads public datasets and analyzes them in a sandbox.",
        instruction=(
            "You work inside an isolated remote sandbox with internet access, so"
            " you can safely download data and run Python. Install what you need"
            " on demand, write a script rather than guessing, and read the error"
            " output when a command fails."
        ),
        tools=[EnvironmentToolset(environment=DaytonaEnvironment())],
    )
    ```

`EnvironmentToolset` calls `initialize()` the first time it is asked for its
tools, which happens while the request to the model is being prepared. The
sandbox is created then, so if your agent spends its first turn talking to the
user rather than running a command, the sandbox is already alive and the clock
is already running.

**The working directory is `/workspaces`**, so a relative path resolves against
it. If you hard-code either path in a script, the script does not survive a
switch between backends, because E2B's working directory is `/home/user`.

If you construct an environment and drive it yourself rather than handing it to
a toolset, call `initialize()` first. Until you do, every method raises
`RuntimeError: Sandbox is not started. Call initialize() first.`, including the
`working_dir` property.

## How it differs from E2B

Five things differ between the two backends, and three of those change how you
write your code. Of the three, the empty `stderr` is the one most often
missed.

| | `DaytonaEnvironment` | `E2BEnvironment` |
| :--- | :--- | :--- |
| Working directory | `/workspaces` | `/home/user` |
| When the sandbox expires | Operations fail | A fresh empty sandbox is created silently |
| `stderr` on a failed command | Always empty; Daytona folds it into `stdout` | Populated |
| `write_file` to a new directory | ADK creates the parent directories first | ADK passes the path straight to the SDK |
| `timeout` granularity | Rounded down to whole minutes | Seconds |

**Expiry.** E2B's keepalive recreates a lost sandbox, which loses your files
without telling you. `DaytonaEnvironment` calls `refresh_activity()` before
every operation to push the deadline out, but it has no recreate path. The two
failure modes trade against each other. Work fails loudly against a stopped
Daytona sandbox, so the cause is visible and the conversation stops there,
where E2B carries on in an empty replacement that looks like the old one.

**`stderr` is always empty.** Daytona's `process.exec` combines the two streams,
so `ExecutionResult.stderr` is set to `""` unconditionally and the error text is
in `stdout`. Any code you have that reads `result.stderr` to decide whether a
command failed reads an empty string on every failure. Check `exit_code`
instead, which is what `BaseEnvironment` says to do anyway.

**Parent directories.** `write_file("reports/2026/q1.md", ...)` creates
`reports` and `reports/2026` one level at a time before uploading.
`E2BEnvironment` hands the path to its SDK unchanged and does no such
preparation.

## How it works

Two behaviors account for most of the surprises with this backend: when the
sandbox exists, and what a command result contains.

### The sandbox lifecycle

`initialize()` creates one sandbox and returns; calling it again is a no-op
while a sandbox exists. `close()` deletes the sandbox and closes the underlying
HTTP client, which is what stops the process leaking sockets across repeated
create-and-close cycles.

Between those two, every operation first calls `refresh_activity()` on the
sandbox. That resets Daytona's auto-stop timer, so an actively used workspace
does not stop underneath the agent. The timeout is really an idle timeout:
`timeout=300` means "five minutes with no tool call", not "five minutes of
work".

The sandbox is created with `auto_delete_interval=0`; consult
[Daytona's documentation](https://www.daytona.io/docs) for what that value means
for a stopped sandbox in your account. There is no reconnection from ADK's side:
the sandbox id is not exposed and not persisted, so a process restart abandons
the running sandbox.

### Command results

`execute` returns the same `ExecutionResult` every environment returns, and a
non-zero exit code is a normal result rather than an exception. `exit_code`
comes straight from Daytona, `stdout` holds the combined output, and `stderr` is
always `""`.

A command that exceeds its timeout comes back with `timed_out=True` and
`exit_code=-1`, and **both `stdout` and `stderr` are empty**, because output
produced before the timeout is not recovered. The per-command `timeout`
argument and the constructor's sandbox-level `timeout` are different things
that happen to share a name, and the per-command one defaults to the sandbox
one when you leave it out.

Anything that is not a Daytona timeout error propagates. Unlike E2B, which
catches its client's exceptions and turns them into results, this environment
re-raises, so a transport failure reaches your code as an exception rather than
as `exit_code=-1`.

`read_file` maps Daytona's `DaytonaNotFoundError` to a plain `FileNotFoundError`,
so callers do not need to know about the Daytona exception types.

## Configuration options

All five constructor arguments are keyword-only.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `str \| Image \| None` | `None` | Daytona image or template the sandbox is built from. |
| `timeout` | `int` | `300` | Idle timeout in seconds, reset on every operation. |
| `api_key` | `str \| None` | `None` | Daytona API key. Falls back to the SDK's own environment lookup. |
| `api_url` | `str \| None` | `None` | Daytona API URL. Defaults to Daytona Cloud. |
| `env_vars` | `dict[str, str] \| None` | `None` | Environment variables set inside the sandbox. |

`image` chooses between two ways of creating a sandbox. Leave it `None` and the
sandbox is created from Daytona's default Python snapshot, which is the fast
path and carries Python and the usual command-line tools. Set it and the sandbox
is built from that image instead. Build your own when the agent keeps installing
the same packages: baking them in turns a minute of `pip install` at the start
of every conversation into nothing.

`timeout` is the cost control and the interruption risk in one number, pulling
in opposite directions. It caps what an abandoned conversation can spend, and it
decides how long a workspace survives an idle user. **It is converted to whole
minutes**: the sandbox's auto-stop interval is `timeout // 60`, with a floor of
one minute for any positive value. So `timeout=90` and `timeout=119` both mean
one minute, and `timeout=30` also means one minute rather than thirty seconds.
Pass a multiple of 60 if you want the number you wrote.

`api_key` and `api_url` are both optional, and omitting them is the usual
arrangement, because the Daytona SDK reads its own configuration from the
environment. Pass them explicitly when one process serves several tenants with
separate Daytona accounts, or when you run a self-hosted Daytona.

`env_vars` is applied at sandbox creation. There is no way to change it
afterwards short of `close()` and `initialize()`.

## Advanced applications

Two patterns matter once an agent does real work in the sandbox: sharing one
sandbox with a skill toolset, and getting results out before the sandbox
stops.

### Share the sandbox with a skill toolset

An agent that has both the environment toolset and a skill toolset should be
given the same environment object for both, so a skill's scripts and the agent's
own commands see one filesystem:

```python
environment = DaytonaEnvironment(timeout=1800)
skills = load_skills_from_dir(pathlib.Path(__file__).parent / "skills")

root_agent = Agent(
    name="analyst",
    description="Analyzes data using skills that ship executable scripts.",
    instruction="Use the available skills; fall back to your own scripts.",
    tools=[
        EnvironmentToolset(environment=environment),
        SkillToolset(skills=skills, environment=environment),
    ],
)
```

### Get results out before the sandbox goes

The sandbox is the wrong place for anything the user asked for. Read the file
back and save it as an artifact in the same turn that produced it:

```python
async def save_report(tool_context: ToolContext) -> dict[str, str]:
  """Copies report.md out of the sandbox and stores it as an artifact."""
  content = await environment.read_file("report.md")
  await tool_context.save_artifact(
      "report.md", types.Part.from_bytes(data=content, mime_type="text/markdown")
  )
  return {"status": "saved"}
```

## Limitations

*   **`stderr` is never populated.** Daytona folds the error text into
    `stdout`, so read `exit_code` and `stdout` instead.
*   **`timeout` is rounded down to whole minutes**, so any value under 120
    seconds behaves as 60.
*   **`close()` is the only thing that removes the sandbox.** A
    `DaytonaEnvironment` that is garbage-collected without `close()` leaves the
    sandbox running until Daytona's own auto-stop fires, and billing with it. It
    also leaves the HTTP client open.
*   **No reconnection.** One sandbox per environment object, and a process
    restart abandons it.
*   **A failed `mkdir` inside `write_file` is swallowed.** The parent-directory
    loop catches every exception and only continues explicitly for a conflict,
    so a permission error while creating a directory is neither raised nor
    logged. The upload that follows fails with a less specific message.
*   **The feature is flag-gated.** The class is registered under the
    `DAYTONA_ENVIRONMENT` feature flag. It defaults to on and warns once per
    process that it is experimental; setting `ADK_DISABLE_DAYTONA_ENVIRONMENT=1`
    makes the constructor raise `RuntimeError`.
*   **The extra is not installed by default.** Without it, `initialize()` raises
    `ImportError: The daytona package is required to use DaytonaEnvironment.
    Install it with `pip install google-adk[daytona]`.` Note that this is raised
    at `initialize()`, not at import or construction, so an agent module that
    defines a `DaytonaEnvironment` imports cleanly on a machine that cannot run
    it.
*   **It costs money and needs the network.** Unlike `LocalEnvironment` there is
    a third party in the loop, with an account, a quota, and latency on every
    single file read.

## Related samples

*   [Daytona environment](../../../../../contributing/samples/environment_and_skills/daytona_environment/agent.py)
    is a data-analysis agent that downloads a public dataset and analyzes it in
    the sandbox.
*   [E2B environment](../../../../../contributing/samples/environment_and_skills/e2b_environment/agent.py)
    is the same agent shape against the other remote-sandbox backend.

## Related guides

*   [E2BEnvironment](../../e2b/e2b_environment/index.md) is the other hosted
    sandbox, with the differences listed above.
