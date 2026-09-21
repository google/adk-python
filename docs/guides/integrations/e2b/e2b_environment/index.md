# E2BEnvironment

`E2BEnvironment` gives an agent a remote Linux sandbox to work in: it can run
shell commands, install packages, and keep files there, none of it on your
machine. The sandbox has a time-to-live, and when it runs out the workspace is
gone: the next operation continues in a brand-new empty sandbox without raising
anything.

Before any of it works you need an E2B account and an API key, because E2B is a
third-party hosted service rather than something that runs locally. Sandbox
time is billed to that account, so a conversation nobody closes keeps costing
money until its time-to-live runs out.

## Introduction

An agent that writes and runs code needs somewhere to do it. `LocalEnvironment`
gives it your process's user account, your environment variables and your
network. That is fine while you are developing, and wrong the moment the code
being run is whatever a model has produced.

E2B is a hosted sandbox service. `E2BEnvironment` implements ADK's
`BaseEnvironment` interface on top of it, so the swap from local to sandboxed
is one constructor argument in the toolset. The agent gets a full machine:
a writable home directory, internet access, and root enough to `pip install`
and `apt install` whatever the task needs.

The trade is that the workspace is remote, metered, and not permanent.

## Get started

Three steps get an agent working in a sandbox.

1.  Create an E2B account and obtain an API key.

2.  Install the extra and set the key in your environment:

    ```bash
    pip install "google-adk[e2b]"
    export E2B_API_KEY=...
    ```

3.  Hand the environment to the environment toolset:

    ```python
    from google.adk.agents import Agent
    from google.adk.integrations.e2b import E2BEnvironment
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
        tools=[EnvironmentToolset(environment=E2BEnvironment())],
    )
    ```

`EnvironmentToolset` calls `initialize()` the first time it is asked for its
tools, which happens while the request to the model is being prepared. The
sandbox is created then, so if your agent spends its first turn talking to the
user rather than running a command, the sandbox is already alive and the clock
is already running.

If you give a relative path, it resolves against the sandbox's working
directory, which is `/home/user`.

## How it works

Three behaviors account for most of the surprises with this backend: when the
sandbox exists, what happens when it stops existing, and what a command result
contains.

### The sandbox lifecycle

`initialize()` creates one sandbox from the template named by `image` and
returns. `close()` kills it. Between those two, every operation goes through a
keepalive step before it does anything else:

1.  Ask the sandbox whether it is still running.
2.  If it is, reset its time-to-live to `timeout` seconds.
3.  If it is not, create a replacement.

Step 2 is why an actively used workspace does not expire underneath the agent.
Each call pushes the deadline out again, so the TTL is really an idle timeout:
`timeout=300` means "five minutes with no tool call", not "five minutes of
work".

### What happens when the sandbox does expire

Step 3 is the part to plan around. **When the TTL runs out, the workspace is
gone, and the next operation silently continues in a brand-new empty sandbox.**
Every installed package, every file the agent created, and every bit of shell
state is lost. Replacing the sandbox is what keeps a conversation usable after
a long pause, because the alternative is every later tool call failing until
someone restarts the agent. Nothing raises, which is the price of that
recovery. The only signal is a log line at warning level:

```text
E2B sandbox expired; recreating a fresh sandbox. Workspace state (installed
packages and files) has been lost.
```

From the model's point of view this is worse than an error, because the failure
does not look like one. The agent asks to run `python analyze.py`, gets back
exit code 1 and `python: can't open file 'analyze.py'`, and has no way to tell
"I never wrote that file" from "the machine it was on no longer exists". It
usually rewrites the script, which recovers, and sometimes concludes the task
is impossible, which does not.

Three ways to guard against it:

*   **Set `timeout` above the longest pause you expect** between tool calls.
    The gap that matters is not the agent's thinking time but the human's: a
    conversation where the user reads a result and replies twenty minutes later
    outlives the 300-second default many times over.
*   **Watch for the warning.** It is logged on `google_adk` at `WARNING`, so it
    is visible in any normal logging setup, and it is the only way to tell this
    apart from a confused agent.
*   **Do not treat the sandbox as storage.** Anything the user needs to keep
    should be read back out with `read_file` and saved as an artifact while the
    agent still has it.

### Command results

`execute` returns the same `ExecutionResult` every environment returns, and it
never raises for a failing command: E2B signals a non-zero exit by raising
`CommandExitException`, which the environment catches and turns back into a
result carrying the real exit code, stdout and stderr.

A command that exceeds its `timeout` argument comes back with
`timed_out=True` and `exit_code=-1`, and **both `stdout` and `stderr` are
empty**, because the output produced before the timeout is not recovered. This
per-command `timeout` and the constructor's sandbox-level `timeout` are
different things that happen to share a name.

`read_file` maps E2B's `FileNotFoundException` to a plain `FileNotFoundError`,
so callers do not need to know about the E2B exception types.

## Configuration options

All four constructor arguments are keyword-only.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `str` | `"base"` | E2B template name or ID the sandbox is created from. |
| `timeout` | `int` | `300` | Sandbox time-to-live in seconds, reset on every operation. |
| `api_key` | `str \| None` | `None` | E2B API key. Falls back to `E2B_API_KEY`. |
| `env_vars` | `dict[str, str] \| None` | `None` | Environment variables set inside the sandbox. |

`image` defaults to E2B's public `base` template, which every account can use
and which carries Python and the usual command-line tools. Build your own
template when the agent keeps installing the same packages: baking them in
turns a minute of `pip install` at the start of every conversation into
nothing, and removes one thing the agent can get wrong.

`timeout` is the cost control and the data-loss risk in one number, pulling in
opposite directions. It caps what an abandoned conversation can spend, and it
decides how long a workspace survives an idle user. Raise it for interactive
agents; leave it low for a batch job that runs unattended and finishes.

`env_vars` is set on the sandbox at creation. Because it is applied at creation
time, a sandbox recreated after expiry gets the same variables, which makes it
the one piece of workspace state that does survive.

`api_key` is read once, at sandbox creation. Leaving it `None` and setting
`E2B_API_KEY` is the usual arrangement; pass it explicitly when one process
serves several tenants with separate E2B accounts.

## Advanced applications

Two patterns matter once an agent does real work in the sandbox: sharing one
sandbox with a skill toolset, and getting results out before the sandbox
expires.

### Share the sandbox with a skill toolset

An agent that has both the environment toolset and a skill toolset should be
given the same environment object for both, so a skill's scripts and the
agent's own commands see one filesystem:

```python
environment = E2BEnvironment(timeout=1800)
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

### Get results out before they expire

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

*   **Expiry is silent and lossy.** The workspace vanishes and the next call
    gets an empty replacement, with a log warning as the only signal.
*   **One sandbox per environment object, and no reconnection.** The sandbox id
    is not exposed and is not persisted, so a process restart abandons the
    running sandbox rather than reattaching to it. It expires on its own TTL.
*   **`close()` is the only thing that kills the sandbox.** An
    `E2BEnvironment` that is garbage-collected without `close()` leaves the
    sandbox running until its TTL expires, and billing with it.
*   **`is_initialized` can be stale.** It is set by `initialize()` and cleared
    by `close()`, and nothing updates it when the sandbox expires, so it
    reports `True` for a sandbox that no longer exists.
*   **The feature is flag-gated.** The class is registered under the
    `E2B_ENVIRONMENT` feature flag. It defaults to on and warns once per
    process that it is experimental; setting `ADK_DISABLE_E2B_ENVIRONMENT=1`
    makes the constructor raise `RuntimeError`.
*   **It costs money and needs the network.** Unlike `LocalEnvironment` there
    is a third party in the loop, with an account, a quota, and latency on
    every single file read.

## Related samples

*   [E2B environment](../../../../../contributing/samples/environment_and_skills/e2b_environment/agent.py)
    is a data-analysis agent that downloads a public dataset and analyzes it in
    the sandbox.
*   [E2B with a skill toolset](../../../../../contributing/samples/environment_and_skills/e2b_env_skill_toolset/agent.py)
    shares one sandbox between the environment tools and the skill scripts.
*   [Daytona environment](../../../../../contributing/samples/environment_and_skills/daytona_environment/agent.py)
    is the same agent shape against the other remote-sandbox backend.

## Related guides

*   [DaytonaEnvironment](../../daytona/daytona_environment/index.md) is the
    other hosted sandbox, with the differences listed above.
