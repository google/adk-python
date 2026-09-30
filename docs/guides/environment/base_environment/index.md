# BaseEnvironment and LocalEnvironment

`BaseEnvironment` is the interface ADK uses for "a place where the agent can run
shell commands and keep files". `LocalEnvironment` is the implementation that
runs those commands as subprocesses on the same machine that hosts your agent.

## Introduction

Two different ADK features need somewhere to put files and run commands. The
environment toolset gives the agent four tools called `Execute`, `ReadFile`,
`WriteFile` and `EditFile`, and the skill toolset needs the same facility
whenever a skill ships executable scripts. Neither of them wants to know about
subprocesses, sandboxes and remote containers, so both take a `BaseEnvironment`
instead and call three methods on it: `execute`, `read_file` and `write_file`,
all relative to its `working_dir`.

That indirection is the entire reason the class exists. One agent definition can
run against a local subprocess while you develop and against an isolated remote
sandbox in production, and the only thing that changes between them is a
constructor argument. ADK ships `LocalEnvironment` along with the
E2B and Daytona environments
under `google.adk.integrations`.

The interface stays deliberately small, covering a working directory, a shell
command, and byte-level file read and write. It is neither a filesystem
abstraction nor a container API, and it will not grow into one.

## Get started

The agent below works in a scratch directory on the local machine:

```python
from google.adk.agents import Agent
from google.adk.environment import LocalEnvironment
from google.adk.tools.environment import EnvironmentToolset

root_agent = Agent(
    name="local_environment_agent",
    description="Runs shell commands and edits files in a scratch directory.",
    instruction=(
        "You have a working directory where you can create files and run"
        " shell commands. Write a script before guessing, and read the error"
        " output when a command fails."
    ),
    tools=[EnvironmentToolset(environment=LocalEnvironment())],
)
```

`EnvironmentToolset` calls `initialize()` for you, which is why nothing above has
to manage the lifecycle. If you drive an environment yourself, that job becomes
yours:

```python
env = LocalEnvironment()
await env.initialize()

await env.write_file(Path("notes/hello.txt"), "hi there")
print(await env.read_file(Path("notes/hello.txt")))  # b'hi there'

result = await env.execute("cat notes/hello.txt")
print(result.exit_code, result.stdout)  # 0 hi there

await env.close()
```

When you leave `working_dir` out, `initialize()` creates a temporary directory
named `adk_workspace_*`, and `close()` later deletes it along with everything
inside it.

## How it works

Three things are worth understanding before you rely on an environment, and each
one has a section below:

*   When the working directory comes into existence, and when it goes away.
*   What an `execute` call gives you back when the command fails.
*   How much of that a `LocalEnvironment` actually protects you from.

### The lifecycle

An environment moves through four steps, always in this order.

1.  **Construct.** The constructor stores settings and touches nothing else. No
    directory exists yet and no process has been started.
2.  **`initialize()`.** Creates the working directory, or connects to the remote
    workspace. Implementations are expected to make this idempotent, so calling
    it twice is safe.
3.  **`execute`, `read_file`, `write_file`.** The working phase.
4.  **`close()`.** Releases whatever `initialize` acquired. For
    `LocalEnvironment` this deletes an auto-created temporary directory and
    leaves a directory you supplied alone.

Both `initialize` and `close` have no-op default implementations on the base
class, so an environment that needs neither can leave them alone. The other four
members, `working_dir`, `execute`, `read_file` and `write_file`, are abstract,
and a subclass that does not provide all of them cannot be instantiated at all.

`is_initialized` is a plain boolean property that each implementation sets for
itself. Nothing in the framework checks it, so read it as a status report rather
than as a guard that will stop you. Calling `execute` before `initialize` on
`LocalEnvironment` raises `RuntimeError: `working_dir` is not set. Call
initialize() first.`

### ExecutionResult

Every `execute` call returns an `ExecutionResult` dataclass rather than raising,
whatever happened to the command:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `exit_code` | `int` | `0` | Process exit status. |
| `stdout` | `str` | `""` | Captured standard output. |
| `stderr` | `str` | `""` | Captured standard error. |
| `timed_out` | `bool` | `False` | Whether the command exceeded `timeout`. |

**A non-zero exit code is a normal result, not an exception.** A command that
fails comes back with its exit code set and its `stderr` filled in, and noticing
that is the caller's job. In practice the caller is the tool, and through the
tool it is the model. Check `exit_code` before you trust `stdout`.

`timed_out` is the one field you have to read together with another. When
`LocalEnvironment` times a command out, it kills the process, so `exit_code`
comes back as `-9`, the negative of `SIGKILL`, rather than as anything the
command chose for itself, and `stdout` holds only what had been flushed by that
point.

### LocalEnvironment specifics

Commands run through `asyncio.create_subprocess_shell`, which means the shell
interprets the string you pass. Pipes, `&&` and redirection all work, and so
does anything else a shell would do with that text. The subprocess runs with
`cwd` set to the working directory and an environment made of `os.environ` with
`env_vars` merged over the top.

Both file methods resolve a relative path against the working directory, then
check that the result is still inside it. A path that climbs out raises
`ValueError: Path escapes working directory: <path>` before any I/O happens at
all. `write_file` creates parent directories as needed, accepts either `str` or
`bytes`, and writes text as UTF-8 with newline translation off, while
`read_file` always hands back `bytes`.

Both file methods run on a worker thread through `asyncio.to_thread`, so reading
or writing a large file will not block the event loop.

## Configuration options

`LocalEnvironment` takes two keyword-only arguments, and the first of them
decides more than it looks like it does.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `working_dir` | `Path \| None` | `None` | Directory the agent works in. A temporary one is created and later deleted when omitted. |
| `env_vars` | `dict[str, str] \| None` | `None` | Extra variables merged into the subprocess environment. |

`working_dir` settles ownership as well as location. Supply a directory and it is
created if it does not exist (`makedirs(exist_ok=True)`) and left in place by
`close()`, which is what you want whenever the output is the point, such as a
report the agent produced or a repository it edited. Leave the argument out and
you get a throwaway directory under the system temporary location that vanishes
with the environment, which is what you want for a scratch workspace.

`env_vars` is merged over a copy of the parent process environment rather than
replacing it, so the subprocess inherits everything the host process has,
credentials in the environment included. There is no way to start from an empty
environment.

## Advanced applications

The first section below is for backing the interface with something other than a
subprocess, and the second is for the case where two toolsets have to agree on
one working directory.

### Write your own environment

Writing an environment means implementing the four abstract members. The
lifecycle hooks are optional, and the one detail worth getting right is that
`initialize` and `close` may both be called more than once.

```python
class MemoryEnvironment(BaseEnvironment):
  """Keeps files in a dict; refuses to run commands."""

  def __init__(self):
    self._files: dict[str, bytes] = {}

  @property
  def working_dir(self) -> Path:
    return Path("/")

  async def execute(self, command: str, *, timeout: float | None = None):
    return ExecutionResult(exit_code=1, stderr="This environment has no shell.")

  async def read_file(self, path: Path) -> bytes:
    try:
      return self._files[str(path)]
    except KeyError as e:
      raise FileNotFoundError(path) from e

  async def write_file(self, path: Path, content: str | bytes) -> None:
    self._files[str(path)] = (
        content.encode("utf-8") if isinstance(content, str) else content
    )
```

There are two conventions the built-in tools rely on, so keep them even in an
environment of your own. `read_file` raises `FileNotFoundError` for a missing
file rather than returning empty bytes, and `execute` reports failure through
`exit_code` rather than by raising.

### Share one environment across a conversation

Construct the environment once and pass the same instance everywhere it is
needed. Both `EnvironmentToolset` and `SkillToolset` take an `environment=`
argument, and handing them the same object means a file the agent writes through
one toolset is there when it looks through the other.

Sharing comes with a catch that is worth understanding before you depend on it.
Each toolset closes the environment in its own `close()`, and neither one checks
whether the other is still using it. Close one toolset and the shared working
directory is deleted out from under the other. Every tool on the surviving
toolset then fails with the same message:

```text
`working_dir` is not set. Call initialize() first.
```

It will not recover on its own either, because it still believes it initialized
the environment already. Either give each toolset an environment of its own, or
close both at the same time.

An environment's lifetime is therefore the lifetime of the object you built, and
it has nothing to do with a session. An environment held in a module-level
variable is shared by every user of the process, which is fine for a single-user
command-line agent and wrong for a server.

## Limitations

*   **`LocalEnvironment` is not a sandbox.** The command runs as your process's
    user, with your process's environment variables, and only *file paths* are
    confined to the working directory. `execute` is not confined at all, so a
    command can read anything that user can read, reach the network, and write
    outside the working directory. Use it where you trust the model's output,
    and use a remote sandbox where you do not.
*   **The path check applies to `read_file` and `write_file` only.** It is a
    check on the resolved path, so it stops `../../etc/passwd`, but it does
    nothing about `execute("cat /etc/passwd")`.
*   **No streaming.** `execute` returns once the command has finished, with the
    whole of stdout and stderr buffered in memory. A long-running command
    produces no output until it exits, and a command that prints a gigabyte
    holds a gigabyte.
*   **`close()` does not stop running work.** All it does is release the
    directory, so a process the agent left running in the background survives
    it.
*   **The classes are experimental.** Import them from `google.adk.environment`
    rather than from any module beneath it, and expect the signatures to be
    able to change.

## Related samples

*   [Local environment](../../../../contributing/samples/environment_and_skills/local_environment/agent.py)
    is an agent with the environment toolset over `LocalEnvironment`.
*   [Local environment with skills](../../../../contributing/samples/environment_and_skills/local_env_skill_toolset/agent.py)
    shares that same environment with a skill toolset.
*   [E2B environment](../../../../contributing/samples/environment_and_skills/e2b_environment/agent.py)
    keeps the agent shape identical and points it at a remote sandbox instead.

## Related guides

*   `E2BEnvironment` is the
    remote sandbox implementation of this interface.
*   [Skill, Frontmatter, and Resources](../../skills/skill/index.md) covers
    skills, whose scripts run inside an environment like this one.
