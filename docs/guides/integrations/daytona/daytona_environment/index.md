# DaytonaEnvironment

DaytonaEnvironment provides a persistent remote workspace for code execution
and file management within a Daytona sandbox. It allows agents to run shell
commands and manipulate files in an isolated environment that is separate
from the host system.

## Introduction

The unit manages remote infrastructure through the Daytona API, providing a
consistent environment for agents that need to perform data analysis,
software development, or other tasks requiring a shell. It solves the
problem of local execution risks by moving all operations to a remote,
ephemeral container.

Developers typically use DaytonaEnvironment in conjunction with the
EnvironmentToolset. This combination allows an agent to treat the remote
sandbox as its own local file system and terminal. The environment handles
the complexities of sandbox provisioning, file uploads, and process
execution, while ensuring that resources are cleaned up when no longer
needed.

## Get started

To use a remote sandbox for code execution, define a DaytonaEnvironment and
provide it to an EnvironmentToolset.

```python
from google.adk import Agent
from google.adk.integrations.daytona import DaytonaEnvironment
from google.adk.tools.environment import EnvironmentToolset

# Configure the remote environment with a specific timeout.
environment = DaytonaEnvironment(timeout=600)

# The EnvironmentToolset provides the agent with file and shell access.
agent = Agent(
    name="data_assistant",
    instruction="Analyze data files by writing and running Python scripts.",
    tools=[EnvironmentToolset(environment=environment)],
)
```

## How it works

The unit manages the lifecycle of a remote Daytona sandbox. When you call
the initialize method, the environment contacts the Daytona API to
provision a new workspace. Subsequent calls to execute, read_file, and
write_file operate within that specific workspace. The close method
ensures the remote resources are deleted and the underlying HTTP sessions
are closed to prevent resource leaks.

The environment resolves all relative file paths against /workspaces, which
serves as the persistent home directory inside the sandbox. When writing
files to nested paths, the environment automatically creates missing
parent directories. The execute method runs shell commands and returns an
ExecutionResult. Because Daytona combines standard output and standard
error into a single stream, the stderr attribute of the result is always
empty.

## Configuration options

The following options control how the Daytona sandbox is provisioned and
accessed.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `str \| Image \| None` | `None` | The Daytona template or image name used to create the sandbox. |
| `timeout` | `int` | `300` | The sandbox time-to-live and execution timeout in seconds. |
| `api_key` | `str \| None` | `None` | The Daytona API key, which defaults to the environment variable if not provided. |
| `api_url` | `str \| None` | `None` | The Daytona API URL, which defaults to the Daytona Cloud API. |
| `env_vars` | `dict[str, str] \| None` | `None` | Environment variables to set inside the sandbox. |

The image parameter determines the base environment for the sandbox. If it
is not provided, the environment defaults to a Python-based snapshot. The
timeout value sets the interval after which the sandbox will automatically
stop to save resources.

Use api_key and api_url when you need to connect to a specific Daytona
instance or when you are not using the standard environment variables for
authentication. The env_vars dictionary allows you to inject configuration
or secrets directly into the sandbox shell.

## Advanced applications

If you require specific system libraries or pre-installed tools, you can
provide a custom image name. The custom image allows the sandbox to start
with a tailored environment instead of the default Python snapshot. You
can also configure api_url to point the environment toward a self-hosted
Daytona instance instead of the default cloud service.

## Limitations

The DaytonaEnvironment is an experimental feature and requires the daytona
Python package, which you can install using the daytona extra. Since the
sandbox is remote, every operation incurs network latency, and the
environment requires valid credentials and an active internet connection.
Initialization can take several seconds as the remote infrastructure is
provisioned.

## Related samples

- [daytona_environment](../../../../../contributing/samples/environment_and_skills/daytona_environment/agent.py) - A data analysis agent that runs Python in a Daytona remote sandbox.
