# E2BEnvironment

E2BEnvironment provides a persistent remote workspace backed by an E2B sandbox. It
enables agents to execute shell commands, manage files, and install software in
an isolated environment.

## Introduction

The E2BEnvironment class creates a dedicated remote sandbox for code execution
and file manipulation. Developers use this unit when they need to run untrusted
code, perform data analysis with complex Python dependencies, or provide a
persistent filesystem for an agent that exceeds the local host's capabilities.

The environment is typically used as a backend for an `EnvironmentToolset` or a
`SkillToolset`. These tools allow an agent to interact with the sandbox through
natural language instructions, translating them into shell commands or file
operations. The E2BEnvironment handles the underlying communication with the E2B
API, including sandbox lifecycle management and automatic keepalive signals.

## Get started

The following example demonstrates how to configure an E2BEnvironment and
provide it to an agent through an `EnvironmentToolset`. The agent can then use
the sandbox to download data and run Python scripts.

```python
from google.adk import Agent
from google.adk.integrations.e2b import E2BEnvironment
from google.adk.tools.environment import EnvironmentToolset

# Initialize the environment with a custom timeout
environment = E2BEnvironment(timeout=600)

# The agent uses the environment via a toolset
root_agent = Agent(
    name="data_analyst",
    description="An agent that analyzes data in a remote sandbox.",
    instruction=(
        "You have access to a remote Linux sandbox. Use it to download "
        "datasets, install necessary Python packages, and run analysis scripts."
    ),
    tools=[EnvironmentToolset(environment=environment)],
)
```

## How it works

E2BEnvironment manages a remote virtual machine instance. When the `initialize`
method is called, the unit requests a new sandbox from the E2B service. The
sandbox remains active for a duration specified by the `timeout` parameter.

The environment implements a keepalive mechanism. Every operation, such as
calling `execute`, `read_file`, or `write_file`, resets the time-to-live (TTL)
counter of the sandbox. This ensures that an actively used workspace does not
expire during a long-running task. If the sandbox does expire due to inactivity,
the environment transparently recreates a fresh sandbox on the next operation.
However, any state stored in the previous sandbox, such as installed packages or
unsaved files, is lost when recreation occurs.

All relative file paths provided to `read_file` or `write_file` are resolved
against the sandbox home directory located at `/home/user`. The `working_dir`
property returns this path as a `Path` object, but it is only accessible after
the environment has been initialized.

## Configuration options

The following options control the behavior and identity of the E2B sandbox:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `str` | `"base"` | The E2B template name or ID used to create the sandbox. |
| `timeout` | `int` | `300` | The sandbox time-to-live in seconds, reset on every operation. |
| `api_key` | `str \| None` | `None` | The E2B API key for authentication. |
| `env_vars` | `dict[str, str] \| None` | `None` | Environment variables set inside the sandbox. |

The `image` option allows the use of custom E2B templates that may contain
pre-installed software or specific configurations. If no image is specified, the
environment uses the standard E2B "base" template.

The `timeout` value determines how long the sandbox stays alive while idle. A
shorter timeout reduces credit consumption but increases the risk of losing
state if the agent pauses for too long between steps.

If the `api_key` is not provided directly in the constructor, the unit attempts
to read it from the `E2B_API_KEY` environment variable on the host machine.

## Advanced applications

### On-demand software installation

Because the E2B sandbox provides a full shell, agents can install software
packages at runtime. This is useful for tasks that require libraries not
included in the default image.

```python
from google.adk.integrations.e2b import E2BEnvironment

async def setup_custom_env(env: E2BEnvironment):
    # Install a specific version of pandas
    await env.execute("pip install pandas==2.2.0")
    
    # Verify the installation
    result = await env.execute("python -c 'import pandas; print(pandas.__version__)'")
    print(f"Installed version: {result.stdout}")
```

This pattern allows the agent to adapt its environment to the specific
requirements of a user's request without requiring a custom E2B template for
every possible scenario.

## Limitations

The E2BEnvironment requires the `e2b` Python package, which must be installed
using `pip install google-adk[e2b]`.

The primary limitation of the environment is the volatility of its state. While
the TTL is extended during use, a genuine period of inactivity exceeding the
`timeout` value results in the sandbox being reclaimed by E2B. The environment
automatically recreates the sandbox when the next call is made, but the new
instance starts from the original template state. Any files created or packages
installed in the expired sandbox are not preserved.

The `working_dir` property is not available until `initialize` is called.
Accessing it before initialization results in a `RuntimeError`.

## Related samples

- [e2b_env_skill_toolset](../../../../../contributing/samples/environment_and_skills/e2b_env_skill_toolset/agent.py) - Demonstrates using E2BEnvironment with a SkillToolset to run scripts.
- [e2b_environment](../../../../../contributing/samples/environment_and_skills/e2b_environment/agent.py) - Shows a data analysis agent performing tasks inside an E2B sandbox.
