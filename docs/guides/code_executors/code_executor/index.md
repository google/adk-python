# BaseCodeExecutor

`BaseCodeExecutor` is the interface an `LlmAgent` uses to run a code block the
model wrote and feed the result back into the conversation. The interesting
decision is not the interface, it is where that code actually runs. On this
machine, inside a container, or in a cloud sandbox are all valid answers, and
picking between them is what the subclass is for.

## Introduction

A model asked for an exact calculation will happily write the Python for it, but
it cannot run that Python itself. Something has to, and where you let it run is
a security decision rather than a plumbing one. The code came out of a model, no
human reviewed it on the way, and whoever influenced the conversation had some
influence over what it says. Run it in the process serving your agent and it
inherits that process's user, its environment variables, and its network access,
so a snippet that loops forever, reads a credential file, or opens a socket is
doing so as you.

`BaseCodeExecutor` puts that whole decision behind one object. The agent hands it
a code block and gets back standard output, standard error, and any files the
code produced, while everything about where the code actually ran stays the
implementation's business. Six implementations ship in
`google.adk.code_executors`, and a seventh lives alongside its integration in
`google.adk.integrations.cloud_run`. What separates them is almost entirely how
much isolation they give you.

## Get started

You turn code execution on by setting `code_executor` on the agent.
`UnsafeLocalCodeExecutor` runs the model's code in a child Python interpreter on
your own machine with no isolation whatsoever. That is a reasonable trade while
you are developing, because there is nothing to install and you are the only
person shaping the prompt, and it stops being reasonable once somebody else can.
The alternatives, and what each of them costs, are under
[Choose an implementation](#choose-an-implementation).

The agent below answers arithmetic by writing Python and running it:

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import UnsafeLocalCodeExecutor

agent = LlmAgent(
    name="calculator_agent",
    instruction="When asked a math problem, write Python code to compute the exact result.",
    code_executor=UnsafeLocalCodeExecutor(timeout_seconds=30),
)
```

Ask that agent "what is 6 times 7?" and it writes `print(6 * 7)` in a Python
block. The user never sees that block, because ADK pulls it out, runs it, and
sends `42` back to the model, which then answers the question in words.

## How it works

Setting `code_executor` is the whole of the wiring, and everything after that
happens on the agent's own turn. Five behaviors decide what the model ends up
seeing.

1. **Extraction.** After the model responds, its text is scanned for the
   delimiters in `code_block_delimiters`, normally ```` ```python ```` or
   ```` ```tool_code ````. The first block becomes a
   `types.Part.from_executable_code`. Everything the model wrote after that
   block is discarded.
2. **Execution.** The code and any session-attached files are wrapped in a
   `CodeExecutionInput` and passed to
   `code_executor.execute_code(invocation_context, code_execution_input)`.
   That method is the one abstract member a subclass has to implement.
3. **Result formatting.** The executor returns a `CodeExecutionResult` holding
   `stdout`, `stderr` and `output_files`, which reaches the model as a
   `types.Part.from_code_execution_result` whose outcome is `OUTCOME_OK` or
   `OUTCOME_FAILED`.
4. **Retry on error.** A non-empty `stderr` counts as a failed run. While the
   number of consecutive failures is within `error_retry_attempts`, the error
   text goes back into the conversation and the model is asked to fix its code.
5. **Replay as text.** On later model calls, the earlier code and result parts
   are rewritten as plain text, so a chat model with no native code-execution
   parts can still read what happened.

## Configuration options

Every option below is defined on the base class, so you can set it on any
implementation:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `optimize_data_file` | `bool` | `False` | Extract and process CSV data files from the model request and attach them to the executor. |
| `stateful` | `bool` | `False` | Whether state and variables persist across multiple code executions within a session. |
| `error_retry_attempts` | `int` | `2` | Number of consecutive execution error retries before giving up. |
| `code_block_delimiters` | `list[tuple[str, str]]` | `[('```tool_code\n', '\n```'), ('```python\n', '\n```')]` | Delimiter pairs used to locate code blocks in model output. |
| `execution_result_delimiters` | `tuple[str, str]` | `('```tool_output\n', '\n```')` | Delimiters used when formatting execution output for text-based model turns. |
| `timeout_seconds` | `int \| None` | `None` | Wall-clock execution timeout in seconds. |

`optimize_data_file` makes the flow search the user's message for `text/csv`
parts, parse them, and put them on `CodeExecutionInput.input_files`. The
generated code can then load the dataset by filename, with no upload code of your
own anywhere. Only three implementations accept it, though.

`stateful` says whether a variable defined in one turn is still around in the
next. A stateless executor starts a fresh process every time, so the model has to
re-import and re-load on every snippet. Be careful about what setting this flag
means: it does not make an executor stateful, it declares that the backend
already is, and an implementation that is not raises rather than pretending.

`error_retry_attempts` bounds the self-correction loop. Each failed run sends the
error text back to the model for another attempt, and once the count is spent the
agent stops asking and the error stands as the result.

`code_block_delimiters` decides what counts as a code block in the model's
output, and `execution_result_delimiters` decides how the result is written back
for a model that has no native code-execution parts and can only see text.

`timeout_seconds` caps a single execution, and how that cap is enforced is left
to the subclass. `UnsafeLocalCodeExecutor` puts the child in its own process
session so it can kill the entire group, containers get an alarm, and Kubernetes
gets a watch interval. The base default is `None`, which means no limit at all.

## Choose an implementation

Start from the question of who can influence the prompt, because that decides
how much you have to trust the code. If you are the only person writing to the
agent, running that code on your own host is a fair trade. If anyone else can
reach it, you need a boundary between the model's code and everything else, and
the remaining question is whether you operate that boundary or a provider does.
The three groups below are those three answers, in that order. Every class is
imported from `google.adk.code_executors`,
except `CloudRunSandboxCodeExecutor`, which lives with the rest of the Cloud Run
integration in `google.adk.integrations.cloud_run`.

### No isolation, on your own host

Reach for this while you are developing, and stop reaching for it the moment
somebody else can shape the conversation.

*   **`UnsafeLocalCodeExecutor`** runs the snippet in a child Python interpreter
    through `subprocess.Popen`, in its own process session so a timeout can kill
    everything the snippet started. It asks nothing of your deployment beyond
    the process you already have, which is exactly why the model's code ends up
    next to your credentials.

### A boundary you run yourself

Choose one of these if you need real isolation and you would rather own the
infrastructure than hand code to a provider. You pay for that in operations: a
Docker daemon or a Kubernetes cluster, and a container started for every snippet
the model writes.

*   **`ContainerCodeExecutor`** starts a local or self-hosted Docker container
    with the network disabled and Linux capabilities dropped.
*   **`GkeCodeExecutor`** runs the snippet on a Kubernetes cluster in
    gVisor-sandboxed Pods, or through the Agent Sandbox client, so the code
    talks to a sandboxed kernel rather than the node's own.

### A sandbox somebody else runs

Choose one of these if you would rather not operate a sandbox at all. The
operational work goes to a provider, and what you accept in return is an
account, a quota, and code executing somewhere you do not administer.

*   **`BuiltInCodeExecutor`** uses Gemini's native server-side execution. No
    infrastructure of yours is involved at all, because the sandbox belongs to
    the model provider, and the matching constraint is that it works only on
    models that offer one.
*   **`VertexAiCodeExecutor`** uses the Google Cloud Vertex AI Code Interpreter
    Extension.
*   **`AgentEngineSandboxCodeExecutor`** uses a Vertex AI Reasoning Engine or
    Agent Engine sandbox environment.
*   **`CloudRunSandboxCodeExecutor`** suits an agent already running inside a
    Cloud Run container with sandboxes enabled. It shells out to the guest
    `sandbox` binary, so you cannot drive it from a local machine.

### What narrows the field after that

Once you have settled on a level of trust, two details can take a choice away
from you again.

If your agent needs a variable to survive from one snippet to the next, or wants
a CSV attached for it, three of the seven are already out.
`UnsafeLocalCodeExecutor`, `ContainerCodeExecutor` and
`CloudRunSandboxCodeExecutor` reject `stateful` and `optimize_data_file`.

If you pick `CloudRunSandboxCodeExecutor`, read its options before you configure
it, because it is the one implementation that changes a base default. Its
`timeout_seconds` is `300` rather than `None`, because the base default would
wait forever and a snippet that never terminates would hang the agent along with
it. It also adds two options of its own. The first is `sandbox_bin`, the path to
the guest binary, which defaults to `/usr/local/gcp/bin/sandbox`. The second is
`allow_egress`, which defaults to `False`, so the sandboxed code has no network
access until you turn it on.

## Advanced applications

The two examples below are the first two steps away from running the model's
code on your host, in order of how much you have to operate to get there.

### Run in a container

`ContainerCodeExecutor` starts a Docker container as a non-root user, with the
network off and Linux capabilities dropped. If you are moving off
`UnsafeLocalCodeExecutor`, this is the shortest step that puts a real boundary
between the model's code and your host:

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import ContainerCodeExecutor

agent = LlmAgent(
    name="data_analyst",
    instruction="Analyze data using Python scripts.",
    code_executor=ContainerCodeExecutor(
        image="python:3.11-slim",
        network_enabled=False,
        timeout_seconds=60,
    ),
)
```

### Run in a gVisor sandbox on Kubernetes

`GkeCodeExecutor` creates one short-lived Job per execution on the gVisor
(`runsc`) runtime, which gives the model's code a sandboxed kernel to talk to
instead of the node's own:

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import GkeCodeExecutor

agent = LlmAgent(
    name="k8s_code_agent",
    code_executor=GkeCodeExecutor(
        namespace="agent-sandboxes",
        image="python:3.11-slim",
        cpu_limit="1000m",
        mem_limit="1Gi",
        timeout_seconds=120,
    ),
)
```

## Limitations

*   **Only the first code block runs.** ADK takes the first block that matches
    the delimiters and discards everything after it, later blocks and prose
    alike. A model that writes two snippets in one turn gets one of them
    executed, so instruct it to write a single block per turn and let each
    result drive the next one.
*   **`UnsafeLocalCodeExecutor` runs on your host.** There is no sandbox, no
    resource limit beyond `timeout_seconds`, and the child interpreter inherits
    your environment. Never point it at untrusted input.
*   **Three implementations reject `stateful` and `optimize_data_file`.**
    `UnsafeLocalCodeExecutor`, `ContainerCodeExecutor` and
    `CloudRunSandboxCodeExecutor` raise `ValueError` at construction if you set
    either to `True`, with a message naming the class.
*   **Four implementations need extra packages.** `VertexAiCodeExecutor`,
    `ContainerCodeExecutor`, `GkeCodeExecutor` and
    `AgentEngineSandboxCodeExecutor` arrive with
    `pip install "google-adk[extensions]"`. Without them the import still
    succeeds, and construction is where it fails.

## Related samples

*   [Built-in Code Execution](../../../../contributing/samples/code_execution/code_execution/agent.py) is a data science agent that uses `BuiltInCodeExecutor`.
*   [GKE Sandbox Code Execution](../../../../contributing/samples/code_execution/code_execution/gke_sandbox_agent.py) runs Python inside GKE with `GkeCodeExecutor`.
*   [Custom Code Execution](../../../../contributing/samples/code_execution/custom_code_execution/agent.py) extends `VertexAiCodeExecutor` to add stateful execution.
*   [Agent Engine Sandbox](../../../../contributing/samples/code_execution/agent_engine_code_execution/agent.py) uses the managed `AgentEngineSandboxCodeExecutor`.
*   [Vertex AI Code Execution](../../../../contributing/samples/code_execution/vertex_code_execution/agent.py) drives the Vertex Code Interpreter with session state.
