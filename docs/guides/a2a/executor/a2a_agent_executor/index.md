# A2aAgentExecutor

`A2aAgentExecutor` sits between the A2A server and an ADK `Runner`. It takes an
incoming request from another agent, runs your agent on it, and translates the
ADK events that come back into A2A task updates the caller can understand. If
you want to inspect or change anything on the way in or the way out, you do it
here.

## Introduction

[`to_a2a`](../../utils/agent_to_a2a/index.md) builds an `A2aAgentExecutor` for
you, and for a plain deployment that is the end of it. You construct one
yourself when the default translation is not what you want, which usually means
one of these:

*   You want to turn a request away before it costs you a model call.
*   You want to copy a value from a header into the session.
*   You want to scrub text before it leaves your process.
*   You want your own audit trail as tasks come and go.

All of that lives in `A2aAgentExecutorConfig.execute_interceptors`, and you
install a configured executor by passing it back through
`to_a2a(agent, agent_executor_factory=...)`.

The import paths are nested and not re-exported:

```python
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.a2a.executor.config import ExecuteInterceptor
```

## Get started

The interceptor below replaces the text of everything the agent sends back,
which is the shape you would start from for redaction. The
`agent_executor_factory` you hand to `to_a2a` receives the runner it resolved,
and has to return the executor.

```python
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.a2a.executor.config import ExecuteInterceptor
from google.adk.a2a.utils.agent_to_a2a import to_a2a


async def redact_outgoing(executor_context, a2a_event, adk_event):
  """Replaces the text of every outgoing status message."""
  message = getattr(getattr(a2a_event, "status", None), "message", None)
  if message is not None:
    for part in message.parts:
      if getattr(part, "text", None):
        part.text = "[redacted]"
  return a2a_event


def build_executor(runner):
  return A2aAgentExecutor(
      runner=runner,
      config=A2aAgentExecutorConfig(
          execute_interceptors=[
              ExecuteInterceptor(after_event=redact_outgoing)
          ]
      ),
  )


a2a_app = to_a2a(root_agent, port=8001, agent_executor_factory=build_executor)
```

`ExecuteInterceptor` is a dataclass with three optional hooks, so an interceptor
that only needs one of them leaves the other two unset. Every hook is async.

## How it works

A2A is a protocol, so every request your agent answers runs the same fixed
sequence, and that sequence begins before your executor is involved at all. If
you are writing an interceptor, the steps you can reach start at 3. The first
two happen inside the client.

1.  **The client reads your agent card.** That tells it where to post and what
    your agent can do.
2.  **The client sends one message to the RPC endpoint,** naming the session
    that message belongs to. From this point on it is watching a single A2A
    task. Every answer it gets, the last one included, arrives as an update
    published on that task rather than as the return value of a call, which is
    why a request that produces no updates at all leaves a client waiting
    rather than failing.
3.  **The executor resolves the runner** and takes the session named in the
    request, creating that session if it does not exist.
4.  **The executor opens the task.** The first update announces a brand-new
    task as submitted, and the next moves it to a `working` status carrying the
    app name, user id, and session id as ADK metadata.
5.  **The executor runs your agent** on the incoming message and turns each ADK
    event the run produces into an update on the task. Whether the client sees
    those updates as they are published or collects them by polling depends on
    the card. The one [`to_a2a`](../../utils/agent_to_a2a/index.md) builds by
    default does not advertise streaming, and an A2A server refuses a streaming
    request against such a card, so a default deployment is polled.
6.  **The task ends in one of three ways,** and the ending is how the client
    finds out what happened.
    *   A run that finished in a working state with content publishes an
        artifact update carrying that content, followed by a `completed`
        status.
    *   A run that finished in some other state publishes that state as its
        final status.
    *   A run that raised publishes a `failed` status carrying the exception
        text, so a crash inside your agent reaches the caller as a failed task
        rather than as a dropped connection.

Your interceptors sit at three points in that sequence.

### The hooks

| Hook | Runs | Receives | Returns | Order |
| :--- | :--- | :--- | :--- | :--- |
| `before_agent` | once, before the agent starts | `RequestContext` | `RequestContext` | list order |
| `after_event` | on each outgoing A2A event, before it is enqueued | `ExecutorContext`, the A2A event, the ADK event it came from | the event, a list of events, or `None` | list order |
| `after_agent` | once, on the terminal status event | `ExecutorContext`, the terminal `TaskStatusUpdateEvent` | the event | **reverse** list order |

Each `before_agent` hook receives the `RequestContext` the previous one
returned, so a chain of them composes.

The reversal on `after_agent` is deliberate, and it catches people out. With
interceptors `[a, b]`, `before_agent` runs `a` then `b`, while `after_agent`
runs `b` then `a`, so each interceptor's own pair of hooks nests around the ones
that follow it in the list. `after_event` does *not* reverse.

`ExecutorContext` is a small read-only object carrying `app_name`, `user_id`,
`session_id`, and the `runner`. Use it when a hook needs to know which
session it is looking at, or needs to get at the session service.

**`after_event` returning `None` drops the event, and takes the rest of the task
with it.** The final `completed` status and the artifact update reach the hook
like any other event, so an interceptor that returns `None` unconditionally
leaves the client watching a task that never finishes. Filter on the specific
event you mean to drop and return the rest unchanged.

`after_event` may also return a list, which replaces one outgoing event with
several. Each is enqueued in order, and each is fed to the next interceptor in
the chain individually.

**Mutating an event in place propagates further than you might expect.** The
final artifact carries the same content as the status message it was built from,
so rewriting the text of a status update also rewrites that artifact. That is
usually what you want for redaction, and a surprise if you were only trying to
annotate.

## Configuration options

`A2aAgentExecutor` itself takes four arguments, all keyword-only:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `runner` | `Runner \| Callable[..., Runner \| Awaitable[Runner]]` | *required* | The runner, or a factory for one. |
| `config` | `A2aAgentExecutorConfig \| None` | `None` | Interceptors and converters. A default config is built when omitted. |
| `use_legacy` | `bool` | `False` | Force the legacy implementation regardless of the request. |
| `force_new_version` | `bool` | `False` | Force the new implementation regardless of the request. |

**`runner`.** Passing a callable defers building the runner until the first
request, which matters when the runner owns a connection pool you do not want
opened at import time. Sync and async callables both work, and the result is
cached after the first call. Anything that is neither a `Runner` nor callable
raises `TypeError` on the first request, not at construction.

`A2aAgentExecutorConfig` is a Pydantic model holding the interceptors and the
converters used along the way:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `execute_interceptors` | `list[ExecuteInterceptor] \| None` | `None` | The hooks described above. |
| `a2a_part_converter` | callable | `convert_a2a_part_to_genai_part` | Turns an inbound A2A part into a GenAI part. |
| `gen_ai_part_converter` | callable | `convert_genai_part_to_a2a_part` | Turns an outbound GenAI part into an A2A part. |
| `request_converter` | callable | `convert_a2a_request_to_agent_run_request` | Turns the A2A request into runner arguments. |
| `event_converter` | callable | `convert_event_to_a2a_events` | Turns one ADK event into A2A events. Used by the legacy implementation. |
| `adk_event_converter` | callable | `convert_event_to_a2a_events` | The same job for the new implementation. |

Replacing a converter is a much heavier change than adding an interceptor,
because you take over the whole translation for that stage instead of adjusting
what it produced. Start with an interceptor, and move to a converter only when
the shape of the translation itself is wrong for you.

## Advanced applications

The two examples below take the same hooks from opposite ends of a run. The
first acts before the agent has done anything, where a refusal costs nothing;
the second acts on the last event of the task, where everything is already
known.

### Reject a request before the agent runs

*   **Problem solved**: you want to check something about the caller, such as a
    header, a metadata field, or a remaining quota, before you spend a model
    call on them.
*   **Implementation**: raise from `before_agent`.

```python
async def enforce_quota(context):
  if _over_quota(context):
    raise PermissionError("quota exhausted for this caller")
  return context
```

`before_agent` must return a `RequestContext`; there is no "return `None` to
abort" convention here. Returning the context you were given, possibly modified,
is the only non-raising outcome.

Raising here is not the same as failing later, and the caller can tell the
difference. An exception from `before_agent` reaches the A2A server, which turns
it into a JSON-RPC error,
`{"code": -32603, "message": "quota exhausted for this caller"}`, with no task
created at all. A hook that raises in `after_event` or `after_agent` gives the
client a task that ends in a `failed` status carrying that same text, arriving
after whatever events had already been published. Either way your message
reaches the caller, but rejecting early leaves less debris behind.

### Annotate the terminal event

*   **Problem solved**: you want every finished task to carry a trace id or a
    cost figure the client can read.
*   **Implementation**: add to the final event's metadata in `after_agent`. It
    already carries the ADK app name, user id, session id, invocation id, author
    and event id under `adk_`-prefixed keys.

```python
async def stamp_trace(executor_context, final_event):
  final_event.metadata["trace_id"] = _trace_id_for(executor_context.session_id)
  return final_event
```

## Limitations

*   **Two implementations, selected by the client.** Which one handles a request
    depends on whether the caller requested the
    `https://google.github.io/adk-docs/a2a/a2a-extension/` extension, so the
    same server can behave differently for two clients. Pin it with
    `use_legacy` or `force_new_version` if you need one behavior.
*   **Interceptors are not installed by name or priority.** They run in list
    order, reversed for `after_agent`, and one interceptor has no way to skip
    the rest except by emptying the event list in `after_event`.
*   **`after_event` never sees an ADK event the converter discarded.** An ADK
    event with no content, such as a workflow node emitting only
    `Event(output=...)`, produces no A2A event, so no hook fires for it. When
    *every* event from a run is discarded that way, there is nothing left to
    publish: the task ends in `working` with no artifact, and the client waits
    for a completion that never comes. A served `Workflow` hits this by
    default, because returning a value from a node is the usual way to write
    one. See [to_a2a](../../utils/agent_to_a2a/index.md).
*   **A failure inside a hook fails the whole request.** There is no
    per-interceptor error isolation. An exception in `after_event` or
    `after_agent` ends the task `failed`, and the events already published stay
    published, so a client can see an artifact for a task that then fails.
*   **Experimental.** Both the executor and its config are decorated
    `@a2a_experimental` and emit a `UserWarning` on construction. Set
    `ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS` to silence it.

## Related samples

*   [A2A root agent](../../../../../contributing/samples/a2a/a2a_root) serves an
    agent through `to_a2a`, which builds this executor with its defaults.
*   [A2A human in the loop](../../../../../contributing/samples/a2a/a2a_human_in_loop)
    is a served agent that pauses mid-task, which exercises the non-terminal
    status updates your hooks will see.

## Related guides

*   [to_a2a](../../utils/agent_to_a2a/index.md) is the function that builds this
    executor, and it documents the `agent_executor_factory` argument that
    replaces it with yours.
*   [AgentCardBuilder](../../utils/agent_card_builder/index.md) describes the
    card a client reads before it ever sends you one of these requests.
*   [A2A remote agent configuration](../../agent/config/index.md) has the
    mirror-image interceptors, the ones that run on the client side of the same
    conversation.
