# CallerPrincipal

`CallerPrincipal` records whether a serving layer authenticated the caller of
an invocation, and as whom. The human-in-the-loop tool confirmation flow reads
it to decide whether an approval that arrived over the wire is one the agent
may act on.

## Introduction

A tool that requires confirmation pauses the agent until a user approves the
call. When the agent is served remotely, the approval arrives as a message on
the same channel as everything else the remote caller sends, so without a
principal the framework cannot tell an operator's approval from a remote peer
approving the dangerous tool call it just caused. `CallerPrincipal` carries the
one fact that decides this: whether the layer that received the request
verified who sent it.

The principal lives on `InvocationContext.caller_principal`. `Runner.run_async`
accepts it as a keyword argument, and the A2A executor sets it for every
request it serves, so an application that serves agents over A2A gets it
without writing any code. An application with its own serving layer sets it
from its own authenticator.

The design keeps the principal separate from the transport on purpose. A
transport is a proxy for identity, and a proxy for identity fails in both
directions: it refuses authenticated callers because of how they connected,
and it admits anyone who reaches a path the proxy does not cover. Asking the
authentication question directly avoids both failures.

## Get started

This example serves an agent behind an application's own request handler. The
handler runs its authenticator first, then tells the runner what it found. A
confirmation-gated tool is included so that the principal has something to
decide.

```python
from google.adk.agents import CallerPrincipal
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.bash_tool import ExecuteBashTool

agent = LlmAgent(name="ops_agent", tools=[ExecuteBashTool()])
runner = InMemoryRunner(agent=agent, app_name="ops")


async def handle_request(request, new_message):
  # Your authenticator runs before this point. Report what it established;
  # do not infer anything from the message the caller sent.
  user = request.authenticated_user
  principal = CallerPrincipal(
      authenticated=user is not None,
      user_name=user.name if user else None,
      source="gateway",
  )
  async for event in runner.run_async(
      user_id=request.user_id,
      session_id=request.session_id,
      new_message=new_message,
      caller_principal=principal,
  ):
    yield event
```

An in-process caller, such as a test or a command-line tool that calls
`runner.run_async` directly, does not pass a principal. Leaving it unset says
that no remote trust boundary was crossed, which is true for that caller.

## How it works

`InvocationContext.caller_principal` has three meaningful states, and the tool
confirmation flow treats each differently.

| State | Meaning | Tool confirmation |
| :--- | :--- | :--- |
| `None` | The invocation started in process. Nothing had to vouch for the caller, because the caller is the operator. | Honored. |
| `authenticated=False` | A serving layer handled the request and could not say who sent it. | Warned about by default; refused when strict mode is on. |
| `authenticated=True` | The serving layer verified the caller. | Honored. |

A refusal is explicit rather than silent. The confirmation is delivered to the
tool with `confirmed=False`, which is the same state a user's decline produces,
so a tool that follows the confirmation contract returns its rejection response
and the turn ends with a reason the caller can see. The pending call never
hangs waiting for an approval that will not come.

The principal is derived only from what the serving layer established. It is
never read from message content, from event authorship, or from transport
metadata, because a remote caller controls all three and could set any of them
to whatever the framework wanted to see.

### A2A

`A2aAgentExecutor` builds the principal from the A2A server's call context on
every request. The caller is authenticated exactly when the A2A server
authenticated it, which is also the case in which the invocation's user id is
the authenticated user name rather than a generated `A2A_USER_` value. The two
cannot disagree about the same request.

An A2A server that runs without an authenticator produces an unauthenticated
principal for every request. With the default settings that yields a warning
per approval and the approval is honored, which is the behavior those
deployments had before the principal existed.

### Strict mode

Strict mode refuses confirmations from unauthenticated callers instead of
warning about them. It is the `STRICT_CALLER_PRINCIPAL` feature and is off by
default, because turning it on changes what an existing deployment does with an
approval, and an upgrade should not do that on its own. The intent is to make
strict mode the default at the next major version.

Turn it on with the environment variable ADK uses for every feature:

```bash
export ADK_ENABLE_STRICT_CALLER_PRINCIPAL=1
```

Or programmatically, when environment variables are not practical in your
deployment:

```python
from google.adk.features import FeatureName
from google.adk.features import override_feature_enabled

override_feature_enabled(FeatureName.STRICT_CALLER_PRINCIPAL, True)
```

Strict mode never affects an invocation with no principal. An in-process
caller crossed no boundary, so there is nothing to refuse.

## Configuration options

`CallerPrincipal` is a frozen model with three fields. It rejects unknown
fields, so a typo cannot silently produce a principal that says nothing.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `authenticated` | `bool` | required | Whether a serving-layer authenticator verified the caller. |
| `user_name` | `str \| None` | `None` | The verified identity when `authenticated` is `True`. |
| `source` | `str \| None` | `None` | A short label for the serving edge that set the principal. |

`authenticated` is the only field a trust decision reads. Set it to `True` only
when your authenticator actually verified the caller. A request that carries a
name it asserted about itself, without verification, is not authenticated, and
saying otherwise reopens the problem the principal exists to close.

`user_name` is informational. It is the identity the authenticator established,
kept alongside the decision so that a log line or an audit record can say who
approved a tool call. Leave it `None` when `authenticated` is `False`; a name
without verification behind it is a claim, not an identity.

`source` is a label such as `"a2a"` or `"gateway"` that names the edge that
set the principal. It appears in the confirmation flow's log messages so that
an operator reading a warning can tell which entry point produced it. Nothing
makes a trust decision on it, so a transport cannot become a proxy for identity
through this field either.

## Advanced applications

### Custom A2A request converters

`A2aAgentExecutorConfig.request_converter` lets an application replace the
function that turns an A2A request into a run request. The executor sets the
principal from the server's call context after the converter returns,
regardless of what the converter produced. A converter can therefore neither
drop the principal nor assert one the server never established, which keeps
the trust decision structural rather than dependent on every converter
remembering to make it.

### Serving layers with more than one entry point

An application that exposes an agent through several edges can give each a
distinct `source` and let them share the same rule for `authenticated`. The
confirmation flow's log messages then identify the edge, and the trust
decision stays uniform across all of them.

## Limitations

- The principal describes the caller of the invocation as a whole. ADK does
  not verify the identity behind each individual event within a session.
- Only the tool confirmation flow reads the principal today. Other consumers
  can read `InvocationContext.caller_principal`, but the framework does not
  yet gate anything else on it.
- The synchronous `Runner.run` wrapper and live sessions do not accept a
  principal. Both behave as in-process callers.
- Setting a principal does not authenticate anything by itself. The ADK API
  server's `/run` endpoint still requires its own authentication; the principal
  records the outcome of authentication, it does not perform it.
