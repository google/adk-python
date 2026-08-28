# TelemetryConfig

`TelemetryConfig` decides what ADK puts in its OpenTelemetry traces, and above
all whether the text of your prompts and the model's replies is copied onto the
spans your exporter ships out of the process. That question already has an
answer if you never set one, and the answer is yes. An agent you have not
configured writes the conversation onto its spans, and whatever your exporter
points at receives it.

## Introduction

ADK emits spans for every model call, every tool call, and the data it sends to
an agent. Those spans are useful because they carry detail: which tool ran,
with which arguments, and what came back. The same property makes them a data
path out of your application. A span attribute holding a serialized
`LlmRequest` holds the conversation.

`TelemetryConfig` is the single place where that is decided. Every content
toggle in ADK's tracing code reads one of its properties, so there is one
precedence ladder rather than a scatter of environment variables checked in
different places. You attach an instance to a run through
`RunConfig.telemetry`, or you set nothing and let the environment variables
answer.

Three things are configurable.

*   Whether prompt and response content is captured.
*   Whether the experimental GenAI semantic conventions are used instead of
    ADK's older attribute names.
*   Whether experimental telemetry is emitted at all.

## The privacy decision, first

**With no configuration at all, prompt and response content is written to
ADK's spans.** The relevant properties resolve like this out of the box:

| Property | Default | What it means |
| :--- | :--- | :--- |
| `should_add_content_to_legacy_spans` | `True` | Prompts, model replies, tool arguments and tool results go on ADK's own spans. |
| `should_add_content_to_logs` | `False` | Nothing goes on emitted log records. |
| `should_add_content_to_experimental_spans` | `False` | Nothing goes on the experimental inference span. |

Those three rows are three destinations, not three grades of the same span, so
it is worth being clear about which is which.

A **legacy span** is one ADK creates itself and writes its own attributes onto,
`call_llm` above all, under names beginning `gcp.vertex.agent.`. Those are the
spans you get today, and the ones the ADK web UI and existing dashboards read.

An **experimental span** is the model-call span once you opt in to the
experimental GenAI semantic conventions with
`genai_semconv_stability_opt_in`. Opt in and the conversation is written onto
that span under the `gen_ai.*` attribute names the convention defines, rather
than ADK's own.

There is no ordinary span between the two because the stable GenAI convention
does not put conversation content on a span at all. Without the experimental
opt-in, ADK still opens a `generate_content <model>` span for the model call,
but it carries only metadata such as the model name and the operation name. The
messages leave as log records named `gen_ai.system.message`,
`gen_ai.user.message` and `gen_ai.choice`, which is exactly what the middle row,
`should_add_content_to_logs`, gates. So the real choice is between ADK's own
span attributes, the stable convention's log records, and the experimental
convention's span attributes.

The first row is the one that surprises people. Concretely, run an agent with no
telemetry configuration and the user's message appears in full on the `call_llm`
span, under the attribute `gcp.vertex.agent.llm_request`. Whatever your exporter
points at then receives that text unless you say otherwise, whether that is
Cloud Trace, an OTLP collector, or a third-party observability vendor.

Two things explain the default, and neither makes it safer. The first is that a
trace without the prompt is close to useless for the problem tracing is usually
opened for, which is an agent that answered wrongly, so the setting was chosen
to make a default installation debuggable. The second is age:
`ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` predates the OpenTelemetry content
variable and defaults on, where the OpenTelemetry one defaults off. Whatever the
reason, the effect on a deployment you have not configured is that the
conversation leaves the process, so decide about it before you ship rather than
after.

**To turn it off for the whole process**, which is the setting to reach for as
soon as the traces leave systems you control:

```bash
export ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false
```

**To turn it off for one run**, pass an explicit mode:

```python
run_config = RunConfig(
    telemetry=TelemetryConfig(
        capture_message_content=ContentCapturingMode.NO_CONTENT
    )
)
```

The two are not equivalent. The environment variable silences ADK's own spans
and leaves the OpenTelemetry paths at their own defaults. `NO_CONTENT` silences
everything. If you want a deployment-wide guarantee that application code
cannot undo, see
[Lock the policy against application code](#lock-the-policy-against-application-code).

Some redaction happens regardless. Request `http_options` are excluded from the
serialized request, because `headers` routinely carries an authorization
bearer token and `extra_body` is a free-form passthrough. Inline binary parts
are replaced with `<inline_data: image/png, 20481 bytes>` rather than
base64-encoded onto an attribute. Neither of those touches the actual message
text, which is the part a privacy review asks about.

## Get started

Attach a config to a run:

```python
from google.adk.agents.run_config import RunConfig
from google.adk.telemetry import ContentCapturingMode
from google.adk.telemetry import TelemetryConfig

telemetry = TelemetryConfig(
    capture_message_content=ContentCapturingMode.NO_CONTENT,
)

async for event in runner.run_async(
    user_id="user-123",
    session_id=session.id,
    new_message=message,
    run_config=RunConfig(telemetry=telemetry),
):
  ...
```

`TelemetryConfig` is frozen, so one instance can be shared across concurrent
runs without any copying. It reads the environment lazily, at the moment a
property is consulted, so a variable changed later in the process still takes
effect.

## How it works

Every setting is answered the same way, from three fields that a fixed
precedence ladder reads in order. Knowing that order is how you predict what a
given deployment actually does.

### Resolution

Every property answers the same four-step question, in this order.

1.  **The admin lock.** If `ADK_TELEMETRY_IGNORE_RUN_CONFIG` is `1` or `true`,
    the per-request fields are skipped entirely and resolution starts at step
    3. The lock exists so that whoever operates the deployment can stop
    application code from overriding a telemetry policy.
2.  **The per-request field**, if it is not `None`.
3.  **The environment variable** for that setting.
4.  **The built-in default.**

A field left at `None` is not "off". It means "I have no opinion", and
resolution falls through to the environment.

### The three fields

`TelemetryConfig` carries three fields, each with an environment variable it
falls back to when the field is left at `None`.

| Field | Type | Default | Environment variable |
| :--- | :--- | :--- | :--- |
| `capture_message_content` | `ContentCapturingMode \| None` | `None` | `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` |
| `genai_semconv_stability_opt_in` | `'stable' \| 'experimental' \| None` | `None` | `OTEL_SEMCONV_STABILITY_OPT_IN` |
| `adk_experimental_telemetry_opt_in` | `bool \| None` | `None` | `ADK_EXPERIMENTAL_TELEMETRY` |

The model forbids unknown fields, so a misspelled keyword raises at
construction rather than being ignored.

**`capture_message_content`** takes a `ContentCapturingMode`, which has four
members naming where content is allowed to go.

*   `NO_CONTENT` allows it nowhere.
*   `EVENT_ONLY` allows it on emitted log records.
*   `SPAN_ONLY` allows it on the active span.
*   `SPAN_AND_EVENT` allows both.

Its environment variable accepts the matching uppercase strings, and also
accepts a legacy `true` or `1`, which is read as `EVENT_ONLY`. Any other value,
including a typo, resolves to `NO_CONTENT` without a warning.

**`genai_semconv_stability_opt_in`** switches between ADK's established
`gcp.vertex.agent.*` span attributes and the experimental GenAI semantic
conventions. The environment path only understands opting *in*: it looks for
the token `gen_ai_latest_experimental` in the comma-separated variable and
infers "stable" from its absence. So `'stable'` is a per-request value with no
environment-variable equivalent, and it is the way to hold one run on the
legacy attributes while the deployment as a whole has opted in.

**`adk_experimental_telemetry_opt_in`** gates telemetry whose shape is still
changing, which so far means spans and attributes for skills. It is off by
default, because an attribute name that moves between releases breaks a
dashboard built on it.

### Why one field drives four properties

`capture_message_content` is read by four separate `should_*` properties
because ADK has two generations of spans, and the routing rules differ.

*   `should_add_content_to_logs` and `should_add_content_to_experimental_spans`
    follow the OpenTelemetry rule exactly: log records get content under
    `EVENT_ONLY` and `SPAN_AND_EVENT`, spans get it under `SPAN_ONLY` and
    `SPAN_AND_EVENT`.
*   `should_add_content_to_legacy_spans` governs the attributes ADK writes
    itself, which come in three groups: the model exchange, under
    `gcp.vertex.agent.llm_request` and `llm_response`; the tool exchange, under
    `tool_call_args` and `tool_response`; and the agent's input, under `data`.
    **It is the one with its own environment fallback**,
    `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS`, defaulting
    to on. Set `capture_message_content` to anything at all and that fallback
    is bypassed: the OpenTelemetry span routing applies instead, so
    `EVENT_ONLY` turns legacy span content *off*.

That is what separates the two ways of disabling capture.
`ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false` silences the legacy spans and
leaves the OpenTelemetry-spec paths at their own defaults.
`capture_message_content=NO_CONTENT` silences everything, including any content
the OpenTelemetry variable had switched on. `EVENT_ONLY` also silences the
legacy spans, which people miss because its name says nothing about spans.

When content is suppressed the attributes are not dropped; they are set to the
string `"{}"`. The span shape stays the same so that consumers expecting those
keys keep working.

### The instrumentation-library caveat

If `opentelemetry-instrumentation-google-genai` is installed, it wraps
`google.genai.Models.generate_content` and creates the inference span itself,
reading its own OpenTelemetry environment variables. Per-request overrides do
not reach that span. They still apply to every span ADK owns, so the effect is
a split policy: one library's rules for the model call, yours for everything
around it.

## Advanced applications

Two needs pull in opposite directions. An operator wants the policy fixed for
the whole deployment, and a developer wants content captured for one session.
The configuration serves both, and the admin lock is what keeps the second from
defeating the first.

### Lock the policy against application code

A platform team that has decided prompt content must never leave the process
cannot rely on every agent author passing the right `RunConfig`. Set both
variables in the deployment environment:

```bash
export ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false
export ADK_TELEMETRY_IGNORE_RUN_CONFIG=1
```

The first sets the policy, the second makes `TelemetryConfig` ignore the
per-request fields, so an application passing
`capture_message_content=SPAN_AND_EVENT` gets `NO_CONTENT` anyway.

### Turn content on for one run

The reverse case is the reason the per-request field exists at all. You keep
content out of traces normally, then switch it on for the one session you are
debugging.

```python
debug_telemetry = TelemetryConfig(
    capture_message_content=ContentCapturingMode.SPAN_AND_EVENT
)
run_config = RunConfig(telemetry=debug_telemetry if is_debug else None)
```

Passing `None` is the right way to say "no opinion", because it leaves the
deployment defaults in charge rather than pinning them to whatever this run
happened to want.

## Limitations

*   **Content capture is on by default for ADK spans.** Prompt text, model
    replies, tool arguments and tool results leave the process with your traces
    until you set `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false` or pass
    `capture_message_content=NO_CONTENT`.
*   **An unrecognized environment value fails quietly.** Anything outside the
    four mode names, and the legacy `true` and `1`, resolves to `NO_CONTENT` with
    no warning, so a typo silently disables capture. `ADK_TELEMETRY_IGNORE_RUN_CONFIG`
    and `ADK_EXPERIMENTAL_TELEMETRY` behave the same way in reverse: only `1`
    and `true` count as set.
*   **`'stable'` cannot be expressed as an environment variable.** The semconv
    variable only supports opting in.
*   **The granularity is the whole request.** There is no per-field or
    per-agent redaction, and no hook to rewrite an attribute before export. It
    is all of the content or none of it. Selective redaction needs an
    OpenTelemetry span processor of your own.
*   **`RunConfig.telemetry` does not reach an external instrumentation
    library.** An installed `opentelemetry-instrumentation-google-genai`
    creates the inference span itself and reads only its own OpenTelemetry
    environment variables.

## Related guides

*   `RunConfig` is the object a `TelemetryConfig` attaches
    to, and covers the rest of what it controls per run.
