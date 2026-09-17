# FallbackModel

`FallbackModel` wraps an ordered list of models and moves to the next one when
a call fails. It is a `BaseLlm`, so an agent takes it wherever it would take
any other model.

## Introduction

LLM providers rate-limit and go down. Without a recovery path a 429 or a 503
propagates out of the model call, past the flow, and ends the invocation — the
provider's bad minute becomes the agent's outage.

`FallbackModel` gives that failure somewhere to go. It holds several models,
tries them in order, and returns the first response it gets. Because it is
itself a `BaseLlm`, nothing else in ADK has to know: `LlmAgent.model` accepts
it, the flow calls it through `generate_content_async` as usual, and the
delegate that actually served the request is the one named on the request and
on the trace.

It is deliberately narrow. It does not retry a model, and it does not route by
cost or task — a model that fails is simply abandoned for the next one. The
model layer already has answers for the neighbouring problems, and
[How it works](#how-it-works) describes how they fit together.

## Get started

Give `models` the model you want and the ones to fall back to.

```python
from google.adk.agents import LlmAgent
from google.adk.models import FallbackModel

agent = LlmAgent(
    name='reliable_agent',
    model=FallbackModel(models=['gemini-3.1-pro-preview', 'gemini-3.5-flash']),
    instruction='You are a helpful assistant.',
)
```

If `gemini-3.1-pro-preview` returns a 429, the same request goes to `gemini-3.5-flash`,
and the agent sees a normal response.

Entries can be instances rather than names, which is how a backup gets settings
of its own:

```python
from google.adk.models import FallbackModel
from google.adk.models.google_llm import Gemini
from google.genai import types

FallbackModel(
    models=[
        'gemini-3.1-pro-preview',
        Gemini(
            model='gemini-3.5-flash',
            retry_options=types.HttpRetryOptions(attempts=3),
        ),
    ],
)
```

## How it works

Each entry of `models` is resolved to a `BaseLlm` — an instance is used as is,
a string goes through `LLMRegistry.new_llm` once and is cached. Resolution is
deferred to first use, as it is for `LlmAgent.model`, so constructing a
`FallbackModel` never imports a provider — naming a `Claude` backup does not
pull in the `anthropic` package until that backup is actually reached. The cost
is that a misspelled backup name is reported when the fallback is first needed
rather than when the agent is defined.

Models edit the request in place before sending it — appending a user turn,
preprocessing tools, and for a live connection writing the speech config,
system instruction, tools and http options onto it — so a model that fails
after doing so is rolled back before the next one is tried. Otherwise the
backup would inherit settings the caller never asked it for, and since a model
writes some of them only when it has them, a backup with no voice of its own
would speak in the primary's. This applies on both paths, a turn and a
connection alike.

Only the contents, the config, the live connect config and the request's own
bookkeeping are captured for that rollback: `tools_dict` holds live tool
objects, which the models read but never edit, and an MCP tool reaches a
`threading.Lock` that cannot be copied at all. The model that succeeds keeps
its edits, which is what traces show.

The delegate's
name is written to `LlmRequest.model` before the call, because models read the
name off the request rather than off themselves; without that step a backup
would be handed the primary's name and call the wrong model.

A failure moves to the next model only when it carries one of
`retriable_status_codes`. The status is read from whichever shape the provider
raises: `code` on a `google.genai` `APIError`, `status_code` on litellm, OpenAI
and Anthropic errors, and `response.status_code` on `httpx` errors. An error
with no status — a connection reset, a bug in your callback — never reached the
service and is not a reason to try a different one, so it propagates untouched.

Some shapes are deliberately excluded even though they carry a status. litellm
hard-codes status 500 on `APIConnectionError` and `APIResponseValidationError`,
and neither is a server error — one never reached the service, the other says a
response did arrive and then failed a client-side check. Both are recognised
and treated as having no status. 408 is likewise absent from the default set,
unlike the lists ADK retries on, because a timeout does not say whether the
request was processed first.

If every model you want is reachable through LiteLLM, `LiteLlm` already has
this: `LiteLlm(model=..., fallbacks=[...])` passes the list to litellm, which
fails over inside the provider. `FallbackModel` is the option that works across
model classes — a `Gemini` primary with a `Claude` backup, or anything that
subclasses `BaseLlm` — and the two compose, since a `LiteLlm` configured that
way can be one of the entries here.

Retrying a single model is a separate layer and stays with that model:
`Gemini` and `ApigeeLlm` take `retry_options`, which the genai SDK applies at
the HTTP layer. `FallbackModel` tries each model exactly once, so a single 429
is never retried twice over by two layers that do not know about each other.
Server-side routing is a third layer, reached by model name — a
`model-optimizer-*` entry routes on Vertex AI, and can itself be the first
entry here with a client-side backup behind it:

```python
FallbackModel(models=['model-optimizer-exp-04-09', 'gemini-3.5-flash'])
```

Streaming constrains when a fallback is still possible. Once a model has
yielded the turn's first response the turn belongs to it, and a later failure
propagates instead of falling back: the caller already holds the chunks emitted
so far, and starting a second model would splice two models' output into one
turn. A non-streaming call yields once and so almost always fails before that
point, leaving it free to fall back.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `models` | `list[str \| BaseLlm]` | required | The models to try, in order. |
| `retriable_status_codes` | `frozenset[int]` | `{429, 500, 502, 503, 504}` | Statuses that move on to the next model. |

`models` must hold at least one entry; an empty list is rejected at
construction. The first entry is the primary — it is what `capabilities`
reports and what the `model` attribute is derived from. `model` is inherited from `BaseLlm` and cannot be set directly; list the
models to try in `models` instead.

`retriable_status_codes` defaults to the set ADK retries on, less 408 — see
[How it works](#how-it-works) for why. Narrow it to fall back on rate limiting
alone (`frozenset({429})`), or widen it for a provider that signals overload
some other way:

```python
from google.adk.models import FallbackModel

FallbackModel(
    models=['gemini-3.1-pro-preview', 'gemini-3.5-flash'],
    retriable_status_codes=FallbackModel.DEFAULT_STATUS_CODES | {529},
)
```

When every model has failed, the last provider's error propagates. To answer
with something instead of ending the invocation, handle it where ADK already
handles model errors — `LlmAgent.on_model_error_callback`, or the equivalent
plugin hook:

```python
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import FallbackModel
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def on_model_error(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse:
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part(text='Every model is unavailable; try again.')],
      )
  )


agent = LlmAgent(
    name='reliable_agent',
    model=FallbackModel(models=['gemini-3.1-pro-preview', 'gemini-3.5-flash']),
    on_model_error_callback=on_model_error,
)
```

That hook is not specific to this class, so it also covers the failures a
fallback cannot absorb: a non-retriable status, and a model that fails after
the turn has started streaming.

Live connections follow the same rule at their own boundary. `connect` tries
each model in turn and yields the connection of the first that accepts one:
until the connection is open nothing has crossed it, so handing the attempt to
another model loses nothing. Once it is open the session belongs to that
model — a backup cannot resume a bidirectional session already under way — so
a failure after that point reaches the caller unchanged. A failed attempt is
rolled back before the next model is tried, as it is for a turn, and whichever
connection was opened is closed on the way out, including when the `async
with` body raises.

## Limitations

`FallbackModel` is experimental. It is enabled by default and warns once the
first time you construct one, but its API can still change. Setting
`ADK_DISABLE_FALLBACK_MODEL` turns it off, which makes constructing it raise.

A provider that is unreachable, rather than one that answers with an error, is
not a trigger for failover — see [How it works](#how-it-works). A backup only
helps against a service that is up and refusing work.

A live session that drops is reconnected to the model that owns it, not failed
over, because the session-resumption handle is only meaningful to the model
that issued it. If that model stays down, the reconnect keeps failing rather
than silently starting a different model's session. The owner is remembered
against the request the live flow builds for the run, so it is the session
that is followed rather than the model's name — two entries can carry the same
name, as one model behind two keys or regions does.

This is limited to resuming inside the run that opened the session. A handle
carried into a *new* run through `RunConfig.session_resumption` belongs to a
request this model has never seen, so the only thing left to go on is the name
the flow put on it, which is the agent's own. Such a run is pinned to the
primary and has no failover for its first connection, and if a backup owned
that session the handle goes to a model that never issued it. Two entries
reporting the same model name are indistinguishable in that case, and
reconnecting raises rather than guessing.

Wrapping a model hides its concrete type from code that checks for it. The live
flow sets `session_resumption.transparent` only for a `Gemini` on Vertex AI, and
a `FallbackModel` is not a `Gemini`, so that default is not applied.

`capabilities` reports the primary model's capabilities even when a backup
serves the request. The request is built before any call is made, so it is
built for the primary; keep the entries close enough in capability that one
request suits all of them.

## Related samples

None yet. The closest existing sample is
[litellm_with_fallback_models](../../../../contributing/samples/models/litellm_with_fallback_models/agent.py),
which uses LiteLLM's own provider-level fallbacks rather than this class.
