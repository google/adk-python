# EventsCompactionConfig and BaseEventsSummarizer

The `EventsCompactionConfig` class is the `App` setting that keeps a long
conversation from outgrowing the model's context window. It tells ADK when to
replace a range of older events with a short summary, and the
`BaseEventsSummarizer` class is the interface you implement when you want to
write that summary yourself.

## Introduction

Every turn of a conversation adds events to the session, and ADK sends those
events to the model on every later turn. In a short conversation that is what
you want, because the model needs the history to answer well. In a conversation
that a user returns to for weeks, or one whose tools return whole documents, the
prompt keeps growing. Each turn costs more and takes longer than the one before
it, and eventually the model rejects a request because it no longer fits in the
context window.

Compaction lets you limit that growth. When you set `events_compaction_config`
on your `App`, ADK watches the session as it grows. At the points you choose, it
asks a summarizer to condense a range of older events into one summary, stores
the summary in the session as a new event, and from then on sends the summary to
the model in place of the events it covers.

Compaction does not delete anything. The original events stay in the session,
so you can still read the full history, show it in your UI, or export it. Only
the prompt that ADK builds for the model changes.

You decide when compaction runs by choosing which triggers to set. The
sliding-window trigger writes a summary every few turns. The token-threshold
trigger writes one when the prompt reaches a size you set. By default ADK writes
the summary with an agent's own model, and you can supply your own summarizer
when you want different summaries or a different model.

## Get started

The following example adds compaction to an app with one agent. ADK writes a
summary after every second turn, and each new summary also covers the last turn
of the previous one.

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig

root_agent = LlmAgent(
    name="concierge",
    instruction="Answer the user's questions.",
)

app = App(
    name="concierge_app",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=2,
        overlap_size=1,
    ),
)
```

Import `EventsCompactionConfig` from `google.adk.apps.app`. The
`google.adk.apps` package exports only `App` and `ResumabilityConfig`, so
`from google.adk.apps import EventsCompactionConfig` fails.

Nothing else in your code changes. If you run the agent with `adk run`,
`adk web`, or `adk api_server`, export the `App` from your agent module as `app`
and the CLI picks up the configuration, as described in
[Running your app](../app/index.md#running-your-app). If you drive the agent
yourself, pass the `App` to a `Runner`, and compaction happens inside
`runner.run_async` with no extra calls.

## How it works

From your code's point of view, a compaction goes through the following steps:

1.  ADK checks the triggers. It checks the sliding-window trigger after each
    invocation, once it has yielded the last event of the turn to you. It
    checks the token-threshold trigger at that point too, and also before each
    model call, so ADK can compact a long turn with many tool calls before the
    turn ends.
2.  If a trigger fires, ADK chooses a range of events. It passes the range to
    the summarizer's `maybe_summarize_events` method only when replacing the
    range would remove more characters from the prompt than the last summary
    it wrote for this session takes up. Otherwise it writes no summary, which
    keeps it from paying for a summary that makes the prompt no smaller.
3.  The summarizer returns one event whose `actions.compaction` is an
    `EventCompaction`. It holds the summary text and the start and end
    timestamps of the range. ADK appends that event to the session.
4.  The next time ADK builds a prompt, it leaves out every event whose
    timestamp falls inside a summary's range and puts the summary in their
    place. ADK attributes the summary to the agent whose prompt it is building,
    so the model reads it as part of its own history.

The following rules about the range affect what ends up in a summary:

*   ADK never summarizes turns that you undid with `Runner.rewind_async`, so a
    rewound turn cannot reach the model again through a summary. The
    [rewind guide](../../runners/runner/rewind.md) explains rewinding.
*   A range never ends between a tool call and its response. A tool call that
    is waiting for the user to confirm it or to provide a credential stays out
    of the summary, together with every event after it, until the call
    completes.

Compaction adds work to the turn that triggers it. ADK compacts inside the
`run_async` generator after it yields the last event of the turn, so your
`async for` loop over that turn does not end until the summarizer's model call
has finished.

## Configuration options

The trigger fields come in pairs, and you must set both fields of a pair
together. You must set at least one pair. Setting half of a pair, or no pair at
all, raises a Pydantic `ValidationError` when you construct the config. The
`EventsCompactionConfig` class also rejects field names it does not define, so
a misspelled field raises an error instead of being ignored.

### Summarize every few turns: `compaction_interval` and `overlap_size`

The `compaction_interval` field is the number of new invocations that triggers
a summary. An invocation is one call to `run_async`, which is one user turn no
matter how many model calls and tool calls it made. The `overlap_size` field is
the number of earlier invocations that each new summary covers again.

After each invocation, ADK counts the invocations that are newer than the end of
the last summary. When the count reaches `compaction_interval`, it summarizes
those invocations together with the `overlap_size` invocations before them, as
long as that range passes the size check in step 2 above. With
`compaction_interval=2` and `overlap_size=1`, the sequence is as follows:

1.  After turn 1, one invocation is new, so nothing happens.
2.  After turn 2, two invocations are new, and ADK writes a summary of turns 1
    and 2.
3.  After turn 3, only turn 3 is newer than that summary, so nothing happens.
4.  After turn 4, turns 3 and 4 are new. The range starts one invocation
    earlier, at turn 2, so the new summary covers turns 2 through 4. ADK
    writes it only if that range is bigger than the summary from step 2. If it
    is not, the summary waits. The range keeps every turn since the last
    summary, so it grows until those turns add up to more than that summary,
    and a conversation of short turns gets summaries less often rather than
    never.

The overlap gives each summary some of the context the previous one had, so the
model does not read a summary that starts in the middle of a topic. Set
`overlap_size=0` if you want summaries that share no turns.

### Summarize when the prompt gets large: `token_threshold` and `event_retention_size`

The `token_threshold` field is a prompt size in tokens. ADK compares it with the
prompt token count from a model response, which it reads from an event's
`usage_metadata`. The check before a model call reads only the counts recorded
by the agent that is about to call the model, because a count describes the
prompt of the agent that made the call, and an app whose turn ends in a small
sub-agent would otherwise measure that sub-agent forever. The check after an
invocation takes the newest count from any agent. If no event since the last
summary reports a count, ADK estimates the size by counting the characters it
would send and dividing by four. When the count reaches the threshold, ADK runs
compaction.

The `event_retention_size` field is the number of recent events that stay out
of the summary, so that the model keeps seeing them word for word. ADK takes
every event since the last summary, keeps the newest `event_retention_size` of
them as they are, and summarizes the rest. With `token_threshold=50_000` and
`event_retention_size=20`, a session with 34 events since the last summary and a
reported prompt of 52,000 tokens gets its oldest 14 events summarized. If only
18 events have arrived since the last summary, nothing happens, because all of
them are within the 20 that ADK keeps.

### Use both triggers

You can set both pairs. After each invocation, ADK checks the token threshold
first, and if that writes a summary, ADK skips the sliding window for that
invocation. Together they give you a summary at a regular pace, plus an extra
one when a single turn makes the prompt grow quickly, for example because a tool
returned a large document.

### Choose the summarizer: `summarizer`

The `summarizer` field is not part of a pair. If you leave it unset, ADK creates
an `LlmEventSummarizer` the first time it needs a summary. For a check
after an invocation, it uses the root agent's model. For a token-threshold check
before a model call, it uses the model of the agent that is about to make that
call.

The root agent must be an `LlmAgent` for the default to work after an
invocation. If your root agent is a `Workflow` or another kind of agent, set
`summarizer` yourself. Otherwise, when either trigger fires after an invocation
before ADK has created a default summarizer, ADK raises `ValueError` at the end
of that turn.

## Advanced applications

### Use a different model or prompt

The `LlmEventSummarizer` class takes the model to use and an optional prompt
template. Create one yourself when you want summaries from a smaller, cheaper
model than the one your agent uses, or when the summary has to keep something
the default prompt does not ask for. To use one, import the class from
`google.adk.apps.llm_event_summarizer`, pass a model such as
`Gemini(model="gemini-2.5-flash")` as the `llm` argument and your template as
the `prompt_template` argument, and set the result as the `summarizer` field of
the config.

The template must contain the `{conversation_history}` placeholder, which ADK
replaces with the formatted events. ADK fills in the template with Python's
`str.format`, so you have to double any other brace in your template, as `{{`
and `}}`, or formatting fails.

### Write your own summarizer

Subclass `BaseEventsSummarizer` when a summary does not need a model, or needs
logic that a prompt cannot express. You implement its abstract
`maybe_summarize_events` method. The method receives the events in the range as
the keyword-only argument `events`, and it returns either one `Event` that
carries the summary, or `None` to skip this compaction and leave the events as
they are.

The following summarizer writes a list of the tools the agent called, without a
model call. It suits an agent whose useful history is the actions it took more
than the words it used.

```python
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events import Event
from google.adk.events import EventActions
from google.adk.events.event_actions import EventCompaction
from google.genai import types

class ToolLogSummarizer(BaseEventsSummarizer):
  """Summarizes a range of events as the list of tools that ran."""

  async def maybe_summarize_events(
      self, *, events: list[Event]
  ) -> Event | None:
    tool_names = [
        call.name for event in events for call in event.get_function_calls()
    ]
    if not tool_names:
      return None

    summary = types.Content(
        role="model",
        parts=[
            types.Part(
                text="Earlier in this conversation the agent called: "
                + ", ".join(tool_names)
            )
        ],
    )
    return Event(
        invocation_id=Event.new_id(),
        author="user",
        actions=EventActions(
            compaction=EventCompaction(
                start_timestamp=events[0].timestamp,
                end_timestamp=events[-1].timestamp,
                compacted_content=summary,
            )
        ),
    )
```

Pass an instance to the config as `summarizer=ToolLogSummarizer()`. A custom
summarizer has to follow a few rules:

*   The returned event must carry an `EventCompaction` in `actions.compaction`.
    ADK appends whatever event you return, but only an event with a compaction
    hides anything from the prompt.
*   The timestamps decide which events are hidden. ADK hides every event whose
    timestamp falls between `start_timestamp` and `end_timestamp`, so take them
    from the first and last event you received. A wider range hides events that
    your summary does not describe.
*   When the token threshold fires and an earlier summary exists, the first
    event you receive carries the earlier summary's text and its start time. A
    summary that covers `events[0]` through `events[-1]` replaces the earlier
    one, so include what the earlier summary said.

Import the `EventCompaction` class from `google.adk.events.event_actions`,
because the `google.adk.events` package does not export it.

## Limitations

*   Storage keeps growing. Compaction adds a summary event and never removes an
    event, so the session in your session service grows with every turn even
    though the prompt does not. Plan your storage for the full history.
*   The model sees only the summary. Anything the summarizer leaves out is gone
    from the model's point of view, even though the original events are still
    stored. If a later turn refers to an order number or a tool result that the
    summary dropped, the agent cannot get it back. Write the summarizer, or its
    prompt, so that it keeps the identifiers your agent needs.
*   Each summary costs a context cache miss. If you also set
    `context_cache_config` on the `App`, the cache holds the start of the
    request, which includes the first part of the conversation. A summary
    replaces events in that part, so the next model call runs without the
    cache, and ADK creates a new cache on a later call. A longer interval
    between summaries makes those misses rarer.
*   The `EventsCompactionConfig` and `BaseEventsSummarizer` classes are
    experimental. Constructing the config emits a `UserWarning` once per
    process, and setting the environment variable
    `ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS=true` turns the warning off.
    Their fields and behavior may change in a later release.

## Related guides and samples

*   [App](../app/index.md) is the container that you set this config on.
*   [EventActions](../../events/event_actions/index.md) describes the
    `compaction` field that carries a summary.
*   [Application configuration](../../../../contributing/samples/core/app/agent.py)
    is a runnable app that sets `compaction_interval=2` and `overlap_size=1`
    alongside plugins and context caching.
