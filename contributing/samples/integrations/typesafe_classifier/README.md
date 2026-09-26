# TypeSafe classifier tool

This sample demonstrates `TypesafeClassifierTool`: an agent that triages a
support ticket by calling TypeSafe AI's Jev model instead of reasoning about
the ticket itself.

## What this sample demonstrates

Three separate `TypesafeClassifierTool` instances, one per question type,
sharing a single `TypesafeClassifier`:

- `classify_ticket_category` -- a `Choice` question
  (`billing` / `technical` / `other`).
- `check_ticket_urgency` -- a `Noul` question (does this need a same-day
  response, yes or no).
- `rate_ticket_severity` -- a `Score` question (low / medium / high, with
  a description per level).

The agent's instruction tells it to call all three tools with the ticket
text, then summarize the results, so you can watch each tool call and its
typed answer separately from the agent's own reply.

## Prerequisites

Install the extra:

```bash
pip install 'google-adk[typesafe]'
```

Copy `.env-sample` to `.env` in this directory and fill in both keys:

```bash
cp .env-sample .env
```

- `GOOGLE_API_KEY`: the agent's own model (Gemini).
- `TYPESAFE_API_KEY`: TypeSafe AI, from the [TypeSafe console](https://typesafe.ai).

## Running the sample

From the repository root:

```bash
adk web contributing/samples/integrations
```

Open the URL it prints, pick `typesafe_classifier` from the agent dropdown,
and send a message describing a ticket, for example:

```
I was charged twice for my subscription this month.
```

The agent calls all three tools, then reports the category, urgency, and
severity Jev returned. Open a tool call in the web UI's event trace to see
its raw typed answer:

- `classify_ticket_category`: `{"choice": ..., "confidence": ..., "probabilities": ...}`
- `check_ticket_urgency`: `{"noul": ...}`
- `rate_ticket_severity`: `{"score": ..., "confidence": ..., "legend": ..., "probabilities": ...}`

You can also run it from the CLI instead of the browser:

```bash
adk run contributing/samples/integrations/typesafe_classifier
```

## Related

- [TypesafeClassifierTool guide](../../../../docs/guides/integrations/typesafe/typesafe_classifier_tool/index.md)
- [TypesafeClassifier guide](../../../../docs/guides/integrations/typesafe/typesafe_classifier/index.md)
- Unit tests: `tests/unittests/integrations/typesafe/`
