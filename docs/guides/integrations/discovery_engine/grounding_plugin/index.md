# GroundingPlugin

`GroundingPlugin` verifies each final model response against facts you provide and attaches citations to the response. `CheckGroundingClient` performs the verification with the Discovery Engine [CheckGrounding API](https://cloud.google.com/generative-ai-app-builder/docs/check-grounding).

## Introduction

A model can state things that no source supports. Grounding verification checks a response against a fixed set of facts, reports which claims those facts support, and returns citations that point back to the supporting facts.

The integration has two parts. `GroundingPlugin` is a `BasePlugin` subclass that holds your facts and runs a check in `after_model_callback`. `CheckGroundingClient` calls the CheckGrounding API and maps its answer onto ADK `GroundingMetadata`. The plugin depends only on the `GroundingClient` protocol, so you can supply another backend.

## Get started

Install the dependency:

```shell
pip install 'google-adk[gcp]'
```

Create the plugin with the facts the agent's answers should rest on, and add it to an `App`:

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.integrations.discovery_engine import Fact
from google.adk.integrations.discovery_engine import GroundingPlugin

plugin = GroundingPlugin(
    facts=[
        Fact(
            text="Employees receive 16 weeks of paid parental leave.",
            title="Employee handbook",
            link="https://example.com/handbook#leave",
        )
    ]
)

agent = LlmAgent(
    name="policy_assistant",
    instruction="Answer questions about company policy.",
)

app = App(name="policy_app", root_agent=agent, plugins=[plugin])
```

Each final text response from the agent now carries `grounding_metadata` with the facts it cites and the claims they support. The plugin uses a `CheckGroundingClient` for the project in `GOOGLE_CLOUD_PROJECT`, with Application Default Credentials. Pass `client=` to set the project or other options.

## How it works

The plugin acts in `after_model_callback`. It skips streamed partial responses, because citation offsets computed against incomplete text would not match the final text, it skips responses that call a tool, because they are not the final answer, and it skips responses with no text. For any other response, it sends the response text, without thought parts, and the facts to the client.

When the check succeeds and cites at least one fact, the plugin adds the returned citations to the response's `grounding_metadata`. An answer that cites nothing gets no metadata. Metadata the model attached itself, such as results from its own Google Search, is kept, and the check's citations follow it. Each grounding support names a segment of the response, by the index of the part it falls in and byte offsets within that part, along with the indices of the facts that support it and a confidence score for each fact. A claim that spans two parts becomes one support per part. When the API scores the claim, the score is the claim's own score. Otherwise it is the answer-level support score, and it is left unset when the API returns neither.

A failed check leaves the response unchanged. `CheckGroundingClient` reports errors returned by the API through the result's status. The plugin also catches any other error the client raises, such as missing credentials, logs it as a warning and reports it as `API_ERROR`, so a verification problem never fails the turn.

The plugin splits a `Fact` whose text is longer than the client's `limits.max_fact_chars` into chunks at paragraph, line, sentence or word boundaries. A later fact with the same `id` replaces an earlier one. Without an `id`, the ID is a hash of the text, title, link and attributes, so the same fact passed twice is sent once. Construction raises `ValueError` when the facts need more than `limits.max_facts` chunks, because the API evaluates at most that many per check.

## Configuration options

`GroundingPlugin` takes the following options:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `client` | `GroundingClient` | `CheckGroundingClient()` | Backend that performs the check. |
| `facts` | `Sequence[Fact]` | `()` | Facts every response is checked against. |
| `on_result` | `OnResult` | `None` | Hook called with each check's result. |
| `name` | `str` | `'grounding_plugin'` | Name of this plugin instance. |

### OnResult

`on_result` receives the callback context, the response with any citations already attached, and the `GroundingResult`. It runs for every final text response, including checks skipped because there are no facts and checks that failed, so it can apply a policy such as replacing an unsupported answer. Return `None` to keep the response, or an `LlmResponse` to replace it. A replacement skips later plugins and agent callbacks, as a response returned from `after_model_callback` does.

### CheckGroundingClient

`CheckGroundingClient` takes the following options:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `project_id` | `str` | `GOOGLE_CLOUD_PROJECT` | Google Cloud project of the grounding config. |
| `location` | `str` | `'global'` | Location of the grounding config. |
| `grounding_config` | `str` | `'default_grounding_config'` | Name of the grounding config resource. |
| `citation_threshold` | `float` | `0.6` | Minimum confidence, from 0 to 1, for a fact to be cited. |
| `enable_claim_level_score` | `bool` | `True` | Whether the API scores each claim. |
| `limits` | `GroundingLimits` | API limits | Maximum facts per check and characters per fact. |
| `client` | `GroundedGenerationServiceAsyncClient` | `None` | Pre-built API client, for example with custom credentials. |

A higher `citation_threshold` cites fewer facts, each with stronger support. A regional `location` sends requests to that region's endpoint. Leave `limits` at its default unless your grounding config enforces different limits.

## Advanced applications

### Replace unsupported answers

Use `on_result` to return a fallback when the facts support too little of an answer. An answer with no checkable claims, such as a greeting, has no claims to hold to the threshold:

```python
from google.adk.integrations.discovery_engine import GroundingPlugin
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def enforce_grounding(*, callback_context, llm_response, grounding_result):
  score = grounding_result.support_score
  if grounding_result.num_claims and score is not None and score < 0.5:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="I could not verify that answer.")],
        )
    )
  return None


plugin = GroundingPlugin(on_result=enforce_grounding)
```

### Use a custom GroundingClient

Pass any object that implements `GroundingClient`: a `limits` property, an async `check_grounding_async(*, answer_candidate, facts)` that returns a `GroundingResult`, and an async `close()`. A client you pass in stays open when the runner closes, so you can share it; close it yourself. The plugin closes only the default client it creates.

## Limitations

- The plugin is experimental and its API may change between releases. The `CHECK_GROUNDING` feature is on by default, so the first construction logs a one-time experimental warning, and construction raises `RuntimeError` when the feature is disabled.
- Facts are fixed at construction and apply to every session and user the plugin serves. Do not pass documents retrieved for one request; session- and turn-scoped facts follow in a separate change.
- The CheckGrounding API rejects answers over its token limit. Such checks return `REJECTED_TOO_LONG`, and the response keeps no citations.
- Responses that contain a function call are not checked. With `output_schema` and tools, an agent returns its final answer through the `set_model_response` function call, so that answer is not checked.
- `AgentTool` passes the parent's plugins to the wrapped agent by default, so each sub-agent answer gets its own check, and its own API call, before the parent answers. Use `AgentTool(include_plugins=False)` to check only the parent, but note that this also drops every other plugin for that sub-agent.
- When the agent has a tool with `propagate_grounding_metadata=True`, such as `GoogleSearchAgentTool`, ADK replaces the response's grounding metadata with that tool's after the plugin runs. The check's citations then reach your application only through `on_result`.
