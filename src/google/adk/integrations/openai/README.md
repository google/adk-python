# OpenAI Integration

This folder contains the integration for OpenAI models in ADK.

## Usage in Code

A bare GPT model ID uses the OpenAI Responses API by default.

```python
from google.adk.agents.llm_agent import LlmAgent

# Create an agent and assign the model
agent = LlmAgent(
    name="my_openai_agent",
    model="gpt-6-astra",
    instruction="You are a helpful assistant.",
)
```

Use `OpenAILlm` to explicitly select the Chat Completions API.

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.openai import OpenAILlm

# Create the OpenAI model instance
openai_model = OpenAILlm(model="gpt-6-astra")

# Create an agent and assign the model
agent = LlmAgent(
    name="my_openai_agent",
    model=openai_model,
    instruction="You are a helpful assistant.",
)
```

Use `OpenAIResponsesLlm` to explicitly select the Responses API.

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.openai import OpenAIResponsesLlm

# Create the OpenAI model instance
openai_model = OpenAIResponsesLlm(model="gpt-6-astra")

# Create an agent and assign the model
agent = LlmAgent(
    name="my_openai_agent",
    model=openai_model,
    instruction="You are a helpful assistant.",
)
```

The `openai:` prefix selects the Responses API and `openaillm:` selects
Chat Completions.

Requires the `openai` Python package and the `OPENAI_API_KEY` environment
variable. Install the package with ADK's `openai` extra:

```bash
pip install "google-adk[openai]"
```

## OpenAI-Compatible Endpoints

To reach a host that speaks the OpenAI API, set `base_url` and `api_key`
directly. `api_key` may be a string or a zero-argument callable (sync or async)
that returns one; the client re-invokes a callable on every request, so a
credential that expires (e.g. a Vertex AI OAuth token) is refreshed for you:

```python
from google.adk.integrations.openai import OpenAILlm

openai_model = OpenAILlm(
    model="my-model",
    base_url="https://my-host.example/v1",
    api_key="...",  # or a callable returning a (possibly refreshed) key
)
```

`OpenAIResponsesLlm` takes the same `base_url` and `api_key` fields.

For anything else the client supports (organization, timeout, retries, custom
headers, ...), build an `AsyncOpenAI` yourself and pass it as `client`. Each
model instance keeps its own client, so one process can talk to several hosts:

```python
from openai import AsyncOpenAI
from google.adk.integrations.openai import OpenAILlm

openai_model = OpenAILlm(
    model="my-model",
    client=AsyncOpenAI(base_url="https://my-host.example/v1", api_key="..."),
)
```

`OpenAIResponsesLlm` takes the same `client` field.

> **Tip:** To send every request to one compatible host, leave `client` unset
> and set `OPENAI_BASE_URL` in the environment. The default client reads it.

## Decisions Classifier

`OpenAIDecisionsClassifier` answers the `Choice`, `Binary`, and `Score`
questions from `google.adk.labs.classifiers` with the OpenAI Decisions API, so
it works anywhere ADK takes a `BaseClassifier`, such as a `ClassifierNode`:

```python
from google.adk.integrations.openai import OpenAIDecisionsClassifier
from google.adk.labs.classifiers import Binary

classifier = OpenAIDecisionsClassifier()  # reads OPENAI_API_KEY
response = await classifier.classify(
    input="Please delete my account and every file in it.",
    questions={
        "deletes_data": Binary(instructions="Does the user ask to delete data?"),
    },
)
response.answers["deletes_data"].probability
```

It takes the same `api_key`, `base_url`, and `client` fields as `OpenAILlm`.
OpenAI can decline to answer a question while answering the others; the
answer to that question is a `RefusalAnswer`.
