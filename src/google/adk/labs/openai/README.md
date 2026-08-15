# OpenAI Integration (Experimental)

This folder contains an experimental integration for OpenAI models in ADK.

## Usage in Code

To use the OpenAI integration in your Python code, instantiate `OpenAILlm` and assign it to your agent's `model` field:

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.labs.openai import OpenAILlm

# Create the OpenAI model instance
openai_model = OpenAILlm(model="gpt-4o")

# Create an agent and assign the model
agent = LlmAgent(
    name="my_openai_agent",
    model=openai_model,
    instruction="You are a helpful assistant.",
)
```

Requires the `openai` Python package and `OPENAI_API_KEY` environment variable.

## OpenAI-Compatible Endpoints

To reach a host that speaks the OpenAI API, or to configure anything else the
client supports, build an `AsyncOpenAI` yourself and pass it as `client`. Each
model instance keeps its own client, so one process can talk to several hosts:

```python
from openai import AsyncOpenAI
from google.adk.labs.openai import OpenAILlm

openai_model = OpenAILlm(
    model="my-model",
    client=AsyncOpenAI(base_url="https://my-host.example/v1", api_key="..."),
)
```

`OpenAIResponsesLlm` takes the same `client` field.

> **Tip:** The OpenAI Python client also honors `OPENAI_BASE_URL` for OpenAI-compatible multi-model gateways — for example [DaoXE](https://daoxe.com/?utm_source=github&utm_medium=organic&utm_campaign=adk-python&utm_content=openai-labs) at `https://api.daoxe.com/v1`.
