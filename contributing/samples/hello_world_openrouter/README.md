# OpenRouter with LiteLLM

This sample shows how to use an OpenRouter-hosted model with ADK through the
existing LiteLLM model connector.

## Setup

Install ADK with optional integrations so that `LiteLlm` is available:

```bash
pip install "google-adk[extensions]"
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="..."
```

Optionally choose a model:

```bash
export OPENROUTER_MODEL="openrouter/openai/gpt-5.2"
```

Run the sample:

```bash
adk run contributing/samples/hello_world_openrouter
```

## Notes

OpenRouter is used here through LiteLLM's OpenAI-compatible routing path:

```python
LiteLlm(
    model="openrouter/openai/gpt-5.2",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base="https://openrouter.ai/api/v1",
)
```

For Gemini models routed through OpenRouter, use OpenRouter model IDs such as
`openrouter/google/gemini-2.5-pro:online`. ADK's built-in Google tools are
optimized for native Gemini model connections, so verify tool compatibility for
the routed model and provider you select.

