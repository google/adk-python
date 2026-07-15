# DaoXE (OpenAI-compatible multi-model gateway) with ADK

This sample shows using [DaoXE](https://daoxe.com) through ADK's `LiteLlm` wrapper
and LiteLLM's OpenAI-compatible provider.

DaoXE exposes Chat Completions at `https://daoxe.com/v1`. Model IDs are
**account-scoped** — use exact IDs from authenticated `GET /v1/models` or the
dashboard. DaoXE is not available in mainland China.

## Setup

```bash
export OPENAI_API_KEY=your_daoxe_api_key
export OPENAI_API_BASE=https://daoxe.com/v1
# optional alias used by some LiteLLM paths:
export OPENAI_BASE_URL=https://daoxe.com/v1
```

Edit `agent.py` and set `model=LiteLlm(model="openai/<your-account-model-id>")`
to an exact model ID available on your DaoXE account.

## Run

```bash
# from repo root, with ADK installed
adk web
# or
python -m contributing.samples.models.hello_world_daoxe.main
```

## Notes

- Uses OpenAI Chat Completions via LiteLLM (`openai/...` model prefix).
- DaoXE also supports other protocols (e.g. Anthropic Messages) for other clients.
- Contributor disclosure: this sample was contributed by a DaoXE affiliate.
