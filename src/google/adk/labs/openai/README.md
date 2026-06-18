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
