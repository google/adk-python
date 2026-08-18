# Using NVIDIA Models with ADK and LiteLLM

This example demonstrates how to use NVIDIA NIM (NVIDIA Inference Microservices) models with ADK through LiteLLM integration.

For comprehensive information about using NVIDIA models with LiteLLM, refer to the [official LiteLLM documentation](https://github.com/BerriAI/litellm/blob/main/docs/my-website/docs/providers/nvidia_nim.md).

## Setup

### 1. Get an NVIDIA NIM API Key

Use the following procedure to get an NVIDIA NIM API key.

1. Sign up at [NVIDIA AI Foundation](https://www.nvidia.com/en-us/ai-data-science/cloud/)
1. Navigate to API section and generate an API key
1. Copy your API key

### 2. Install LiteLLM

Install LiteLLM by running the following code.

```bash
pip install litellm
```

## Using NVIDIA Models in ADK

### Environment Variables

Set the required environment variables:

```bash
export NVIDIA_NIM_API_KEY="your-nvidia-api-key"
export NVIDIA_NIM_API_BASE="https://integrate.api.nvidia.com/v1/"  # Optional
```

### Code Examples

#### Basic Agent Creation

```python
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm

# Create agent with NVIDIA NIM model
agent = Agent(
    model=LiteLlm(model="nvidia_nim/meta/llama3-8b-instruct"),
    name="nvidia_agent",
    instruction="You are a helpful assistant.",
    description="Agent using NVIDIA model",
)
```

#### Available Models

All NVIDIA NIM models are supported. Use the `nvidia_nim/` prefix:

```python
# Examples of available models
models = [
    "nvidia_nim/meta/llama3-8b-instruct",
    "nvidia_nim/meta/llama3-70b-instruct",
    "nvidia_nim/microsoft/phi-3-medium-4k-instruct",
    "nvidia_nim/mistralai/mistral-7b-instruct",
    "nvidia_nim/google/gemma-7b",
    "nvidia_nim/nvidia/nemotron-4-340b-instruct",
    # ... and many more
]
```

## Integration Details

ADK uses LiteLLM as a wrapper to access NVIDIA NIM models. The integration:

1. **Model Format**: Uses `nvidia_nim/<organization>/<model-name>` format
1. **Authentication**: Requires `NVIDIA_NIM_API_KEY` environment variable
1. **Base URL**: Optional `NVIDIA_NIM_API_BASE` for custom endpoints
1. **Compatibility**: Works with all ADK features (tools, sessions, etc.)
1. **Supported Endpoints**: `/chat/completions`, `/completions`, `/embeddings`
