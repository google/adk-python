# LiteLLM Agent Example

## What is LiteLLM?

LiteLLM is a Python library that provides a unified interface for interacting with multiple Large Language Model (LLM) providers through a single, consistent API. It serves as an adapter that allows you to:

- Use the same code to access 100+ different LLMs from providers like OpenAI, Anthropic, Google, AWS Bedrock, and more
- Standardize inputs and outputs across different LLM providers
- Track costs, manage API keys, and handle errors consistently
- Implement fallbacks and load balancing across different models

In essence, LiteLLM acts as a unified wrapper that makes it easy to switch between different LLM providers without changing your application code.

## Why Use LiteLLM with ADK?

The Agent Development Kit (ADK) is designed to be model-agnostic, meaning it can work with various LLM providers. LiteLLM enhances this capability by:

1. **Provider Flexibility**: Easily switch between LLM providers (OpenAI, Anthropic, etc.) without changing your agent code
2. **Cost Optimization**: Choose the most cost-effective model for your specific use case
3. **Model Exploration**: Experiment with different models to find the best performance for your task
4. **Future-Proofing**: As new models are released, you can quickly adopt them without major code changes

This example demonstrates how to use LiteLLM with ADK to create an agent powered by models through OpenRouter rather than Google's Gemini models.

## Limitations When Using Non-Google Models

When using LiteLLM to integrate non-Google models with ADK, there are some important limitations to be aware of:

1. **No Access to Google Built-in Tools**: Non-Google models (like OpenAI, Anthropic, etc.) cannot use ADK's built-in Google tools such as:
   - Google Search
   - Code Execution
   - Vertex AI Search

2. **Custom Function Tools Only**: When using non-Google models, you can only use custom function tools (like the `get_dad_joke()` function in this example).


These limitations exist because built-in tools are specifically designed to work with Google's models and infrastructure. However, you can still create powerful agents using custom function tools and the wide variety of models available through LiteLLM.

## Getting Started

This example uses the same virtual environment created in the root directory. Make sure you have:

1. Activated the virtual environment from the root directory:
```bash
# macOS/Linux:
source ../.venv/bin/activate
# Windows CMD:
..\.venv\Scripts\activate.bat
# Windows PowerShell:
..\.venv\Scripts\Activate.ps1
```

2. Set up your OpenRouter API key:
   - Create an account at [OpenRouter](https://openrouter.ai/) if you don't have one
   - Generate an API key at https://openrouter.ai/keys
   - Rename `.env.example` to `.env` in the openrouter_dad_joke_agent folder
   - Add your OpenRouter API key to the `OPENROUTER_API_KEY` variable in the `.env` file

## Understanding the Code

This example demonstrates:

1. How to use the `LiteLlm` model adapter with ADK
2. How to connect to models through OpenRouter (specifically Claude 3.5 Sonnet)
3. How to create a simple agent with a custom tool

The agent is configured to tell dad jokes using a custom function tool `get_dad_joke()` and powered by Anthropic's Claude 3.5 Sonnet model through OpenRouter instead of Google's Gemini.

## Running the Example

To run the LiteLLM agent example:

1. Navigate to the 3-litellm-agent directory containing your agent folder.

2. Start the interactive web UI:
```bash
adk web
```

3. Access the web UI by opening the URL shown in your terminal (typically http://localhost:8000)

4. Select the "openrouter_dad_joke_agent" from the dropdown menu in the top-left corner of the UI

5. Start chatting with your agent in the textbox at the bottom of the screen

### Example Prompts to Try

- "Tell me a dad joke"

You can exit the conversation or stop the server by pressing `Ctrl+C` in your terminal.

## Modifying the Example

You can easily modify this example to use different models from different providers through OpenRouter by changing the `LiteLlm` configuration. For example:

```python
# To use Claude 3.5 Sonnet from Anthropic through OpenRouter
model = LiteLlm(
    model="openrouter/anthropic/claude-3-5-sonnet",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# To use GPT-4o from OpenAI through OpenRouter
model = LiteLlm(
    model="openrouter/openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# To use Llama 3 70B from Meta through OpenRouter
model = LiteLlm(
    model="openrouter/meta-llama/meta-llama-3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# To use Mistral Large through OpenRouter
model = LiteLlm(
    model="openrouter/mistral/mistral-large-latest",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
```

## Additional Resources

- [Google ADK LiteLLM Integration Documentation](https://google.github.io/adk-docs/tutorials/agent-team/#step-2-going-multi-model-with-litellm-optional)
- [LiteLLM Documentation](https://docs.litellm.ai/docs/)
- [LiteLLM Supported Providers](https://docs.litellm.ai/docs/providers)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Anthropic Claude Models Overview](https://docs.anthropic.com/en/docs/about-claude/models/all-models)
