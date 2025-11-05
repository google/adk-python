# LiteLLM with Fallback Models

This agent is built for resilience using LiteLLM's built-in fallback mechanism. It automatically switches models to guard against common disruptions like token limit errors and connection failures, while ensuring full conversational context is preserved across all model changes.

To run this example, ensure your .env file includes the following variables:
```
GOOGLE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```
