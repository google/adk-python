# Finish Reason Test Agent

This sample contains a script to verify that the `finish_reason` from a LiteLLM model is correctly propagated to the `LlmResponse` object.

The script is configured to use the `openai/gpt-3.5-turbo` model through LiteLLM. It sets `max_tokens=50` to force the model to stop execution due to length constraints. An `after_model_callback` is used to inspect the `response.finish_reason` and verify that it is `length`.

## Running the test

To run this sample, you will need to have an OpenAI API key set as an environment variable. Then, run the `agent.py` script directly.

```bash
export OPENAI_API_KEY="your-api-key-here"
python agent.py
```
