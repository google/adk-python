# LiteLLM Integration

## Loop Prevention

When using LiteLLM with certain models (particularly Ollama/Gemma3), be aware that the system includes loop detection to prevent infinite function call loops. The loop detection triggers when the same function is called consecutively more than 5 times.

If your application legitimately needs to call the same function more than 5 times in a row, you can adjust the `_loop_threshold` value in the `LiteLlm` class. However, this is generally not recommended as repeated calls to the same function are often a sign of an issue with the model's understanding or the function's implementation.

For more details on this feature, see [LiteLLM Loop Fix Documentation](./litellm_loop_fix.md).

# Additional Topics 