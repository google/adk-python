# Reflect And Retry Tool Plugin

`ReflectAndRetryToolPlugin` provides self-healing, concurrent-safe error
recovery for tool failures.

**Key Features:**

- **Concurrency Safe:** Uses locking to safely handle parallel tool
executions
- **Configurable Scope:** Tracks failures per-invocation (default) or globally
  using the `TrackingScope` enum.
- **Extensible Scoping:** The `_get_scope_key` method can be overridden to
  implement custom tracking logic (e.g., per-user or per-session).
- **Granular Tracking:** Failure counts are tracked per-tool within the
  defined scope. A success with one tool resets its counter without affecting
  others.
- **Custom Error Extraction:** Supports detecting errors in normal tool
responses that don't throw exceptions, by overriding the
`extract_error_from_result` method.

## Samples

Here are some sample agents to demonstrate the usage of the plugin.

### Basic Usage

This is a hello world example to show the basic usage of the plugin. The
`guess_number_tool` is hacked with both Exceptions and error responses. With the
help of the `CustomRetryPlugin`, both above error types can lead to retries.

For example, here is the output from agent:

```
I'll guess the number 50. Let's see how it is!
My guess of 50 was too high! I'll try a smaller number this time. Let's go with 25.
My guess of 25 was still too high! I'm going smaller. How about 10?
Still too high! My guess of 10 was also too large. I'll try 5 this time.
My guess of 5 is "almost valid"! That's good news, it means I'm getting very close. I'll try 4.
My guess of 4 is still "almost valid," just like 5. It seems I'm still hovering around the right answer. Let's try 3!
I guessed the number 3, and it is valid! I found it!
```

You can run the agent with:

```bash
$ adk web contributing/samples/plugin_reflect_tool_retry
```

Select "basic" and provide the following prompt to see the agent retrying tool
calls:

```
Please guess a number! Tell me what number you guess and how is it.
```

### Hallucinating tool calls

The "hallucinating_func_name" agent is an example to show the plugin can retry
hallucinating tool calls.

For example, we used the `after_model_callback` to hack a tool call with the
wrong name then the agent can retry calling with the right tool name.

You can run the agent with:

```bash
$ adk web contributing/samples/plugin_reflect_tool_retry
```

Select "hallucinating_func_name" and provide the following prompt to see the
agent retrying tool calls:

```
Roll a 6 sided die
```
