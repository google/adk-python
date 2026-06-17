# ADK Workflow Sample: Node Retries

## Overview

In real-world applications, interacting with external APIs, databases, or third-party services can occasionally result in transient failures (e.g., temporary network outages, rate limits, or bad gateways).

The ADK framework allows you to easily handle these scenarios by wrapping the unreliable logic in a `@node` decorator configured with `RetryConfig`. If the node raises one of the expected exceptions, the workflow engine automatically pauses, waits for a backoff delay, and reschedules the node for another attempt.

When a node raises an exception, the framework automatically emits an error event (with `error_code` and `error_message`) so the error is visible in the event stream. If the node has retry configured, it will be retried after the backoff delay.

This sample demonstrates a `get_weather` node that intentionally fails randomly (70% chance) by raising an `HTTPError` representing a 500 Internal Server error. The framework gracefully recovers and eventually succeeds, passing the result to `report_weather`.

## Graph

```mermaid
graph TD
    START --> get_weather[get_weather <br/>Retries on HTTPError]
    get_weather --> report_weather
```

## How To

1. **Import `RetryConfig`**: Ensure you import the configuration class to set your retry parameters.

   ```python
   from google.adk.workflow import RetryConfig
   ```

1. **Configure the Decorator**: Apply the `@node` decorator to your Python function and specify the `retry_config` parameter with your desired logic (e.g., `max_attempts`, `initial_delay`).

   ```python
   @node(retry_config=RetryConfig(max_attempts=5, initial_delay=1))
   def get_weather(ctx: Context) -> str:
       # ... flaky logic here ...
   ```

   When an exception like `HTTPError` occurs, the ADK framework catches it, emits an error event, and processes the backoff delay automatically. As long as `max_attempts` hasn't been exceeded, the node executes again.

1. **Track Retries (Optional)**: If you need to know which attempt the node is currently running, you can access `ctx.attempt_count` from the `Context`.

   ```python
   yield Event(message=f"Getting weather... attempt {ctx.attempt_count}")
   ```
