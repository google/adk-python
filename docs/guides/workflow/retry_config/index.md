# RetryConfig

`RetryConfig` is a configuration class used to define retry policies for workflow nodes, enabling them to automatically handle transient failures.

## Introduction

In distributed systems and AI workflows, transient failures (such as network glitches, rate limits, or temporary service outages) are common. `RetryConfig` allows developers to define a resilient policy for individual nodes. When a node execution fails with a configured exception, the ADK will automatically retrying the node execution according to the specified delay and backoff strategy, before propagating the failure.

Key benefits:
- **Resilience**: Automatically recovers from transient errors.
- **Configurable Backoff**: Supports exponential backoff to avoid overwhelming downstream services.
- **Jitter**: Introduces randomness to retry delays to prevent thundering herd problems.
- **Targeted Retries**: Can be configured to only retry on specific exception types.

## Get started

To enable retries for a node, define a `RetryConfig` and pass it to the node definition.

```python
from google.adk.workflow import RetryConfig, node

# Define a retry configuration
unstable_service_retry = RetryConfig(
    max_attempts=3,          # Try up to 3 times (1 original + 2 retries)
    initial_delay=1.0,       # Wait 1 second before the first retry
    backoff_factor=2.0,      # Double the wait time for subsequent retries (1s, 2s)
    exceptions=[ConnectionError, "TimeoutError"] # Only retry on these exceptions
)

# Apply the retry configuration to a node
@node(retry_config=unstable_service_retry)
async def call_unstable_api(node_input: str):
  # This operation might raise ConnectionError
  return await external_api_client.fetch(node_input)
```

## How it works

When a node configured with `RetryConfig` raises an exception during execution:

1.  **Exception Matching**: The `NodeRunner` catches the exception and checks if it matches any of the types specified in `RetryConfig.exceptions`. If `exceptions` is `None` (the default), it matches all exceptions.
2.  **Attempt Count Check**: It checks if the current attempt count is less than `max_attempts`.
3.  **Delay Calculation**: If a retry is warranted, it calculates the delay:
    $$\text{delay} = \text{initial\_delay} \times (\text{backoff\_factor}^{\text{attempt} - 1})$$
    The delay is capped at `max_delay`.
4.  **Jitter Application**: If `jitter` is enabled (greater than 0.0), a random offset is added to the delay:
    $$\text{delay} = \text{delay} + \text{random}(-jitter \times \text{delay}, jitter \times \text{delay})$$
    The final delay is guaranteed to be non-negative.
5.  **Execution Pause and Retry**: The runner sleeps for the calculated delay and then re-executes the node's logic.

## Configuration options

`RetryConfig` is a Pydantic model with the following fields:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_attempts` | `int \| None` | `5` (if omitted) | Maximum number of attempts, including the original request. If `0` or `1`, retries are disabled. |
| `initial_delay` | `float \| None` | `1.0` | Initial delay before the first retry, in seconds. |
| `max_delay` | `float \| None` | `60.0` | Maximum delay between retries, in seconds. |
| `backoff_factor` | `float \| None` | `2.0` | Multiplier by which the delay increases after each attempt. |
| `jitter` | `float \| None` | `1.0` | Randomness factor for the delay. Set to `0.0` to disable jitter (deterministic delays). |
| `exceptions` | `list[str \| type[BaseException]] \| None` | `None` | Exceptions to retry on. Can be exception classes or their string names. `None` means retry on all exceptions. |

## Advanced applications

### Exception Normalization

You can specify exceptions in `exceptions` using either the class type itself or its string name. This is useful if you want to avoid importing the exception class at the node definition site, or if the exception is dynamically defined.

```python
retry_config = RetryConfig(
    exceptions=[ValueError, "CustomNetworkError"]
)
```

### Deterministic Delays for Testing

For testing purposes, you might want to disable jitter and set low delays to speed up test execution and ensure deterministic behavior.

```python
test_retry_config = RetryConfig(
    max_attempts=3,
    initial_delay=0.1,
    backoff_factor=1.0,
    jitter=0.0
)
```

## Limitations

- **State Persistence**: The retry attempt counter is maintained in memory during the execution loop. If the workflow is interrupted (e.g., waiting for a human-in-the-loop input downstream, or if the application restarts) and subsequently resumed, the retry attempt count for the node is **not** persisted. When resumed, if the node needs to run again, the attempt count starts back at 1.
- **Local Retries**: Retries happen locally within the node execution. If a node fails all its retries, the node enters the `FAILED` state, and the workflow execution fails (or follows error handling paths if configured).

## Related samples

- [Resilient Nodes with RetryConfig](../../../../contributing/samples/workflows/retry/)
