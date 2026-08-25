# RetryConfig

`RetryConfig` is how you tell a node to try again after it fails. You describe how many attempts it gets, how long to wait between them, and which exceptions qualify, then hand that description to the node.

## Introduction

`RetryConfig` is a per-node retry policy. When a node fails with an exception the config covers, ADK re-runs it following the delay and backoff you asked for, and the failure only reaches the rest of the workflow once the attempts are used up. It offers four things:

- **Resilience**: a node recovers from a transient error without the rest of the workflow seeing it.
- **Configurable backoff**: exponential backoff keeps a struggling downstream service from being called again immediately.
- **Jitter**: randomness in the delay spreads retries out instead of letting them fire together as a thundering herd.
- **Targeted retries**: only the exception types you name are retried.

`retry_config` is one of the options every node carries, so the same object works on a function node, an agent node, a `JoinNode` or a nested `Workflow`. The other node options are in [BaseNode](../base_node/index.md).

## Get started

Build a `RetryConfig` and pass it in where you define the node. This example gives a call to an unreliable API three attempts in total, with the wait doubling in between, and only connection and timeout failures counting as worth retrying.

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

When a node with a `RetryConfig` raises an exception, five things happen in order.

1. **Exception matching.** The exception's class name is checked against `RetryConfig.exceptions`. When `exceptions` is left as `None`, which is the default, every exception matches.
2. **Attempt count check.** The node is retried only while `attempt_count` is below `max_attempts`. That counter starts at 1 for the original run, so `max_attempts=3` buys you the original plus two retries.
3. **Delay calculation.** The base delay grows exponentially with the number of attempts already made, as `initial_delay * backoff_factor ** (attempt - 1)`.
4. **Capping, then jitter.** If `jitter` is greater than 0.0, the base delay is first capped at `max_delay / (1 + jitter)`, and a random offset is then drawn uniformly from `-jitter * delay` to `+jitter * delay`. The order matters and it is deliberate. Capping afterwards would still hold the ceiling, but it would collapse every overshooting draw onto exactly `max_delay`, firing at one instant the very retries that jitter exists to spread out. With the defaults of `jitter=1.0` and `max_delay=60.0`, the pre-jitter delay therefore never exceeds 30 seconds, and the widest positive draw lands on 60. The result is then clamped to be non-negative and capped at `max_delay` once more.
5. **Pause, then retry.** The workflow waits for that delay and then runs the node again. Inside the node, `ctx.attempt_count` tells you which attempt you are on.

## Configuration options

`RetryConfig` is a Pydantic model with the following fields:

| Field            | Type                                       | Default          | Description                                                                                                   |
| :--------------- | :----------------------------------------- | :--------------- | :------------------------------------------------------------------------------------------------------------ |
| `max_attempts`   | `int \| None`                              | `5`              | Maximum number of attempts, including the original request. If `0` or `1`, retries are disabled.              |
| `initial_delay`  | `float \| None`                            | `1.0`            | Initial delay before the first retry, in seconds.                                                             |
| `max_delay`      | `float \| None`                            | `60.0`           | Maximum delay between retries, in seconds.                                                                    |
| `backoff_factor` | `float \| None`                            | `2.0`            | Multiplier by which the delay increases after each attempt.                                                   |
| `jitter`         | `float \| None`                            | `1.0`            | Randomness factor for the delay. Set to `0.0` to disable jitter (deterministic delays).                       |
| `exceptions`     | `list[str \| type[BaseException]] \| None` | `None`           | Exceptions to retry on. Can be exception classes or their string names. `None` means retry on all exceptions. |

Every field is stored as `None` when you leave it out, so `RetryConfig()` prints
as all-`None` rather than showing the numbers above. The Default column is what
the retry logic substitutes for `None` at the moment it runs.

## Advanced applications

Two things come up once a retry policy is in real use: naming the exceptions it should catch, and stopping the delays from slowing your test suite down.

### Exception normalization

An entry in `exceptions` can be the exception class itself or the class name as a string. The string form is there for when you would rather not import the class where the node is defined, or when the exception is defined dynamically and there is nothing to import.

```python
retry_config = RetryConfig(
    exceptions=[ValueError, "CustomNetworkError"]
)
```

A class you pass is normalized to its `__name__` at validation time, and matching at retry time compares `type(exception).__name__` against that list. **The match is on the exact class name, not `isinstance`.** Listing `ConnectionError` therefore does not retry a `ConnectionResetError`, and listing `Exception` retries nothing at all. Name every concrete class you want to catch, or leave `exceptions` as `None` to retry on everything.

### Deterministic delays under test

Under test, retries should be fast and should behave the same way every run, which means turning jitter off and dropping the delays right down.

```python
test_retry_config = RetryConfig(
    max_attempts=3,
    initial_delay=0.1,
    backoff_factor=1.0,
    jitter=0.0
)
```

## Limitations

- **The attempt counter does not survive an interrupt.** It is **not** persisted. If the workflow is interrupted, whether by a human-in-the-loop input downstream or by the application restarting, and is later resumed, a node that has to run again starts counting from 1 with its full budget of attempts back.
- **Retries are local to the node.** They happen inside that node's execution and nowhere else. Once a node has used up its attempts it enters the `FAILED` state, and the workflow execution fails with it, or follows whatever error handling you have configured.

## Related samples

- [Node Retries](../../../../contributing/samples/workflows/retry/agent.py): a node that fails at random, retried five times, reporting `ctx.attempt_count` as it goes.
