# ServiceTier

`ServiceTier` specifies the serving tier for model requests executed through the Google GenAI Interactions API. It allows invocations to select standard handling or queued background execution on off-peak capacity through `ServiceTier.DEFERRED`.

> [!NOTE]
> The deferred serving tier is an experimental feature that requires project allowlisting. Please fill out the access request form to get your Google Cloud project allowlisted before enabling `ServiceTier.DEFERRED`.

## Introduction

Model workloads have different latency expectations and cost tolerances. Interactive conversational assistants require immediate responses, while background batch processing, document summarization, and bulk evaluation can run when computing resources are available at lower cost.

The Google GenAI Interactions API exposes serving tiers per request rather than as an immutable configuration on the model class. Configuring the tier per invocation allows a single agent definition to serve both interactive requests with standard capacity and offline batch runs with deferred capacity.

In ADK, `RunConfig` introduces the `service_tier` setting to control this choice. Setting `service_tier=ServiceTier.DEFERRED` places each model call into an off-peak queue on the backend. When capacity opens up, the model processes the request and stores the result.

The transport layer automates the retrieval of queued work. When an application initiates a deferred run, the backend acknowledges the request with an interaction identifier in a pending status. ADK polls the interaction with backoff until the run reaches a final status, converting the finished interaction into standard model response events. Callers consume events through the standard runner interface without managing poll loops or tracking interaction identifiers.

## Get started

Pass `ServiceTier.DEFERRED` to `RunConfig.service_tier` when executing an agent that uses the Interactions API.

The following code configures an agent with `Gemini(use_interactions_api=True)` and executes an invocation using deferred capacity:

```python
import asyncio

from google.adk.agents import LlmAgent
from google.adk.agents import RunConfig
from google.adk.apps import App
from google.adk.models import ServiceTier
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

root_agent = LlmAgent(
    name='batch_agent',
    model=Gemini(use_interactions_api=True),
    instruction='Process input documents and produce summaries.',
)

app = App(
    name='batch_app',
    root_agent=root_agent,
)

runner = InMemoryRunner(app=app)


async def main() -> None:
  session = await runner.session_service.create_session(
      app_name=app.name,
      user_id='user_123',
      session_id='session_456',
  )

  run_config = RunConfig(service_tier=ServiceTier.DEFERRED)

  async for event in runner.run_async(
      user_id='user_123',
      session_id=session.id,
      new_message=types.Content(
          role='user',
          parts=[
              types.Part.from_text(
                  text='Summarize quarterly performance metrics.'
              )
          ],
      ),
      run_config=run_config,
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(part.text)


asyncio.run(main())
```

The runner submits the model call with background execution enabled, waits for the queued interaction to complete, and yields the final response event.

During the wait, `runner.run_async` yields no events while the interaction remains in `queued` or `in_progress` status. In user interfaces or terminal runners, this delay appears as inactivity until the interaction completes. In practice, a deferred invocation may take around thirty-six seconds end to end compared to three to four seconds for an untiered run, with poll requests logged at intervals of five seconds, ten seconds, and twenty seconds.

To confirm that the tier took effect, inspect application logs:

- When debug logging is enabled, the request processor logs `Using service_tier from run_config: deferred`. The interactions transport logs `Interaction <id> is queued; waiting for the result.` at `INFO` level, per-poll updates such as `Interaction <id> is queued.` at `DEBUG` level, and `Interaction <id> reached status completed.` at `INFO` level upon completion.
- When a model is not configured for the Interactions API, ADK drops the tier and logs a warning: `run_config.service_tier='deferred' has no effect for agent <name>: its model does not use the interactions API, which is the only path with a serving tier. Set use_interactions_api=True on the model to apply the tier.` Searching logs for `run_config.service_tier=` or `has no effect for agent` identifies models that need `use_interactions_api=True`.

## How it works

The serving tier takes effect during model invocation and changes how the transport layer transmits the request and handles the response.

Serving tiers are configured per run on `RunConfig` rather than on `LlmAgent` or `Gemini`. Setting the tier on the run keeps model definitions reusable across different invocation patterns, so that interactive callers and automated background jobs can share the same agent. During execution, the runner passes `RunConfig.service_tier` to the invocation context, which forwards the tier to the model request processor.

Only `ServiceTier.DEFERRED` alters the execution model. `ServiceTier.STANDARD` selects standard production capacity and executes synchronously like normal requests, including with streaming. Standard capacity is the default option if `service_tier` is not specified.

When the model uses the Interactions API, the transport includes `service_tier` on the request payload. For `ServiceTier.DEFERRED`, the transport also sets background execution to true. The backend accepts the request into an off-peak queue and immediately returns an interaction object with status `queued` or `in_progress`.

The transport checks the initial interaction status. If the status is final, such as `completed`, the transport yields the response immediately. When the status is `queued` or `in_progress`, the client initiates a polling loop using the interaction identifier:

1. The client sleeps for an initial delay of five seconds.
2. The client fetches the interaction record from the backend.
3. If the interaction remains pending, the client doubles the sleep interval for the next iteration, up to a maximum delay of thirty seconds.
4. When the interaction reports a final status, the loop terminates and yields the converted response event.

The polling schedule avoids adding unnecessary latency to fast executions while restricting request volume during extended queue times. If the interaction status is `requires_action`, the client does not poll. That status indicates that the model has finished its turn and is waiting for tool execution results that only the caller can supply.

Network read errors during polling are absorbed rather than treated as fatal. Because accepted background tasks continue to run and consume resources on the server, dropping the wait on a transient read error would forfeit the result. The client tolerates up to five consecutive read errors before raising an exception, resetting the error counter upon every successful read.

Cancellation propagates through the polling loop. The delay between polling attempts uses an asynchronous sleep that serves as a cancellation point. If the enclosing task is cancelled, the polling loop terminates promptly with `asyncio.CancelledError`.

ADK does not apply client-side timeouts to the queue wait. The backend enforces a completion timeout on the interaction, which defaults to twenty-four hours. Because ADK provides no mechanism to reattach an invocation to a disconnected interaction, terminating early on the client leaves work running on the server without capturing the result. Callers that require shorter limits can bound the execution task directly.

In multi-turn interactions and workflows with tool calls, each model turn submits a new interaction. An agent that executes tools will queue on off-peak capacity once for each turn, so total execution latency reflects the sum of all queue intervals across turns.

Serving tiers apply exclusively to models that use the Interactions API. Models using standard generation APIs do not support serving tiers. If a caller specifies `service_tier` for a non-interactions model, ADK logs a warning once per run and proceeds with default generation. `ManagedAgent` also ignores `service_tier` because it implements its own interaction loop.

## Configuration options

The serving tier is supplied to an invocation through `RunConfig`.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `service_tier` | `Optional[ServiceTier \| str]` | `None` | Serving tier for model calls in the run. |

`service_tier` selects the capacity pool and execution priority for model requests generated during the invocation. Only `ServiceTier.DEFERRED` modifies the transport workflow to run asynchronously with background polling. `ServiceTier.STANDARD` selects default production capacity, executes synchronously as a standard model call, and supports streaming without restriction.

When left unset as `None`, the transport omits the `service_tier` field from the request payload entirely. Setting `ServiceTier.STANDARD` sends `"standard"` explicitly on the wire. Both options produce equivalent behavior on the backend by directing requests to default production capacity. `ServiceTier.STANDARD` is the default option if `service_tier` is not specified.

`ServiceTier.DEFERRED` queues requests on off-peak capacity. The call waits for available capacity rather than failing when demand is high. ADK automatically polls the queued interaction until completion, so the caller receives finished output events. Deferred execution cannot be combined with streaming.

`ServiceTier.STANDARD` directs requests to default production capacity, and it represents the default option when no tier is specified.

Callers can pass string equivalents, such as `"deferred"` or `"standard"`, in place of enum members. `ServiceTier` subclasses `str`, which preserves serialization compatibility and allows applications to supply new tiers supported by the backend before ADK defines new enum constants.

## Advanced applications

Deferred execution supports batch workflows, client deadline controls, and multiple deployment environments.

### Deploying with Vertex AI Agent Engine

Vertex AI Agent Engine, also known as Agent Runtime, provides the recommended deployment target for deferred workloads that may experience multi-hour queue waits.

When deploying an agent to Agent Engine, configure the invocation with `RunConfig(service_tier=ServiceTier.DEFERRED)`:

```python
from google.adk.agents import RunConfig
from google.adk.models import ServiceTier

# Passed in the invocation run_config on Agent Engine:
run_config = RunConfig(service_tier=ServiceTier.DEFERRED)
```

Because Agent Engine manages execution asynchronously within managed container jobs, runs survive even when client network sessions disconnect. The container logs record the complete execution lifecycle, including the initial queued status, the exponential backoff polling intervals, and the completed result payload.

### Deploying on Google Kubernetes Engine

Deployments hosted on Google Kubernetes Engine using `adk api_server` accept `service_tier` in the request body for `/run` and `/run_sse` endpoints.

The following JSON payload requests deferred execution over `/run`:

```json
{
  "app_name": "batch_app",
  "user_id": "user_123",
  "session_id": "session_456",
  "new_message": {
    "role": "user",
    "parts": [{"text": "Summarize batch results."}]
  },
  "service_tier": "deferred"
}
```

When using `/run_sse`, the request must set `"streaming": false` when providing `"service_tier": "deferred"`. Combining `"service_tier": "deferred"` with `"streaming": true` causes the endpoint to return HTTP 422 Unprocessable Entity.

HTTP requests to `/run` hold the network connection open for the entire duration of the queue wait and model execution. When deploying to GKE, configure ingress controllers and reverse proxies with timeouts large enough to accommodate queue latency:

- **Configure ingress backend timeouts.** Default GKE ingress controllers and load balancers often enforce connection timeouts between thirty and sixty seconds. For deferred endpoints, increase the backend service timeout to match expected maximum queue delays.
- **Client disconnection cancels retrieval.** When an HTTP connection closes or times out, the server disconnect monitor cancels the background polling task. Cancelling stops the client wait, but the interaction continues running and billing on the backend with its final output uncollected.
- **Pod logs display polling status.** The GKE pod logs display the queue wait and poll attempts, demonstrating the transition from queued to completed.

### Enforcing a client deadline

Because a deferred request waits on off-peak capacity, execution duration depends on backend load. When an application must place an upper limit on total elapsed time, wrap the runner execution in an asyncio timeout:

```python
import asyncio
import logging

from google.adk.agents import RunConfig
from google.adk.models import ServiceTier
from google.genai import types

logger = logging.getLogger(__name__)

# Continues from the Get started example using existing runner and session objects:
message = types.Content(
    role='user',
    parts=[types.Part.from_text(text='Generate quarterly financial summary.')],
)
run_config = RunConfig(service_tier=ServiceTier.DEFERRED)

try:
  async with asyncio.timeout(300):
    async for event in runner.run_async(
        user_id='user_123',
        session_id=session.id,
        new_message=message,
        run_config=run_config,
    ):
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            print(part.text)
except TimeoutError:
  logger.error(
      'Client timeout exceeded after 300 seconds. The deferred interaction'
      ' continues to run and incur billing on the backend, but its output'
      ' cannot be retrieved.'
  )
```

Cancelling the task stops the client-side polling loop at the next sleep interval, but does not cancel work on the server. The interaction continues running to completion on the backend, continues to consume capacity and billing, and its result cannot be retrieved. Applications that apply client-side deadlines must treat timed-out turns as forfeited.

## Limitations

Serving tiers have specific API and transport requirements.

- **Experimental feature and allowlist.** The deferred serving tier is an experimental capability that requires Google Cloud project allowlisting. Submit the access request form before running deferred requests in production projects.
- **Requires the Interactions API.** Serving tiers function only with models configured for the Google GenAI Interactions API, such as `Gemini(use_interactions_api=True)`. Standard content generation models ignore `service_tier` and log a warning once per run. `ManagedAgent` also ignores `service_tier` because it manages its own interaction lifecycle.
- **Incompatible with streaming.** `ServiceTier.DEFERRED` cannot be used with streaming. A deferred request returns an interaction identifier rather than a stream of partial chunks. Configuring `service_tier=ServiceTier.DEFERRED` with `streaming_mode=StreamingMode.SSE` raises `pydantic.ValidationError` when the `RunConfig` is constructed.
- **HTTP connection limits.** When driven through `adk api_server`, `/run` holds the HTTP connection open throughout the queue wait. If platform timeouts terminate the connection, such as Cloud Run's sixty-minute limit or ingress proxy timeouts, the server disconnect monitor cancels polling while the interaction continues running and billing on the backend with its result uncollected.
- **Per-turn queue latency.** For agents with tools, each turn that produces a model response submits a separate interaction create request. When capacity is constrained, every tool turn incurs its own queue delay.
- **No in-flight resumption across restarts.** Although the Google GenAI Interactions API permits retrieving interactions by identifier, ADK does not persist in-flight interaction identifiers in session storage or provide an interface to reattach a runner invocation to a disconnected interaction. Stopping a client process abandons the pending poll loop, and a subsequent invocation creates a new interaction rather than retrieving the earlier result.

## Related samples

The following sample demonstrates the Interactions API:

* [Interactions API](../../../../contributing/samples/models/interactions_api/agent.py) - An agent configured with `use_interactions_api=True` calling tools over the Interactions API.
