# LlmAgent

`LlmAgent` (aliased as `Agent`) is the primary reasoning and conversational agent class in ADK, orchestrating language models, tool executions, structured outputs, and multi-agent delegations.

## Introduction

While `BaseAgent` provides the abstract container lifecycle, sub-agent tree discovery, and workflow node integration, `LlmAgent` provides the concrete implementation powered by large language models.

An `LlmAgent` manages the complete **Thought-Action-Observation** loop:
* Formulates prompt requests containing system instructions, conversation history, and tool declarations.
* Calls the configured generative language model (e.g. Gemini).
* Automatically handles function call requests by invoking registered Python tools and returning tool outputs to the model.
* Validates final responses against a declared `output_schema` (such as a Pydantic model) or persists outputs to session state using `output_key`.
* Manages autonomous transfers across hierarchical `sub_agents`.

The framework exposes `Agent = LlmAgent` at the package root (`from google.adk import Agent`) and within the agents module (`from google.adk.agents import Agent, LlmAgent`) as the standard entry point for building intelligent components.

## Get started

An `LlmAgent` binds an LLM model, behavioral instructions, and Python tools together into an executable agent.

The example below configures an `LlmAgent` with a tool and dynamic instruction placeholders, then executes it within an application runner.

```python
import asyncio

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def get_current_weather(city: str) -> str:
  """Returns current weather conditions for a given city.

  Args:
    city: The city name.
  """
  return f"The weather in {city} is sunny and 22°C."


async def main() -> None:
  # Configure LlmAgent with instructions, model, and tools
  agent = LlmAgent(
      name="weather_assistant",
      model="gemini-2.5-flash",
      instruction="You are a helpful assistant for user {user_name}.",
      tools=[get_current_weather],
  )

  app = App(name="weather_app", root_agent=agent)
  session_service = InMemorySessionService()
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="weather_app",
      user_id="user_42",
      state={"user_name": "Alice"},
  )

  message = types.Content(
      role="user",
      parts=[types.Part.from_text(text="What is the weather in Paris?")],
  )

  async for event in runner.run_async(
      user_id="user_42",
      session_id=session.id,
      new_message=message,
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print("Output:", part.text)


if __name__ == "__main__":
  asyncio.run(main())
```

When invoked, the runner resolves `{user_name}` from session state into the dynamic instruction, passes tool definitions to the model, and executes tool calls before returning responses.

## How it works

The execution lifecycle of an `LlmAgent` is driven by a multi-step reasoning loop:

```
┌─────────────────────────────────────────────────────────────┐
│                      LlmAgent Run Loop                      │
│                                                             │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────┐ │
│   │ Prompt Format │ ──> │ Model Request │ ──> │ Response  │ │
│   └───────────────┘     └───────────────┘     └─────┬─────┘ │
│                                                     │       │
│           ┌─────────────────────────────────────────┴───┐   │
│           │                                             │   │
│     [Function Call]                             [Final Content]
│           │                                             │   │
│           ▼                                             ▼   │
│   ┌───────────────┐                             ┌───────────┐
│   │ Execute Tools │ ──(Feed Observation)──> ... │ Validate  │
│   └───────────────┘                             │ Schema    │
│                                                 └───────────┘
└─────────────────────────────────────────────────────────────┘
```

1. **Prompt preparation**: Combines system instructions, relevant session events, and converted tool definitions into an `LlmRequest`. Any `{state_variable}` in `instruction` is substituted from session state.
2. **Model generation**: Invokes the configured model or model plugin chain.
3. **Tool dispatch**: If the model response contains function calls, the agent dispatches each call to the corresponding `BaseTool` or Python function, captures the returned payload, and loops back to step 1 to continue reasoning.
4. **Completion or delegation**:
   * If the agent transfers to another agent, execution shifts to the target sub-agent.
   * If an `output_schema` is configured, the final text response is validated against that schema.
   * If `output_key` is set, the response is committed to session state under that key.

### Dynamic vs. static instructions and context caching

`LlmAgent` supports two instruction mechanisms to optimize performance:

* `instruction`: Dynamic instructions evaluated on every turn. Can be a string with `{placeholder}` variables or a dynamic `InstructionProvider` callable. When `static_instruction` is not specified, this text becomes the model's `system_instruction`.
* `static_instruction`: Fixed, unchanging content (documentation, reference sheets, large schemas) that contains no runtime variables. When specified, `static_instruction` is sent as `system_instruction` at the beginning of the request, and dynamic `instruction` is placed after it, enabling model providers (such as Gemini) to leverage prompt context caching.

### Delegation modes

The `mode` parameter governs how `LlmAgent` interacts within a system:

* `mode="chat"` (default for root and sub-agents): Standard multi-turn interactive conversation with user history tracking and autonomous peer/parent agent transfers.
* `mode="task"`: Scoped agent that converses with the user to accomplish a designated task, using task-specific isolation scopes.
* `mode="single_turn"` (default when used as a workflow node): Executes a single turn without inspecting previous conversation history (`include_contents="none"`).

### Multi-agent hierarchy and transfer controls

Agents can declare child agents via `sub_agents=[agent_a, agent_b]`. During reasoning, the model can automatically invoke a transfer to any sub-agent. Transfer permissions can be restricted using:
* `disallow_transfer_to_parent=True`: Disallows LLM-controlled transfers to the parent agent during reasoning. Control still automatically returns to the parent on the next turn to prevent one-way isolation.
* `disallow_transfer_to_peers=True`: Disallows LLM-controlled transfers to sibling agents on the same tier.

## Configuration options

The `LlmAgent` class provides configuration fields and properties organized across several functional domains:

### Core model and instructions

Fields controlling model resolution, prompt instructions, and generation parameters.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `model` | Field | `str \| BaseLlm` | Model name (e.g. `'gemini-2.5-flash'`) or `BaseLlm` instance. Defaults to `'gemini-3.5-flash'`. |
| `instruction` | Field | `str \| InstructionProvider` | Dynamic instructions with runtime `{state_variable}` substitution. |
| `static_instruction` | Field | `ContentUnion \| None` | Immutable system instruction enabling prompt context caching. |
| `generate_content_config` | Field | `GenerateContentConfig \| None` | Generation settings including temperature, top_p, and safety settings. |

### Tools and structured output

Settings for binding capabilities, parsing input, and constraining output formats.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `tools` | Field | `list[ToolUnion]` | Tools available to the agent (callables, `BaseTool`, `BaseToolset`, or `NodeTool`). |
| `input_schema` | Field | `type[BaseModel] \| None` | Expected input schema when the agent is invoked as a tool. |
| `output_schema` | Field | `SchemaType \| None` | Pydantic model or schema type enforcing structured output. |
| `output_key` | Field | `str \| None` | Session state key where the final agent output is automatically stored. |

### Delegation and execution modes

Properties governing workflow behavior and multi-agent delegation boundaries.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `mode` | Field | `'chat' \| 'task' \| 'single_turn' \| None` | Execution mode governing history inclusion and transfer behaviors. |
| `sub_agents` | Field | `list[BaseAgent]` | Child agents available for autonomous transfers. |
| `disallow_transfer_to_parent` | Field | `bool` | Disallows LLM-controlled transfer to parent; control returns to parent on next turn. |
| `disallow_transfer_to_peers` | Field | `bool` | Disallows LLM-controlled transfer to sibling agents. |
| `include_contents` | Field | `'default' \| 'none'` | Controls whether historical session events are included in model requests. |

### Advanced executors and planners

Pluggable engines for multi-step reasoning, thinking budgets, and sandboxed code execution.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `planner` | Field | `BasePlanner \| None` | Planning component (e.g. `BuiltInPlanner` or `PlanReActPlanner`). |
| `code_executor` | Field | `BaseCodeExecutor \| None` | Code execution environment (e.g. `BuiltInCodeExecutor` or sandbox). |

### Callbacks and interception pipeline

Lifecycle hooks for observing or altering model requests, responses, and tool calls.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `before_model_callback` | Field | `BeforeModelCallback \| None` | Hook invoked prior to calling the LLM; can inspect or replace `LlmRequest`. |
| `after_model_callback` | Field | `AfterModelCallback \| None` | Hook invoked after receiving an `LlmResponse`; can transform or replace output. |
| `on_model_error_callback` | Field | `OnModelErrorCallback \| None` | Error recovery handler for model call failures. |
| `before_tool_callback` | Field | `BeforeToolCallback \| None` | Hook executed before dispatching a tool; can intercept arguments. |
| `after_tool_callback` | Field | `AfterToolCallback \| None` | Hook executed after tool execution completes; can modify tool results. |
| `on_tool_error_callback` | Field | `OnToolErrorCallback \| None` | Error recovery handler for tool execution failures. |

## Advanced applications

### Joint tool usage and structured output

ADK allows combining `tools` with `output_schema`. The agent uses tools freely during its reasoning loop, and the framework enforces the structured Pydantic schema only on the final model turn.

```python
from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field


class MarketAnalysis(BaseModel):
  company: str = Field(description="Company name")
  sentiment: str = Field(description="Market sentiment: Bullish, Bearish, or Neutral")
  key_drivers: list[str] = Field(description="Primary financial drivers")


def search_financial_filings(ticker: str) -> str:
  """Retrieves latest financial filing excerpts for a stock ticker."""
  return f"Quarterly filing for {ticker}: revenue increased 14%, margins expanded."


# The agent can invoke tools during thought, and formats final output as MarketAnalysis
analyst_agent = LlmAgent(
    name="analyst",
    model="gemini-2.5-flash",
    instruction="Analyze company performance using available financial filings.",
    tools=[search_financial_filings],
    output_schema=MarketAnalysis,
)
```

The model calls `search_financial_filings` to retrieve financial data. Once information gathering is complete, ADK applies `MarketAnalysis` as the response schema, guaranteeing a type-safe Pydantic result.

### Context caching with static instructions

For workloads with massive system instructions, API documentation, or domain guidelines, you can configure `static_instruction` to leverage Gemini's context caching.

```python
from google.adk.agents import LlmAgent

# Large static reference documentation
STATIC_DOCUMENTATION = """
[Comprehensive System API Documentation v3.2]
... extensive API specs, schema rules, and behavioral constraints ...
"""

cached_agent = LlmAgent(
    name="api_assistant",
    model="gemini-2.5-flash",
    static_instruction=STATIC_DOCUMENTATION,
    instruction="Answer user questions using the API documentation.",
)
```

`static_instruction` is placed at the front of the model prompt and remains unchanged across requests, qualifying for cache hits and reducing latency.

### Intercepting requests with model callbacks

You can use `before_model_callback` to inspect or modify prompts and parameters before they reach the provider.

```python
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.models import LlmResponse


def add_compliance_header(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
  """Injects a compliance audit tag into model metadata."""
  # Append audit metadata to the outgoing LLM request config
  if llm_request.config:
    if llm_request.config.labels is None:
      llm_request.config.labels = {}
    llm_request.config.labels["audit_user"] = callback_context.user_id

  # Returning None allows the standard LLM call to proceed normally
  return None


audited_agent = LlmAgent(
    name="audited_agent",
    model="gemini-2.5-flash",
    instruction="Standard customer support assistant.",
    before_model_callback=add_compliance_header,
)
```

## Limitations

* **Agents cannot be passed directly as tools**: Passing another `BaseAgent` or `LlmAgent` directly into `tools=[other_agent]` raises a `ValueError`. `NodeTool` also explicitly rejects `BaseAgent` instances (`Agent '<name>' cannot be wrapped as a NodeTool`). To delegate to another agent, register it in `sub_agents` for autonomous multi-agent transfers.
* **Static instructions and Live API**: `static_instruction` is designed for text and multimodal content caching in the standard GenerateContent API. The Multimodal Live API uses a distinct session resumption and cache architecture.
* **Generation settings location**: Hyperparameters such as `temperature`, `max_output_tokens`, and `safety_settings` must be passed via `generate_content_config=types.GenerateContentConfig(...)`. Direct keyword arguments for generation parameters on `LlmAgent` are not supported.

## Related samples

- [Output Schema with Tools](../../../../contributing/samples/tools/output_schema_with_tools/agent.py) - Demonstrates simultaneous tool calling and Pydantic structured output validation.
- [Three-Layer Transfer](../../../../contributing/samples/multi_agent/three_layer_transfer/agent.py) - Demonstrates multi-agent transfers across root, child, and grandchild agents.
- [Fields Planner](../../../../contributing/samples/patterns/fields_planner/agent.py) - Demonstrates configuring planners on LlmAgent.
- [Context Cache Analysis](../../../../contributing/samples/context_management/cache_analysis/agent.py) - Demonstrates context caching with instructions and content.
