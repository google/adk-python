# BaseTool

Every ADK tool derives from the `BaseTool` class. You subclass it when you want
to write a tool's declaration yourself, keep configuration on the tool object,
or change the request that goes to the model.

## Introduction

The [`FunctionTool`](../function_tool/index.md) class builds a tool from a
Python function, and it reads the declaration from the function's signature and
docstring. That approach works whenever a Python signature can describe the
tool. You subclass `BaseTool` directly when it cannot, or when a class fits the
tool better than a function does. The following situations are typical.

- The tool's parameters come from somewhere other than Python code, such as an
  OpenAPI document, an MCP server, or a configuration file. ADK builds its own
  `RestApiTool` and `McpTool` classes this way.
- The tool holds something that you set up once and reuse on every call, such
  as an API client, a lookup table, or a list of allowed operations.
- The tool changes the model request and runs no code of its own when the model
  answers. The `GoogleSearchTool` class adds Gemini's built-in search to the
  request, and the `ExampleTool` class adds few-shot examples to the
  instructions.

## Get started

The following example defines a tool that looks up a city's timezone in a
table that you pass to its constructor. The tool writes its own declaration and
runs locally when the model calls it.

> [!NOTE]
> You would not normally write this tool as a class. A plain function, such as
> `def get_timezone(city: str) -> dict`, in the agent's `tools` list gives the
> model the same tool, and ADK writes the declaration for you from the
> function's signature and docstring. The example only shows how the `BaseTool`
> class works, so that you can follow the same steps for a tool that a function
> cannot describe.

```python
from typing import Any

from google.adk import Context
from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool
from google.genai import types

class GetTimezoneTool(BaseTool):
  """Looks up the timezone of a city in a fixed table."""

  def __init__(self, timezones: dict[str, str]):
    super().__init__(
        name="get_timezone",
        description="Returns the IANA timezone name for a city.",
    )
    self._timezones = timezones

  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name."},
            },
            "required": ["city"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: Context
  ) -> dict[str, str]:
    city = args.get("city", "")
    timezone = self._timezones.get(city)
    if timezone is None:
      return {"error": f"No timezone is known for {city}."}
    return {"city": city, "timezone": timezone}

root_agent = LlmAgent(
    name="travel_agent",
    instruction="Answer questions about places. Use get_timezone for timezones.",
    tools=[GetTimezoneTool({"Paris": "Europe/Paris", "Tokyo": "Asia/Tokyo"})],
)
```

You run this agent the same way as any other, for example with `adk web`. Before
you write your own tool, look at the following details in the class.

First, every subclass calls `super().__init__` with a `name` and a
`description`, because the base class requires both. The same call accepts the
optional settings described under [Configuration options](#configuration-options).

Second, the name in the declaration has to match `self.name`. The model calls
the tool by the name it finds in the declaration, and ADK looks the tool up by
`self.name`. If the two differ, the model's call matches no tool, and ADK
answers the model with an error that lists the tools it can call. If you build
the declaration from `self.name` and `self.description`, as the example does,
the declaration always matches the tool. Although its name starts with an
underscore, the `_get_declaration` method is the one that ADK expects a subclass
to override when the tool has a declaration.

Third, the `run_async` method receives the arguments in `args` as the model
sent them, with no conversion or validation. The example reads `city` with
`args.get` and returns a dictionary with an `error` key when it cannot answer.
The model reads that dictionary as the tool's result, so it can tell the user or
ask for a different city. Keep the keyword-only signature of `run_async`,
because ADK passes `args` and `tool_context` by name.

## How it works

The `BaseTool` class has main methods that a subclass can override, and the
methods you override decide what kind of tool you build. Each one has a default,
so ADK does not force you to override any of them.

- The `_get_declaration()` method returns the `types.FunctionDeclaration` that
  describes the tool to the model, or `None`. The default returns `None`.
- The `run_async(*, args, tool_context)` method runs the tool when the model
  calls it and returns the result. The default raises `NotImplementedError`.
- The `process_llm_request(*, tool_context, llm_request)` method runs before
  every model call. The default asks `_get_declaration()` for a declaration
  and, if it gets one, adds it to the request under `self.name`.

A subclass therefore usually takes one of the following forms. A tool that the
model calls returns a declaration from `_get_declaration()`, implements
`run_async`, and keeps the default `process_llm_request`. A tool that only
changes the request returns no declaration, does not implement `run_async`, and
does its work in its own `process_llm_request`.

### Errors raised by `run_async`

If `run_async` raises an exception, ADK runs the plugin `on_tool_error_callback`
callbacks and then the agent's `on_tool_error_callback`. If one of them returns
a result, that result goes to the model. Otherwise the exception propagates to
the code that is running the agent and ends the invocation. When you want the
model to see a failure and recover from it, return a dictionary with an `error`
key instead of raising.

## Configuration options

The constructor arguments that `BaseTool` introduces are all keyword-only, and
each one is also a plain attribute that you can change after construction.

### Identity

The `name` and `description` arguments are required. The name is the key that
ADK uses to find the tool when the model calls it, so keep it unique among an
agent's tools, for the reason that [Limitations](#limitations) describes. ADK
stores the description on the tool, but only your `_get_declaration()` method
decides whether the model reads it, so when you build the declaration by hand,
pass `self.description` into it.

### Long-running calls

Set the `is_long_running` argument to `True` for a tool that starts work that
finishes later, outside the call. If `run_async` then returns a falsy value,
such as `None` or an empty dictionary, ADK sends no response to the model, and
your application sends the response later. The
[`LongRunningFunctionTool` guide](../long_running_tool/index.md) explains how
your application finds the pending call and sends that response.

### Live API settings

The `behavior` and `response_scheduling` arguments apply to the Live API. They
decide whether the model keeps talking while the tool runs and when it reacts to
the result. The [live tools guide](../../live/tools/index.md) explains both
settings.

## Advanced applications

The `BaseTool` class has other methods that you can override when a tool needs
more than a declaration and a `run_async` implementation.

### A tool that changes the request

To add instructions or a model-side capability to every request, override
`process_llm_request` and keep the defaults of `_get_declaration` and
`run_async`. Because the tool contributes no declaration, the model does not see
it as a tool that it can call. The following tool adds a company's writing rules
to the instructions.

```python
from google.adk import Context
from google.adk.models import LlmRequest
from google.adk.tools import BaseTool

class HouseStyleTool(BaseTool):
  """Adds the company's writing rules to every request."""

  def __init__(self, rules: str):
    super().__init__(name="house_style", description="Adds writing rules.")
    self._rules = rules

  async def process_llm_request(
      self, *, tool_context: Context, llm_request: LlmRequest
  ) -> None:
    llm_request.append_instructions([self._rules])
```

ADK calls `process_llm_request` before every model call, and each model call
starts from a new request, so the method adds the rules once to each request,
and they do not accumulate. Any work the method does, such as reading a file or
calling a service, happens before every model call and adds its latency to each
one.

A tool that writes to the system instruction also affects prompt caching.
Context caching reuses a cached prompt prefix only while the system instruction
and the tool declarations stay the same. Text that changes from one request to
the next, such as text built from session state or from the user's latest
message, causes a cache miss on every request, and the model processes the whole
prompt again each time. The
[`ExampleTool` guide](../../examples/example/index.md) describes the same
trade-off for few-shot examples.

A tool can also keep its declaration and change the request. Override
`process_llm_request`, call `super().process_llm_request(...)` first so that the
default still adds the declaration, and then make your changes.

### Asking for approval on some calls

Override `check_require_confirmation` to decide for each call whether a person
must approve it first. ADK calls it before `run_async`, and when it returns
`True`, ADK pauses the call and asks for approval. The following override asks
for approval only when the `amount` argument is above 1,000.

```python
  # Add to your BaseTool subclass, with the imports from the first example.
  async def check_require_confirmation(
      self, args: dict[str, Any], tool_context: Context
  ) -> bool:
    return args.get("amount", 0) > 1000
```

ADK treats only the value `True` as a request for approval, so return a real
`bool`. The method receives the `tool_context`, so it can base the decision on
session state as well as on the arguments. The
[tool confirmation guide](../tool_confirmation/index.md) explains how your
application receives the request and sends back the answer.

## Limitations

A duplicate name shadows a tool without raising an error. If two tools on one
agent have the same `name`, ADK logs a warning, both declarations still reach
the model, and only the tool that you added last runs when the model calls that
name. Give each tool a distinct name, or set `tool_name_prefix` on a toolset as
the [`BaseToolset` guide](../base_toolset/index.md) describes.

## Related samples

- [Skills agent](../../../../contributing/samples/environment_and_skills/skills_agent/agent.py)
  defines `GetTimezoneTool`, a `BaseTool` subclass that writes its own
  `_get_declaration` and `run_async`, next to a plain function tool.
- [Context offloading with artifacts](../../../../contributing/samples/patterns/context_offloading_with_artifact/agent.py)
  has `QueryLargeDataTool`, which overrides `process_llm_request`, calls the
  default first so that its declaration is kept, and then adds an artifact's
  content to the request.
- [GEPA agent](../../../../contributing/samples/integrations/gepa/adk_agent.py)
  is a `BaseTool` subclass that sets `tool_context.actions.skip_summarization`
  from inside `run_async`.
