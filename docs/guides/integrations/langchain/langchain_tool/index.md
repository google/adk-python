# LangchainTool

`LangchainTool` wraps a LangChain tool so an ADK agent can call it. It is a
`FunctionTool` subclass that pulls the underlying callable out of the LangChain
object, builds a Gemini function declaration from LangChain's schema, and
honors the one LangChain behavior ADK has an equivalent for, `return_direct`.

The only setup requirement is a `pip install`, with no account and no service
to configure. You do have to install `langchain-core` yourself, because ADK has
no `[langchain]` extra and does not depend on LangChain at runtime.

## Introduction

`LangchainTool` is the adapter that lets an existing LangChain tool keep
working in ADK. Put a wrapped tool in an agent's `tools` list and it behaves
like any other ADK tool: it appears in the model's function declarations, it is
invoked with the arguments the model chose, and its return value goes back as a
function response.

It is a migration path, not a permanent home. The wrapper carries LangChain's
schema conventions into ADK, and the two do not line up perfectly, so the
places they disagree turn up as the limitations listed below. A tool you own is
usually better rewritten as a plain Python function and passed straight to
`tools=`, which is what `FunctionTool` does with no adapter in the way.

## Get started

Wrap the LangChain tool and put it in the agent:

```python
from google.adk.agents import Agent
from google.adk.integrations.langchain import LangchainTool
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel


async def add(x: int, y: int) -> int:
  """Adds two numbers."""
  return x + y


@tool
def minus(x: int, y: int) -> int:
  """Subtracts two numbers."""
  return x - y


class AddSchema(BaseModel):
  x: int
  y: int


add_tool = StructuredTool.from_function(
    add, name="add", description="Adds two numbers", args_schema=AddSchema
)

root_agent = Agent(
    name="calculator",
    description="Answers arithmetic questions.",
    instruction="Use the tools to compute answers rather than doing it yourself.",
    tools=[LangchainTool(tool=add_tool), LangchainTool(tool=minus)],
)
```

Both forms work: a `StructuredTool` built explicitly, and anything LangChain's
`@tool` decorator produced, which is also a `StructuredTool`. An async
LangChain function is wrapped exactly the same way.

Override the name and description when LangChain's are unhelpful, which they
often are for community tools whose name is a class name:

```python
LangchainTool(
    tool=DuckDuckGoSearchRun(),
    name="web_search",
    description="Searches the public web and returns a text summary.",
)
```

## How it works

Two parts of the adapter show up in behavior: how the function declaration is
built, and what happens to a tool marked `return_direct`.

### Build the declaration

If your tool is a `langchain_core.tools.BaseTool` **with an `args_schema`**,
the declaration is built from that schema, so the Pydantic model you gave
LangChain is what the model sees.

If it is anything else, including a `BaseTool` with no `args_schema`, the
declaration comes from the callable's signature and type hints instead.

The difference shows up in the generated declaration, and in an unexpected
place: the two paths populate *different fields*. With an `args_schema` the
parameters land in `FunctionDeclaration.parameters` as a typed `Schema`. Without
one, they land in `parameters_json_schema` as raw JSON Schema and `parameters`
is `None`, so code that inspects a declaration has to check both. The fallback
schema is also titled `_runParams` rather than after your tool. The
declaration's `name` is corrected to the tool name in both cases.

Either path reports a failure as
`ValueError: Failed to build function declaration for Langchain tool: ...`.

### `return_direct`

LangChain's `return_direct=True` means "give the user this tool's output, do not
send it back to the model". ADK's equivalent is
`tool_context.actions.skip_summarization`, and `LangchainTool` sets it after a
successful call on a tool whose `return_direct` is true.

There is one carve-out: when the result is a dict containing an `error` key,
summarization is *not* skipped. That is the shape `FunctionTool` returns
when the model omitted a mandatory argument, and the tool never actually ran, so
the error has to reach the model for it to retry. Concretely, calling a
`return_direct` tool without its required argument returns

```text
{'error': "Invoking `shout()` failed as the following mandatory input
parameters are not present: text ..."}
```

with `skip_summarization` left unset, while a successful call sets it to `True`.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tool` | `LangchainBaseTool \| object` | required | The LangChain tool, or any object with `run` or `_run`. |
| `name` | `str \| None` | `None` | Overrides the tool's name in the function declaration. |
| `description` | `str \| None` | `None` | Overrides the description the model sees. |

Left as `None`, each falls back to the LangChain tool's own value, and then to
whatever the wrapped function itself supplies.

`name` and `description` are the two that earn their place. A LangChain
community tool's own name is frequently the class name and its description is
frequently a docstring aimed at a Python reader, and both go straight into the
prompt. Rewriting them is the cheapest accuracy improvement available on a
wrapped tool.

## Advanced applications

Two questions come up once the first wrapped tool works: how to declare one
outside Python, and when to rewrite a tool instead of wrapping it.

### Declare the tool in YAML

`LangchainTool` supports ADK's config-driven agent loading through
`LangchainToolConfig`, so a wrapped tool can be named in an agent YAML file
rather than constructed in Python:

```yaml
tools:
  - name: google.adk.integrations.langchain.LangchainTool
    args:
      tool: my_package.tools.search_tool
      name: web_search
      description: Searches the public web and returns a text summary.
```

`tool` is the fully qualified path to a LangChain tool **instance**, not a
class; it is resolved by import at load time.

**Always set `name` and `description` in YAML.** They are not optional in
practice, whatever the config schema suggests. `LangchainToolConfig` defaults
both to the empty string, and the constructor only falls back to LangChain's own
values when the argument is `None`. An empty string is not `None`, so it wins:
omit `name` from the YAML above and you get a tool whose name and description
are both `""`, even though the LangChain tool underneath is called
`search_tool`. Nothing rejects it either: the agent accepts the tool and the
model is shown a function with no name.

### Decide what to migrate rather than wrap

Wrapping is the right answer for a tool you did not write and do not want to
own, such as a community search tool or a vendor integration. For a tool that
is your own Python function underneath, the adapter is pure overhead: pass the
function to
`tools=` and ADK builds a `FunctionTool` from its signature and docstring. You
also get things the adapter cannot give you, notably a `tool_context: ToolContext`
parameter for reading session state, saving artifacts, or requesting
confirmation.

## Limitations

*   **`langchain_core` is a hard import.** The module imports it at module
    scope, so `from google.adk.integrations.langchain import LangchainTool`
    fails with `ModuleNotFoundError` when LangChain is not installed. There is
    no optional-dependency guard and no install hint.
*   **Untyped parameters produce an untyped declaration.** A LangChain tool
    whose function has no type hints and no `args_schema` yields a declaration
    whose properties have names and nothing else, leaving the model to guess
    whether an argument is a string or a number. Give the function type hints,
    or give the tool an `args_schema`.
*   **Only `return_direct` carries across.** LangChain's other tool-level
    behavior is not translated, including its callback manager, its error
    handling and its retry configuration. Only the callable and the schema come
    over.
*   **No `ToolContext` for the wrapped function.** The LangChain function is
    called with the model's arguments, so it cannot read session state or save
    an artifact. Anything needing context has to be an ADK tool.
*   **There is no `[langchain]` extra.** LangChain is not a runtime dependency
    of ADK; `langchain-community` appears only in the `test` extra. Installing
    `langchain-core` for production use is on you, and so is pinning it.
*   **Wrapping a non-LangChain object works, and is undefined.** The duck-typed
    fallback accepts any object with `run` or `_run`. That is handy, and it
    means a typo that hands over the wrong object may construct successfully and
    fail at call time instead.

## Related samples

*   [Structured tool agent](../../../../../contributing/samples/integrations/langchain_structured_tool_agent/agent.py)
    pairs an explicit `StructuredTool` carrying an `args_schema` with a
    `@tool`-decorated function.
*   [YouTube search agent](../../../../../contributing/samples/integrations/langchain_youtube_search_agent/agent.py)
    wraps a `langchain_community` tool as-is.

## Related guides

*   [CrewaiTool](../../crewai/crewai_tool/index.md) is the same adapter for
    CrewAI, with the differences tabled above.
