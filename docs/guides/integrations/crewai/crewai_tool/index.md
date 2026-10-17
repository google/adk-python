# CrewaiTool

CrewaiTool wraps tools from the CrewAI ecosystem for use within the ADK framework.
The wrapper ensures that tool signatures and parameter handling match the
requirements of large language models.

## Introduction

Many developers have existing toolsets built for CrewAI. The `CrewaiTool` class
allows these tools to be used directly in ADK agents without rewriting the
logic. The class handles the translation between the CrewAI `BaseTool` structure
and the ADK tool interface, including automatic schema generation from Pydantic
models.

The wrapper specifically addresses differences in how CrewAI and ADK manage tool
execution. It automatically cleans tool names to meet model requirements and
manages the `**kwargs` pattern commonly found in CrewAI tools, ensuring that
internal framework parameters do not interfere with tool logic.

## Get started

The following example demonstrates how to wrap a custom CrewAI tool and attach
it to an ADK agent.

```python
from typing import Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from google.adk import Agent
from google.adk.integrations.crewai import CrewaiTool

class SearchInput(BaseModel):
    query: str = Field(..., description="The search query string")
    limit: Optional[int] = Field(None, description="Result limit")

class CustomSearchTool(BaseTool):
    name: str = "custom_search"
    description: str = "Search for information with optional limits."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, **kwargs) -> str:
        limit = kwargs.get("limit", 5)
        return f"Searching for {query} with limit {limit}"

# Instantiate the CrewAI tool
crewai_search_tool = CustomSearchTool()

# Wrap it for ADK
adk_search_tool = CrewaiTool(
    crewai_search_tool,
    name="search_with_filters",
    description="Search for information with an optional result limit"
)

# Use the wrapped tool in an agent
search_agent = Agent(
    name="search_agent",
    description="An agent that can search using CrewAI tools",
    tools=[adk_search_tool],
)
```

## How it works

The `CrewaiTool` class inherits from `FunctionTool` and wraps the `run` method
of a CrewAI `BaseTool`. During initialization, the wrapper inspects the
provided tool and prepares it for the ADK environment.

The wrapper performs automatic name normalization. Because many models do not
support spaces in tool names, the class replaces spaces with underscores and
converts the name to lowercase. If the original tool name is "Serper Dev Tool",
the wrapper defaults the name to `serper_dev_tool`.

When the agent invokes the tool, the wrapper handles parameter filtering.
CrewAI tools frequently use a `**kwargs` pattern to accept flexible inputs.
The `CrewaiTool.run_async` method identifies these functions and ensures that
all relevant arguments from the model are passed through, while stripping
away framework-internal arguments like `self` that would cause execution
errors.

The wrapper also uses the CrewAI tool's `args_schema` to build the function
declaration. This ensures that the model receives the correct JSON schema
defined by the tool's Pydantic model, including field descriptions and
optional constraints.

## Configuration options

The following options are used when defining a `CrewaiTool` through a
configuration file or the `from_config` method.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tool` | `str` | | The fully qualified path of the CrewAI tool instance. |
| `name` | `str` | `''` | The name to assign to the tool. |
| `description` | `str` | `''` | The description of the tool for the model. |

The `tool` option requires a string representing the fully qualified name of
the tool instance, which the framework resolves at runtime. If the `name` or
`description` options are left as empty strings, the wrapper attempts to
extract these values from the underlying CrewAI tool instance.

## Advanced applications

The wrapper supports context injection for tools that need access to the
current execution state. If a CrewAI tool defines a parameter named
`tool_context` or uses the `Context` type annotation, the wrapper
automatically injects the ADK `ToolContext` during invocation.

```python
from google.adk.tools.tool_context import ToolContext

def tool_with_context(tool_context: ToolContext, query: str, **kwargs):
    # The wrapper identifies the tool_context parameter and provides it
    session_id = tool_context.invocation_context.session.id
    return f"Searching for {query} in session {session_id}"
```

This injection happens regardless of whether the tool uses explicit parameters
or `**kwargs`. The wrapper removes any existing `tool_context` keys from the
raw argument dictionary to prevent duplicates and then re-inserts the
authorized context object.

## Limitations

The `CrewaiTool` requires the optional extensions package. You must install
the framework using `pip install 'google-adk[extensions]'` to use this
integration.

The wrapper enforces name compatibility by default. If a CrewAI tool has a name
that is already model-compliant, the wrapper uses it as-is, but any spaces are
always replaced with underscores to prevent model invocation failures.

## Related samples

- [crewai_tool_kwargs](../../../../../contributing/samples/integrations/crewai_tool_kwargs/agent.py) - Demonstrates handling arbitrary parameters through **kwargs.
- [crewai_tool_kwargs](../../../../../contributing/samples/integrations/crewai_tool_kwargs/main.py) - A runnable script testing the CrewAI tool integration.
