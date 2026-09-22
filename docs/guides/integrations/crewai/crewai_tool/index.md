# CrewaiTool

`CrewaiTool` wraps a CrewAI tool so an ADK agent can call it. It is a
`FunctionTool` subclass that calls the CrewAI tool's `run` method, builds the
Gemini function declaration from the tool's `args_schema`, and handles the
`**kwargs` signature most CrewAI tools have.

There is no account or service to set up, but there is one hard prerequisite:
CrewAI support requires Python 3.11. On any other version the `extensions`
extra installs without CrewAI, and the import then fails.

## Introduction

`CrewaiTool` is the adapter that lets an existing CrewAI tool keep working in
ADK. Put a wrapped tool in an agent's `tools` list and it behaves like any
other ADK tool: it appears in the model's function declarations, it is invoked
with the arguments the model chose, and its return value goes back as a
function response.

It is a migration path rather than a permanent home. A tool you own is usually
better rewritten as a plain Python function and passed straight to `tools=`,
which is what `FunctionTool` does with no adapter in the way.

[`LangchainTool`](../../langchain/langchain_tool/index.md) is the same idea for
LangChain, and the two adapters are close enough that only four things separate
them:

| | `CrewaiTool` | `LangchainTool` |
| :--- | :--- | :--- |
| `name` argument | Required | Optional |
| Declaration source | Always `tool.args_schema` | `args_schema` when present, otherwise signature introspection |
| Non-framework objects | Rejected: must be a `crewai.tools.BaseTool` | Accepted if it has `run` or `_run` |
| `return_direct` | Not translated | Sets `skip_summarization` |

## Get started

Three steps get a wrapped CrewAI tool into an agent.

1.  Check that you are on Python 3.11, the only version the `extensions` extra
    brings CrewAI in on.

2.  Install the extra:

    ```bash
    pip install "google-adk[extensions]"
    ```

3.  Wrap the CrewAI tool and put it in an agent's `tools` list. **`name` is
    required.** The constructor has no default for it, so if you omit it you
    get a `TypeError` rather than a fallback to the CrewAI name.

    ```python
    from google.adk.agents import Agent
    from google.adk.integrations.crewai import CrewaiTool
    from crewai_tools import SerperDevTool

    search_tool = CrewaiTool(
        SerperDevTool(),
        name="web_search",
        description="Searches the public web and returns a text summary.",
    )

    root_agent = Agent(
        name="researcher",
        description="Answers questions using web search.",
        instruction="Search before answering. Cite what the search returned.",
        tools=[search_tool],
    )
    ```

The tool is the first positional argument; `name` and `description` are
keyword-only.

If you pass `name=""` explicitly, the CrewAI tool's own name is used instead,
lowercased and with spaces turned into underscores, because CrewAI permits
spaces in a tool name and Gemini does not. Relying on that is a bad idea, since
CrewAI names are usually written for a human and land in the prompt unchanged.
Write your own.

## How it works

Three things decide how a wrapped tool looks to the model: where its name and
description come from, how its function declaration is built, and which
arguments reach the CrewAI tool underneath.

### Name and description

Each resolves in two steps: the explicit argument if it is non-empty, then the
CrewAI tool's own attribute.

### The declaration, and the `required` list that is not there

The Gemini declaration is built from the CrewAI tool's `args_schema`. There is
no signature-introspection fallback, which makes `args_schema` effectively
mandatory: if your CrewAI tool has none, wrapping it raises instead of
degrading to an untyped declaration.

The declaration that comes out carries the schema's `properties` and **not its
`required` list**. Every parameter looks optional to the model, even the ones
your Pydantic schema marked required. The check still happens, at call time
instead, and a missing mandatory argument returns an error dictionary to the
model rather than raising:

```text
{'error': "Invoking `web_search()` failed as the following mandatory input
parameters are not present: query\nYou could retry calling this tool, but it is
IMPORTANT for you to provide all the mandatory parameters."}
```

The practical effect is one extra round trip when the model guesses wrong.
Descriptions on your Pydantic fields are the thing that prevents it, so write
them.

### `**kwargs`

Most CrewAI tools are written as `def _run(self, query: str, **kwargs)`, taking
arbitrary filters through the catch-all. ADK's normal argument filtering would
drop everything the signature does not name, so `CrewaiTool` forwards those
arguments instead.

The rule is decided by whether the callable has a `**kwargs` parameter:

*   **With `**kwargs`:** every argument the model supplied is forwarded, minus
    `self` and minus the tool-context parameter. So `category`, `date_range` and
    `limit` all arrive in `kwargs` even though nothing declares them.
*   **Without `**kwargs`:** the usual filtering applies, and anything the
    signature does not name is dropped.

In both cases, if the callable declares a `tool_context` parameter, ADK injects
the real `ToolContext` over anything the model may have put there.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tool` | `crewai.tools.BaseTool` | required | The CrewAI tool. Positional. |
| `name` | `str` | required | The name the model sees. Keyword-only. `''` falls back to the CrewAI name. |
| `description` | `str` | `''` | The description the model sees. Keyword-only. Empty falls back to the CrewAI description. |

`description` is where the accuracy is. A `crewai-tools` community tool's
description is usually written for a Python reader browsing the catalog, and
it goes straight into the prompt. Rewriting it is the cheapest improvement
available on a wrapped tool.

## Advanced applications

Two questions come up once the first wrapped tool works: how to declare one
outside Python, and when to rewrite a tool instead of wrapping it.

### Declare the tool in YAML

`CrewaiTool` supports ADK's config-driven agent loading through
`CrewaiToolConfig`, so a wrapped tool can be named in an agent YAML file rather
than constructed in Python:

```yaml
tools:
  - name: google.adk.integrations.crewai.CrewaiTool
    args:
      tool: my_package.tools.search_tool
      name: web_search
      description: Searches the public web and returns a text summary.
```

`tool` is the fully qualified path to a CrewAI tool **instance**, not a class; it
is resolved by import at load time. `name` and `description` both default to the
empty string here, so the YAML form is the one place `name` is genuinely
optional: leave it out and the CrewAI tool's own name is used.

### Decide what to migrate rather than wrap

Wrapping is the right answer for a tool you did not write and do not want to own.
For a tool that is your own Python function underneath, the adapter is overhead:
pass the function to `tools=` and ADK builds a `FunctionTool` from its signature
and docstring. You also get things this adapter cannot give you cleanly, notably
a real `tool_context: ToolContext` parameter for reading session state, saving
artifacts, or requesting confirmation.

## Limitations

*   **CrewAI support is Python 3.11 only.** The `extensions` extra brings
    CrewAI in on Python 3.11 and on no other version. Elsewhere the extra
    still installs successfully, and the failure only shows up at import.
*   **The import fails as soon as CrewAI is absent.**
    `from google.adk.integrations.crewai import CrewaiTool` raises
    `ImportError: Crewai Tools require pip install 'google-adk[extensions]'.`
    at import time rather than when you construct a tool. There is no lazy
    path.
*   **`args_schema` is effectively required.** No signature fallback exists, so
    a CrewAI tool without one cannot produce a declaration.
*   **The model is never told which arguments are mandatory.** The declaration
    carries no `required` list, so the check happens at call time instead.
*   **Nothing but the callable and the schema carries across.** CrewAI's
    `cache_function`, its result-as-answer behavior, its own error handling and
    its telemetry are not translated. `LangchainTool` at least maps
    `return_direct`; there is no equivalent here.
*   **`google.adk.tools.crewai_tool` is a deprecated shim.** Importing from it
    emits `DeprecationWarning: google.adk.tools.crewai_tool is moved to
    google.adk.integrations.crewai`. Use the new path.

## Related samples

*   [CrewAI tool with \*\*kwargs](../../../../../contributing/samples/integrations/crewai_tool_kwargs/agent.py)
    is a `BaseTool` whose `_run` takes `**kwargs`, so you can watch the
    arbitrary filters arrive intact.

## Related guides

*   [LangchainTool](../../langchain/langchain_tool/index.md) is the same
    adapter for LangChain, with the differences tabled above.
