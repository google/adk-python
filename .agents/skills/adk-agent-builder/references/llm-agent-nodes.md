# LLM Agent Nodes Reference

Embed LLM-powered agents as nodes in workflow graphs.

## 📋 Agent Verification Checklist (LLM Nodes)
Use this checklist to verify your LLM agent configuration:
- [ ] **Output Type**: If no `output_schema` is set, downstream now receives `str` (auto-extracted from `types.Content`). You can safely type-hint `node_input: str`.
- [ ] **State Serialization**: If this agent feeds into a `JoinNode`, did you set `output_schema` to avoid non-serializable `types.Content` errors?
- [ ] **Instructions**: Are `{var}` templates used in instructions resolving ONLY from `ctx.state`? (Not `node_input`)
- [ ] **Config**: Are instructions, tools, and response schema set on the `LlmAgent` directly, and NOT in `generate_content_config`?

## 💡 Quick Reference
- **Chat Mode**: Default. Multi-turn, keeps session history.
- **Single-Turn Mode**: Isolated. Set `mode="single_turn"` or rely on auto-wrapping defaults.
- **Task Mode**: Multi-turn within a task. Set `mode="task"`.
- **Stateless**: Set `include_contents="none"` to ignore session history.

## Imports

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow._llm_agent_wrapper import _LlmAgentWrapper  # private
from google.adk.workflow import Workflow
```

## Choosing the Right LLM Agent

**Use `google.adk.agents.llm_agent.LlmAgent`** in workflow edges. It is auto-wrapped as `LlmAgentWrapper`, which emits `Event(output=...)` for downstream data passing. This is required for any LLM agent that needs to pass output to downstream function nodes via `node_input`.

```python
from google.adk.agents.llm_agent import LlmAgent

writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Write a short story.",
    output_schema=Story,
)

# writer is auto-wrapped as _LlmAgentWrapper — downstream gets Event(output=...)
agent = Workflow(
    name="pipeline",
    edges=[('START', writer), (writer, process_story)],
)
```

## Basic LLM Node

```python
from google.adk.agents.llm_agent import LlmAgent

writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Write a short story based on the user's prompt.",
)

reviewer = LlmAgent(
    name="reviewer",
    model="gemini-2.5-flash",
    instruction="Review the following story and provide feedback.",
)

agent = Workflow(
    name="story_pipeline",
    edges=[
        ('START', writer),      # Auto-wrapped as LlmAgentWrapper
        (writer, reviewer),
    ],
)
```

## LLM Agent Output Types (Critical)

**LlmAgentWrapper auto-extracts text and outputs `str` when no `output_schema` is set.** Previously, it outputted `types.Content` causing type errors. Now, if you type-hint `node_input: str`, it will work correctly for standard text output.

**Solutions (pick one):**

1. **Use `Any` and extract text** (recommended for function nodes after LLM agents):

```python
from typing import Any
from google.genai import types

def process_llm_output(node_input: Any) -> str:
  if isinstance(node_input, types.Content):
    return ''.join(p.text for p in (node_input.parts or []) if p.text)
  return str(node_input) if node_input is not None else ''
```

2. **Use `output_schema`** on the LLM agent to get a parsed `dict` instead:

```python
from pydantic import BaseModel

class CodeOutput(BaseModel):
  code: str
  language: str

writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Write code. Return JSON with 'code' and 'language' fields.",
    output_schema=CodeOutput,
)

# Downstream node receives dict: {"code": "...", "language": "python"}
def process_code(node_input: dict) -> str:
  return node_input["code"]
```

**Summary of LLM agent node output types:**

| LLM Agent Config | `node_input` Type for Next Node |
|-----------------|-------------------------------|
| No `output_schema` | `types.Content` |
| With `output_schema` | `dict` (parsed from Pydantic model) |

**State serialization warning:** When LLM agents feed into a `JoinNode`, the JoinNode stores intermediate results in session state. Without `output_schema`, this stores `types.Content` objects which are **not JSON-serializable** and will cause `TypeError` with SQLite/database session services. Always use `output_schema` on LLM agents that feed into a JoinNode.

## Auto-Wrapping Behavior

When you place an `LlmAgent` in workflow edges, it is auto-wrapped as `_LlmAgentWrapper`. The wrapper:
- Defaults to `single_turn` mode (agent sees only current input, not session history)
- Sets `rerun_on_resume=True` (reruns after HITL interrupts)
- Creates a content branch for isolation between parallel LLM agents

The mode is set on the `LlmAgent` itself, not the wrapper:

```python
from google.adk.agents.llm_agent import LlmAgent

# single_turn (default when auto-wrapped): isolated, no session history
classifier = LlmAgent(
    name="classifier",
    model="gemini-2.5-flash",
    instruction="Classify the input as positive, negative, or neutral.",
    output_schema=ClassificationResult,
)

# task mode: supports HITL, multi-turn within the task
task_agent = LlmAgent(
    name="task_agent",
    model="gemini-2.5-flash",
    mode="task",
    instruction="Process the request.",
)
```

## LlmAgent Configuration

### Instructions

Dynamic instructions with placeholders resolved from session state. **`{var}` templates only resolve from `ctx.state` — `node_input` is NOT available in templates.** To use predecessor data in instructions, store it in state first (via `Event(state={...})` or `output_key`):

```python
agent = LlmAgent(
    name="personalized",
    model="gemini-2.5-flash",
    instruction="""You are helping {user_name}.
Their preferences are: {preferences}.
Respond in {language}.""",
)
# {user_name}, {preferences}, {language} resolved from session state
# Missing variables raise KeyError at runtime — use {var?} for optional:
# instruction="Current mood: {mood?}"  # empty string if 'mood' not in state
```

**Template variable behavior:**

| Syntax | Missing Key Behavior |
|--------|---------------------|
| `{var}` | Raises `KeyError` at LLM call time |
| `{var?}` | Substitutes empty string, logs debug message |
| `{not.an" identifier}` | Left as-is (not substituted) |

Instruction provider function for fully dynamic instructions:

```python
from google.adk.agents.readonly_context import ReadonlyContext

def build_instruction(ctx: ReadonlyContext) -> str:
  agents = ctx.state.get("active_agents", [])
  return f"Coordinate these agents: {', '.join(agents)}"

agent = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction=build_instruction,
)
```

### Output Schema

Structure LLM output with Pydantic models:

```python
from pydantic import BaseModel

class ReviewResult(BaseModel):
  score: int
  feedback: str
  approved: bool

reviewer = LlmAgent(
    name="reviewer",
    model="gemini-2.5-flash",
    instruction="Review the code and provide structured feedback.",
    output_schema=ReviewResult,
)
```

When used as a workflow node, the output becomes a `dict` (via `model_dump()`) as `node_input` for the next node.

### Output Key

Store agent output in session state:

```python
agent = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Write a draft.",
    output_key="draft",  # Stores output in state['draft']
)
```

### include_contents

Control conversation history:

```python
agent = LlmAgent(
    name="stateless",
    model="gemini-2.5-flash",
    instruction="Process this input independently.",
    include_contents="none",  # Don't include session history
)
```

## Tools

Add tools to LLM agents:

```python
def search_database(query: str) -> str:
  """Search the database for relevant records."""
  return f"Results for: {query}"

def send_email(to: str, subject: str, body: str) -> str:
  """Send an email to the specified address."""
  return "Email sent"

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="Help the user with their request.",
    tools=[search_database, send_email],
)
```

Tools can be:
- Python functions (auto-wrapped as `FunctionTool`)
- `BaseTool` instances
- `BaseToolset` instances (e.g., MCP toolsets)

## Callbacks

### Before Model Callback

Intercept or modify LLM requests. Return an `LlmResponse` to skip the LLM call; return `None` to proceed:

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

def guard_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
  for content in llm_request.contents:
    if content.parts:
      for part in content.parts:
        if part.text and "unsafe" in part.text:
          return LlmResponse(
              content=types.ModelContent("I cannot process that.")
          )
  return None  # Proceed with normal LLM call

agent = LlmAgent(
    name="guarded",
    model="gemini-2.5-flash",
    before_model_callback=guard_callback,
)
```

### After Model Callback

Transform LLM responses. Return an `LlmResponse` to replace; return `None` to keep original:

```python
def log_response(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
  print(f"LLM responded: {llm_response.content}")
  return None  # Use original response

agent = LlmAgent(
    name="logged",
    model="gemini-2.5-flash",
    after_model_callback=log_response,
)
```

### Before/After Tool Callbacks

Intercept tool calls. Return a `dict` to use as tool response (skipping actual execution); return `None` to proceed:

```python
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

def audit_tool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict | None:
  print(f"Calling tool {tool.name} with args: {args}")
  return None  # Proceed with tool call

def validate_tool_result(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict,
) -> dict | None:
  if "error" in tool_response:
    return {"result": "Tool execution failed, please try again."}
  return None  # Use original result

agent = LlmAgent(
    name="audited",
    model="gemini-2.5-flash",
    tools=[my_tool],
    before_tool_callback=audit_tool,
    after_tool_callback=validate_tool_result,
)
```

### Multiple Callbacks

Pass a list of callbacks. They execute in order until one returns non-None:

```python
agent = LlmAgent(
    name="multi_callback",
    model="gemini-2.5-flash",
    before_model_callback=[safety_check, rate_limiter, logger],
)
```

### Error Callbacks

Handle LLM or tool errors gracefully:

```python
def handle_model_error(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse | None:
  return LlmResponse(
      content=types.ModelContent("Service temporarily unavailable.")
  )

def handle_tool_error(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    error: Exception,
) -> dict | None:
  return {"error": str(error), "fallback": True}

agent = LlmAgent(
    name="resilient",
    model="gemini-2.5-flash",
    on_model_error_callback=handle_model_error,
    on_tool_error_callback=handle_tool_error,
)
```

## All Callback Types

| Callback | Signature | Return to Override |
|----------|-----------|-------------------|
| `before_model_callback` | `(CallbackContext, LlmRequest) -> LlmResponse?` | Return `LlmResponse` to skip LLM |
| `after_model_callback` | `(CallbackContext, LlmResponse) -> LlmResponse?` | Return `LlmResponse` to replace |
| `on_model_error_callback` | `(CallbackContext, LlmRequest, Exception) -> LlmResponse?` | Return `LlmResponse` to suppress error |
| `before_tool_callback` | `(BaseTool, dict, ToolContext) -> dict?` | Return `dict` to skip tool |
| `after_tool_callback` | `(BaseTool, dict, ToolContext, dict) -> dict?` | Return `dict` to replace result |
| `on_tool_error_callback` | `(BaseTool, dict, ToolContext, Exception) -> dict?` | Return `dict` to suppress error |

All callbacks can be sync or async. All accept a single callback or a list.

## Generate Content Config

Fine-tune LLM generation:

```python
from google.genai import types

agent = LlmAgent(
    name="creative",
    model="gemini-2.5-flash",
    instruction="Write creative stories.",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.9,
        top_p=0.95,
        max_output_tokens=2048,
    ),
)
```

## Agent Transfer

Agents can transfer control to sub-agents:

```python
specialist = LlmAgent(
    name="specialist",
    model="gemini-2.5-flash",
    instruction="Handle specialized requests.",
)

coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="Route requests to the specialist when needed.",
    sub_agents=[specialist],
)
```

Control transfer behavior:

```python
agent = LlmAgent(
    name="isolated",
    model="gemini-2.5-flash",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
```
