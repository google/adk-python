# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool code generator for CodingAgent.

This module provides functions to generate Python code stubs and runtime
headers that allow generated code to call ADK tools via HTTP IPC.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..tools.base_tool import BaseTool

logger = logging.getLogger("google_adk." + __name__)


# Runtime header template that provides HTTP-based tool calling
RUNTIME_HEADER_TEMPLATE = '''
# ============================================================================
# ADK CodingAgent Runtime Header - DO NOT MODIFY
# ============================================================================
import json as __adk_json
import urllib.request as __adk_urllib_request
import urllib.error as __adk_urllib_error

__ADK_TOOL_SERVER_URL = "{tool_server_url}"
__ADK_TOOL_TRACES = []

def _call_adk_tool(__tool_name: str, **kwargs) -> dict:
    """Call an ADK tool via HTTP IPC.
    
    Args:
        __tool_name: Name of the tool to call.
        **kwargs: Arguments to pass to the tool.
        
    Returns:
        The tool result as a dictionary.
    """
    global __ADK_TOOL_TRACES
    
    request_data = __adk_json.dumps({{
        "tool_name": __tool_name,
        "args": kwargs,
    }}).encode("utf-8")
    
    req = __adk_urllib_request.Request(
        __ADK_TOOL_SERVER_URL + "/tool_call",
        data=request_data,
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    
    try:
        with __adk_urllib_request.urlopen(req, timeout=300) as response:
            result = __adk_json.loads(response.read().decode("utf-8"))
            # Record the trace
            __ADK_TOOL_TRACES.append({{
                "tool_name": __tool_name,
                "args": kwargs,
                "result": result,
                "success": True,
            }})
            return result
    except __adk_urllib_error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else str(e)
        __ADK_TOOL_TRACES.append({{
            "tool_name": __tool_name,
            "args": kwargs,
            "error": error_body,
            "success": False,
        }})
        raise RuntimeError(f"Tool call failed: {{error_body}}") from e
    except __adk_urllib_error.URLError as e:
        __ADK_TOOL_TRACES.append({{
            "tool_name": __tool_name,
            "args": kwargs,
            "error": str(e),
            "success": False,
        }})
        raise RuntimeError(f"Tool server connection failed: {{e}}") from e

def __get_tool_traces() -> list:
    """Get all tool call traces."""
    return __ADK_TOOL_TRACES

def __clear_tool_traces():
    """Clear all tool call traces."""
    global __ADK_TOOL_TRACES
    __ADK_TOOL_TRACES = []

# Final answer function for terminating execution
__FINAL_ANSWER_VALUE = None

def final_answer(result):
    """Mark the final answer and terminate the code execution.
    
    Args:
        result: The final result to return to the user.
    """
    global __FINAL_ANSWER_VALUE
    __FINAL_ANSWER_VALUE = result
    print(f"__FINAL_ANSWER__:{{__adk_json.dumps(result) if not isinstance(result, str) else result}}")

# ============================================================================
# End of Runtime Header
# ============================================================================

'''


def generate_runtime_header(
    tool_server_url: str,
) -> str:
  """Generate the runtime header with HTTP client and helper functions.

  Args:
    tool_server_url: URL of the tool execution server.

  Returns:
    Python code string containing the runtime header.
  """
  return RUNTIME_HEADER_TEMPLATE.format(tool_server_url=tool_server_url)


def _get_schema_type(schema: Any) -> str:
  """Get the type from a schema (dict or Pydantic Schema object).

  Args:
    schema: JSON schema dict or google.genai.types.Schema object.

  Returns:
    The type as a lowercase string.
  """
  if hasattr(schema, "type"):
    # Pydantic Schema object from google.genai.types
    schema_type = schema.type
    if schema_type is None:
      return "any"
    # Handle enum (Type.STRING -> "string")
    if hasattr(schema_type, "value"):
      return schema_type.value.lower()
    return str(schema_type).lower()
  elif isinstance(schema, dict):
    return schema.get("type", "any")
  return "any"


def _get_schema_attr(schema: Any, attr: str, default: Any = None) -> Any:
  """Get an attribute from a schema (dict or Pydantic Schema object).

  Args:
    schema: JSON schema dict or google.genai.types.Schema object.
    attr: The attribute name to get.
    default: Default value if attribute not found.

  Returns:
    The attribute value or default.
  """
  if hasattr(schema, attr):
    return getattr(schema, attr, default)
  elif isinstance(schema, dict):
    return schema.get(attr, default)
  return default


def _get_python_type_hint(schema: Any) -> str:
  """Convert JSON schema type to Python type hint.

  Args:
    schema: JSON schema dict or google.genai.types.Schema object.

  Returns:
    Python type hint string.
  """
  schema_type = _get_schema_type(schema)

  type_mapping = {
      "string": "str",
      "integer": "int",
      "number": "float",
      "boolean": "bool",
      "array": "list",
      "object": "dict",
  }

  if schema_type == "array":
    items = _get_schema_attr(schema, "items", {})
    if items:
      item_type = _get_python_type_hint(items)
      return f"list[{item_type}]"
    return "list"
  elif schema_type == "object":
    return "dict"

  return type_mapping.get(schema_type, "Any")


def _generate_tool_stub(tool: BaseTool) -> str:
  """Generate a Python function stub for a single tool.

  Args:
    tool: The BaseTool to generate a stub for.

  Returns:
    Python code string for the tool stub function.
  """
  decl = tool._get_declaration()
  if not decl:
    logger.warning(
        "Tool %s has no declaration, skipping stub generation", tool.name
    )
    return ""

  # Build parameter list with type hints
  params = []
  param_docs = []

  if decl.parameters and decl.parameters.properties:
    required = set(decl.parameters.required or [])

    for param_name, param_schema in decl.parameters.properties.items():
      type_hint = _get_python_type_hint(param_schema)
      description = _get_schema_attr(param_schema, "description", "")

      if param_name in required:
        params.append(f"{param_name}: {type_hint}")
      else:
        params.append(f"{param_name}: {type_hint} = None")

      param_docs.append(f"        {param_name}: {description}")

  param_str = ", ".join(params)
  param_doc_str = "\n".join(param_docs) if param_docs else "        None"

  # Build the function stub
  stub = f'''
def {tool.name}({param_str}) -> dict:
    """{tool.description}
    
    Args:
{param_doc_str}
    
    Returns:
        Tool execution result as a dictionary.
    """
    kwargs = {{k: v for k, v in locals().items() if v is not None}}
    response = _call_adk_tool("{tool.name}", **kwargs)
    # Extract the result from the tool server response
    if isinstance(response, dict) and "result" in response:
        return response["result"]
    return response

'''
  return stub


def generate_tool_stubs(tools: List[BaseTool]) -> str:
  """Generate Python function stubs for all tools.

  Args:
    tools: List of tools to generate stubs for.

  Returns:
    Python code string containing all tool stubs.
  """
  stubs = [
      (
          "# ============================================================================"
      ),
      "# Tool Function Stubs",
      (
          "# ============================================================================"
      ),
      "",
  ]

  for tool in tools:
    stub = _generate_tool_stub(tool)
    if stub:
      stubs.append(stub)

  return "\n".join(stubs)


def generate_final_answer_stub() -> str:
  """Generate the final_answer function documentation.

  This is included in the runtime header, but we generate additional
  documentation here for the system prompt.

  Returns:
    Documentation string about the final_answer function.
  """
  return """
The `final_answer(result)` function is available to mark your final result.
Call this function when you have completed the task and have a result to return.
Example: `final_answer("The calculation result is 42")`
"""


# Few-shot examples for the system prompt
SYSTEM_PROMPT_EXAMPLES = """
## Examples

### Example 1 - Using a tool to search for information:
```tool_code
# Search for relevant information
result = web_search(query="Python async best practices")
# Display the findings
for snippet in result.get("snippets", [])[:3]:
    print(snippet)
```

### Example 2 - Processing data and providing a final answer:
```tool_code
# Read and process data
data = read_file(path="sales_data.csv")
rows = data.get("rows", [])

# Calculate the total
total = sum(float(row.get("amount", 0)) for row in rows)

# Provide the final answer
final_answer(f"The total sales amount is ${total:.2f}")
```

### Example 3 - Multi-step reasoning with tool calls:
```tool_code
# Step 1: Get the current weather
weather = get_weather(city="San Francisco")
temp = weather.get("temperature", "unknown")
print(f"Current temperature: {temp}")
```

Then, after seeing the output:
```tool_code
# Step 2: Based on temperature, provide recommendation
if temp > 70:
    recommendation = "It's warm! Consider light clothing."
else:
    recommendation = "It might be cool. Bring a jacket."
    
final_answer(recommendation)
```
"""


def generate_system_prompt(
    tools: List[BaseTool],
    custom_instruction: str = "",
) -> str:
  """Generate the system prompt for the CodingAgent.

  Args:
    tools: List of available tools.
    custom_instruction: Additional custom instructions.

  Returns:
    Complete system prompt string.
  """
  # Build tool documentation
  tool_docs = []
  for tool in tools:
    decl = tool._get_declaration()
    if decl:
      params_doc = ""
      if decl.parameters and decl.parameters.properties:
        param_lines = []
        required = set(decl.parameters.required or [])
        for name, schema in decl.parameters.properties.items():
          type_hint = _get_python_type_hint(schema)
          req_marker = " (required)" if name in required else " (optional)"
          desc = _get_schema_attr(schema, "description", "")
          param_lines.append(f"    - {name}: {type_hint}{req_marker} - {desc}")
        params_doc = "\n".join(param_lines)

      tool_docs.append(f"""
### {tool.name}
{tool.description}

Parameters:
{params_doc if params_doc else "    None"}
""")

  tools_section = "\n".join(tool_docs) if tool_docs else "No tools available."

  system_prompt = f"""You are a coding agent that solves tasks by writing and executing Python code.

## How to Respond

1. **Write Python code** in code blocks marked with ```tool_code
2. **Use available tools** by calling them as Python functions
3. **Print intermediate results** to see outputs and make decisions
4. **Call final_answer()** when you have the final result

## Available Tools

{tools_section}

## Special Functions

- `final_answer(result)`: Call this to provide your final answer and complete the task.
- `print(...)`: Use print statements to see intermediate results.

{SYSTEM_PROMPT_EXAMPLES}

## Important Guidelines

1. **One step at a time**: Write code for one logical step, wait for output, then continue.
2. **Always print results**: Use print() to see what tools return.
3. **Handle errors gracefully**: If a tool fails, try an alternative approach.
4. **Call final_answer()**: When done, call final_answer() with your result.
5. **Install packages if needed**: If you need external libraries (pandas, matplotlib, numpy, etc.), install them first using subprocess:
   ```python
   import subprocess
   subprocess.run(["pip", "install", "-q", "pandas", "matplotlib", "seaborn"], check=True)
   ```
   Then import and use them normally.

{custom_instruction}
"""

  return system_prompt.strip()


def generate_full_code_with_stubs(
    user_code: str,
    tools: List[BaseTool],
    tool_server_url: str,
) -> str:
  """Generate complete executable code with runtime header and tool stubs.

  Args:
    user_code: The user-generated code to execute.
    tools: List of available tools.
    tool_server_url: URL of the tool execution server.

  Returns:
    Complete Python code ready for execution.
  """
  runtime_header = generate_runtime_header(tool_server_url)
  tool_stubs = generate_tool_stubs(tools)

  full_code = f"""{runtime_header}
{tool_stubs}
# ============================================================================
# User Code
# ============================================================================

{user_code}

# ============================================================================
# Output tool traces for extraction
# ============================================================================
import json as __output_json
print("__TOOL_TRACE__:" + __output_json.dumps(__get_tool_traces()))
"""

  return full_code
