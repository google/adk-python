"""
Code execution utilities for ADK-RLM.

This module provides functions for parsing code blocks from LLM responses
and processing execution results.
"""

import re
from typing import TYPE_CHECKING

from adk_rlm.types import REPLResult
from adk_rlm.types import RLMIteration

if TYPE_CHECKING:
  from adk_rlm.repl.local_repl import LocalREPL


def _extract_balanced_parens(text: str, start_pos: int) -> str | None:
  """
  Extract content inside balanced parentheses starting at start_pos.

  Args:
      text: The full text.
      start_pos: Position of the opening parenthesis.

  Returns:
      The content inside the balanced parentheses, or None if unbalanced.
  """
  if start_pos >= len(text) or text[start_pos] != "(":
    return None

  depth = 0
  start = start_pos + 1  # Skip the opening paren

  for i in range(start_pos, len(text)):
    if text[i] == "(":
      depth += 1
    elif text[i] == ")":
      depth -= 1
      if depth == 0:
        return text[start:i]

  # Unbalanced - return everything after the opening paren
  return text[start:]


def find_code_blocks(text: str | None) -> list[str]:
  """
  Find REPL code blocks in text wrapped in triple backticks.

  Args:
      text: The text to search for code blocks.

  Returns:
      List of code block contents (without the ```repl markers).
  """
  if text is None:
    return []

  pattern = r"```repl\s*\n(.*?)\n```"
  results = []

  for match in re.finditer(pattern, text, re.DOTALL):
    code_content = match.group(1).strip()
    results.append(code_content)

  return results


def find_final_answer(
    text: str | None, repl: "LocalREPL | None" = None
) -> str | None:
  """
  Find FINAL(...) or FINAL_VAR(...) statement in response.

  Args:
      text: The response text to parse.
      repl: Optional REPL environment for FINAL_VAR resolution.

  Returns:
      The final answer string, or None if no final answer pattern is found.
  """
  if text is None:
    return None

  # Check for FINAL_VAR pattern first - must be at start of line
  # Use regex to find the start, then balanced parens for content
  final_var_match = re.search(r"^\s*FINAL_VAR\(", text, re.MULTILINE)
  if final_var_match:
    paren_start = final_var_match.end() - 1  # Position of '('
    variable_name = _extract_balanced_parens(text, paren_start)
    if variable_name is not None:
      variable_name = variable_name.strip().strip('"').strip("'")
      if repl is not None:
        result = repl.execute_code(f"print(FINAL_VAR({variable_name!r}))")
        final_answer = result.stdout.strip()
        if final_answer == "":
          final_answer = result.stderr.strip() or ""
        # Check if FINAL_VAR returned an error (variable not found)
        if final_answer.startswith(
            "Error: Variable '"
        ) and final_answer.endswith("' not found"):
          return None
        return final_answer
      return None

  # Check for FINAL pattern - must be at start of line
  # Use regex to find the start, then balanced parens for content
  final_match = re.search(r"^\s*FINAL\(", text, re.MULTILINE)
  if final_match:
    paren_start = final_match.end() - 1  # Position of '('
    content = _extract_balanced_parens(text, paren_start)
    if content is not None:
      return content.strip()

  return None


def format_execution_result(result: REPLResult) -> str:
  """
  Format the execution result as a string for display.

  Args:
      result: The REPLResult object to format.

  Returns:
      A formatted string representation of the result.
  """
  result_parts = []

  if result.stdout:
    result_parts.append(f"\n{result.stdout}")

  if result.stderr:
    result_parts.append(f"\n{result.stderr}")

  # Show some key variables (excluding internal ones)
  important_vars = {}
  for key, value in result.locals.items():
    if not key.startswith("_") and key not in [
        "__builtins__",
        "__name__",
        "__doc__",
    ]:
      # Only show simple types or short representations
      if isinstance(value, (str, int, float, bool, list, dict, tuple)):
        important_vars[key] = ""

  if important_vars:
    result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

  return "\n\n".join(result_parts) if result_parts else "No output"


def format_iteration(
    iteration: RLMIteration, max_character_length: int = 20000
) -> list[dict[str, str]]:
  """
  Format an RLM iteration to append to the message history.

  Args:
      iteration: The iteration to format.
      max_character_length: Maximum character length for results.

  Returns:
      A list of messages to add to the next prompt.
  """
  # Handle None responses - use empty string to avoid corrupting message history
  response_content = (
      iteration.response if iteration.response is not None else ""
  )
  messages = [{"role": "assistant", "content": response_content}]

  for code_block in iteration.code_blocks:
    code = code_block.code
    result = code_block.result
    result_str = format_execution_result(result)

    if len(result_str) > max_character_length:
      result_str = (
          result_str[:max_character_length]
          + f"... + [{len(result_str) - max_character_length} chars...]"
      )

    execution_message = {
        "role": "user",
        "content": (
            f"Code executed:\n```python\n{code}\n```\n\nREPL"
            f" output:\n{result_str}"
        ),
    }
    messages.append(execution_message)

  return messages
