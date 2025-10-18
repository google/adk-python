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

"""Utility functions for generating enhanced error messages."""

from difflib import get_close_matches


def format_not_found_error(
    item_name: str,
    item_type: str,
    available_items: list[str],
    causes: list[str],
    fixes: list[str],
) -> str:
  """Format an enhanced 'not found' error message with fuzzy matching.

  This utility creates consistent, actionable error messages when tools,
  agents, or other named items cannot be found. It includes:
  - Clear identification of what was not found
  - List of available items (truncated to 20 for readability)
  - Possible causes for the error
  - Suggested fixes
  - Fuzzy matching suggestions for typos

  Args:
    item_name: The name of the item that was not found.
    item_type: The type of item (e.g., 'tool', 'agent', 'function').
    available_items: List of available item names.
    causes: List of possible causes for the error.
    fixes: List of suggested fixes.

  Returns:
    Formatted error message string with all components.

  Example:
    >>> error_msg = format_not_found_error(
    ...     item_name='get_wether',
    ...     item_type='tool',
    ...     available_items=['get_weather', 'calculate_sum'],
    ...     causes=['LLM hallucinated the name', 'Typo in function name'],
    ...     fixes=['Check spelling', 'Verify tool is registered']
    ... )
    >>> raise ValueError(error_msg)
  """
  # Truncate available items to first 20 for readability
  if len(available_items) > 20:
    items_preview = ', '.join(available_items[:20])
    items_msg = (
        f'Available {item_type}s (showing first 20 of'
        f' {len(available_items)}): {items_preview}...'
    )
  else:
    items_msg = f"Available {item_type}s: {', '.join(available_items)}"

  # Build error message from parts
  error_parts = [
      f"{item_type.capitalize()} '{item_name}' is not found.",
      items_msg,
      'Possible causes:\n'
      + '\n'.join(f'  {i+1}. {cause}' for i, cause in enumerate(causes)),
      'Suggested fixes:\n' + '\n'.join(f'  - {fix}' for fix in fixes),
  ]

  # Add fuzzy matching suggestions for typos
  close_matches = get_close_matches(item_name, available_items, n=3, cutoff=0.6)
  if close_matches:
    suggestions = '\n'.join(f'  - {match}' for match in close_matches)
    error_parts.append(f'Did you mean one of these?\n{suggestions}')

  return '\n\n'.join(error_parts)
