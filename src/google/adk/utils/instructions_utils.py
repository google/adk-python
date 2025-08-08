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

from __future__ import annotations

import re

from ..agents.readonly_context import ReadonlyContext
from ..sessions.state import State

__all__ = [
    'inject_session_state',
]


async def inject_session_state(
    template: str,
    readonly_context: ReadonlyContext,
) -> str:
  """Populates values in the instruction template, e.g. state, artifact, etc.

  This method is intended to be used in InstructionProvider based instruction
  and global_instruction which are called with readonly_context.

  e.g.
  ```
  ...
  from google.adk.utils.instructions_utils import inject_session_state

  async def build_instruction(
      readonly_context: ReadonlyContext,
  ) -> str:
    return await inject_session_state(
        'You can inject a state variable like {var_name} or an artifact '
        '{artifact.file_name} into the instruction template.',
        readonly_context,
    )

  agent = Agent(
      model="gemini-2.0-flash",
      name="agent",
      instruction=build_instruction,
  )
  ```

  Args:
    template: The instruction template.
    readonly_context: The read-only context

  Returns:
    The instruction template with values populated.
  """

  invocation_context = readonly_context._invocation_context

  async def _async_sub(pattern, repl_async_fn, string) -> str:
    result = []
    last_end = 0
    for match in re.finditer(pattern, string):
      result.append(string[last_end : match.start()])
      replacement = await repl_async_fn(match)
      result.append(replacement)
      last_end = match.end()
    result.append(string[last_end:])
    return ''.join(result)

  async def _replace_match(match) -> str:
    var_name = match.group().lstrip('{').rstrip('}').strip()
    optional = False
    if var_name.endswith('?'):
      optional = True
      var_name = var_name.removesuffix('?')
    if var_name.startswith('artifact.'):
      var_name = var_name.removeprefix('artifact.')
      if invocation_context.artifact_service is None:
        raise ValueError('Artifact service is not initialized.')
      artifact = await invocation_context.artifact_service.load_artifact(
          app_name=invocation_context.session.app_name,
          user_id=invocation_context.session.user_id,
          session_id=invocation_context.session.id,
          filename=var_name,
      )
      if not var_name:
        raise KeyError(f'Artifact {var_name} not found.')
      return str(artifact)
    else:
      if not _is_valid_state_name_or_nested(var_name):
        return match.group()
      
      try:
        keys = _parse_nested_path(var_name)
        value = _get_nested_value(invocation_context.session.state, keys)
        return str(value)
      except KeyError as e:
        if optional:
          return ''
        else:
          raise KeyError(f'Context variable not found: `{var_name}`. {str(e)}')

  return await _async_sub(r'{+[^{}]*}+', _replace_match, template)


def _parse_nested_path(var_name: str) -> list[str]:
  """Parse a nested variable path into individual keys.
  
  Supports both dot notation (key.subkey) and bracket notation (key['subkey']).
  Mixed notation is also supported (key.subkey['nested']).
  
  Args:
    var_name: The variable name to parse (e.g., "user.profile.name" or "user['profile']['name']")
    
  Returns:
    List of keys to traverse the nested structure.
  """
  if '.' not in var_name and '[' not in var_name:
    return [var_name]
  
  keys = []
  current_key = ""
  i = 0
  
  while i < len(var_name):
    char = var_name[i]
    
    if char == '.':
      if current_key:
        keys.append(current_key)
        current_key = ""
    elif char == '[':
      if current_key:
        keys.append(current_key)
        current_key = ""
      bracket_end = var_name.find(']', i)
      if bracket_end == -1:
        raise ValueError(f"Unclosed bracket in variable name: {var_name}")
      
      bracket_content = var_name[i+1:bracket_end]
      if (bracket_content.startswith('"') and bracket_content.endswith('"')) or \
         (bracket_content.startswith("'") and bracket_content.endswith("'")):
        bracket_content = bracket_content[1:-1]
      
      keys.append(bracket_content)
      i = bracket_end
    else:
      current_key += char
    
    i += 1
  
  if current_key:
    keys.append(current_key)
  
  return keys


def _get_nested_value(data: dict, keys: list[str]):
  """Get a value from nested dictionary structure using a list of keys.
  
  Args:
    data: The dictionary to traverse
    keys: List of keys to traverse the nested structure
    
  Returns:
    The value at the nested path
    
  Raises:
    KeyError: If any key in the path doesn't exist
  """
  current = data
  for key in keys:
    if not isinstance(current, dict):
      raise KeyError(f"Cannot access key '{key}' on non-dict value")
    if key not in current:
      raise KeyError(f"Key '{key}' not found")
    current = current[key]
  return current


def _is_valid_state_name(var_name):
  """Checks if the variable name is a valid state name.

  Valid state is either:
    - Valid identifier
    - <Valid prefix>:<Valid identifier>
  All the others will just return as it is.

  Args:
    var_name: The variable name to check.

  Returns:
    True if the variable name is a valid state name, False otherwise.
  """
  parts = var_name.split(':')
  if len(parts) == 1:
    return var_name.isidentifier()

  if len(parts) == 2:
    prefixes = [State.APP_PREFIX, State.USER_PREFIX, State.TEMP_PREFIX]
    if (parts[0] + ':') in prefixes:
      return parts[1].isidentifier()
  return False


def _is_valid_state_name_or_nested(var_name: str) -> bool:
  """Checks if the variable name is a valid state name or nested path.
  
  Valid state is either:
    - Valid identifier (existing behavior)
    - <Valid prefix>:<Valid identifier> (existing behavior)  
    - Nested path with dot notation (key.subkey.nested)
    - Nested path with bracket notation (key['subkey']['nested'])
    - Mixed notation (key.subkey['nested'])
  
  Args:
    var_name: The variable name to check.
    
  Returns:
    True if the variable name is valid, False otherwise.
  """
  if _is_valid_state_name(var_name):
    return True
  
  try:
    keys = _parse_nested_path(var_name)
    for key in keys:
      if not _is_valid_state_name(key):
        return False
    return len(keys) > 1
  except ValueError:
    return False
