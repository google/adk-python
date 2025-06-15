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
  from google.adk.utils import instructions_utils

  async def build_instruction(
      readonly_context: ReadonlyContext,
  ) -> str:
    return await instructions_utils.inject_session_state(
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
  
  # Apply session state preprocessor to enhance session state for template processing
  enhanced_state = dict(invocation_context.session.state)
  
  try:
    # Try to find and call agent's session_state_preprocessor
    import asyncio
    
    app_name = invocation_context.session.app_name
    
    # Get agents_dir from fast_api module
    from google.adk.cli.fast_api import _agents_dir
    from google.adk.cli.utils.agent_loader import AgentLoader
    
    if _agents_dir:
      # Try to load agent module and look for session_state_preprocessor
      try:
        # Import the agent module directly using standard Python import
        import sys
        import importlib
        
        # Try common agent module patterns
        for module_pattern in [f"{app_name}", f"agents.{app_name}"]:
          try:
            if module_pattern in sys.modules:
              agent_python_module = importlib.reload(sys.modules[module_pattern])
            else:
              agent_python_module = importlib.import_module(module_pattern)
            
            if hasattr(agent_python_module, 'session_state_preprocessor'):
              preprocessor = getattr(agent_python_module, 'session_state_preprocessor')
              
              if callable(preprocessor):
                if asyncio.iscoroutinefunction(preprocessor):
                  enhanced_state = await preprocessor(enhanced_state)
                else:
                  enhanced_state = preprocessor(enhanced_state)
              break
              
          except ImportError:
            continue
            
      except Exception:
        # Silent fallback - continue with original state
        pass
        
  except Exception:
    # Silent fallback - use original state if preprocessor fails
    pass

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
      if not _is_valid_state_name(var_name):
        return match.group()
      if var_name in enhanced_state:
        return str(enhanced_state[var_name])
      else:
        if optional:
          return ''
        else:
          raise KeyError(f'Context variable not found: `{var_name}`.')

  return await _async_sub(r'{+[^{}]*}+', _replace_match, template)


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
