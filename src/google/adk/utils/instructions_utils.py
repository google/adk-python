# Copyright 2026 Google LLC
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

import logging
import re
from typing import Awaitable
from typing import Callable
from typing import Union

import jinja2
from typing_extensions import TypeAlias

from ..agents.readonly_context import ReadonlyContext
from ..sessions.state import State

__all__ = [
    'InstructionProvider',
    'inject_session_state',
]

logger = logging.getLogger('google_adk.' + __name__)

# Type alias for agent instruction providers: a callable that receives a
# ReadonlyContext and returns an instruction string (sync or async).
InstructionProvider: TypeAlias = Callable[
    [ReadonlyContext], Union[str, Awaitable[str]]
]

_TEMPLATE_VAR_PATTERN = re.compile(r'{+[^{}]*}+')


async def inject_session_state(
    template: str,
    readonly_context: ReadonlyContext,
    use_jinja2: bool = False,
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
      model="gemini-2.5-flash",
      name="agent",
      instruction=build_instruction,
  )
  ```

  For more expressive templates with conditionals and loops, set
  ``use_jinja2=True``.  Session state variables are available directly by
  name (``{{ var_name }}``) and artifacts can be loaded with the async
  ``artifact`` helper (``{{ artifact("file_name") }}``).

  e.g.
  ```
  async def build_instruction(
      readonly_context: ReadonlyContext,
  ) -> str:
    return await inject_session_state(
        '{% if user_name %}Hello {{ user_name }}!{% endif %}',
        readonly_context,
        use_jinja2=True,
    )
  ```

  Args:
    template: The instruction template.
    readonly_context: The read-only context.
    use_jinja2: If True, render the template with Jinja2 instead of the
      default regex-based engine.  Defaults to False for backward
      compatibility.

  Returns:
    The instruction template with values populated.
  """
  if use_jinja2:
    return await _render_with_jinja2(template, readonly_context)
  return await _render_with_regex(template, readonly_context)


async def _render_with_regex(
    template: str,
    readonly_context: ReadonlyContext,
) -> str:
  """Renders *template* using the legacy regex-based substitution engine."""

  # The substitution pattern requires a '{', so a template without one can
  # never match. Return it as-is to avoid the regex scan on every LLM call,
  # which is the common case for static instructions.
  if '{' not in template:
    return template

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
      if artifact is None:
        if optional:
          logger.debug(
              'Artifact %s not found, replacing with empty string', var_name
          )
          return ''
        else:
          raise KeyError(f'Artifact {var_name} not found.')
      return str(artifact)
    else:
      if not _is_valid_state_name(var_name):
        return match.group()
      if var_name in invocation_context.session.state:
        value = invocation_context.session.state[var_name]
        if value is None:
          return ''
        return str(value)
      else:
        if optional:
          logger.debug(
              'Context variable %s not found, replacing with empty string',
              var_name,
          )
          return ''
        else:
          raise KeyError(f'Context variable not found: `{var_name}`.')

  return await _async_sub(_TEMPLATE_VAR_PATTERN, _replace_match, template)


async def _render_with_jinja2(
    template: str,
    readonly_context: ReadonlyContext,
) -> str:
  """Renders *template* using a Jinja2 environment.

  Session state variables are exposed as top-level template variables.
  Artifacts can be loaded with the ``artifact(filename)`` async callable
  available inside the template.

  Args:
    template: A Jinja2 template string.
    readonly_context: The read-only context.

  Returns:
    The rendered string.
  """
  invocation_context = readonly_context._invocation_context

  async def _load_artifact(filename: str) -> str:
    if invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    artifact = await invocation_context.artifact_service.load_artifact(
        app_name=invocation_context.session.app_name,
        user_id=invocation_context.session.user_id,
        session_id=invocation_context.session.id,
        filename=filename,
    )
    if artifact is None:
      raise KeyError(f'Artifact {filename} not found.')
    return str(artifact)

  env = jinja2.Environment(
      enable_async=True,
      undefined=jinja2.StrictUndefined,
      autoescape=False,
  )
  jinja_template = env.from_string(template)

  context_vars = dict(invocation_context.session.state)
  context_vars['artifact'] = _load_artifact

  return await jinja_template.render_async(**context_vars)


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
