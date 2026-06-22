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

from ..agents.readonly_context import ReadonlyContext
from ..sessions.state import State

__all__ = [
    'inject_session_state',
]

logger = logging.getLogger('google_adk.' + __name__)


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

  When ``use_jinja2`` is ``True``, the template is rendered with a sandboxed
  Jinja2 environment instead of the default regex-based substitution. This
  enables control flow such as conditionals and loops in addition to plain
  variable injection. Inside a Jinja2 template, session state is available as
  the ``state`` mapping (e.g. ``{{ state['var_name'] }}``), while artifacts are
  loaded with the ``artifact`` helper (e.g. ``{{ artifact('file_name') }}``).
  The artifact helper is asynchronous and is awaited automatically by the
  async Jinja2 environment.

  e.g.
  ```
  return await inject_session_state(
      '{% if state["is_premium"] %}Premium user.{% else %}Free user.{% endif %}'
      '{% for item in state["items"] %}- {{ item }}\\n{% endfor %}',
      readonly_context,
      use_jinja2=True,
  )
  ```

  Args:
    template: The instruction template.
    readonly_context: The read-only context.
    use_jinja2: If True, render the template with a sandboxed Jinja2 environment.
      If False (the default), use the regex-based ``{var}`` substitution. The
      default preserves backward-compatible behavior.

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
  """Renders the template using the regex-based ``{var}`` substitution.

  This is the default, backward-compatible rendering path. It replaces
  ``{var_name}`` with the matching session state value and
  ``{artifact.file_name}`` with the loaded artifact content. A trailing ``?``
  (e.g. ``{var_name?}``) marks the reference as optional, replacing it with an
  empty string when the value is missing instead of raising.

  Args:
    template: The instruction template.
    readonly_context: The read-only context.

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

  return await _async_sub(r'{+[^{}]*}+', _replace_match, template)


async def _render_with_jinja2(
    template: str,
    readonly_context: ReadonlyContext,
) -> str:
  """Renders the template using a sandboxed Jinja2 environment.

  Unlike the regex-based path, this supports full Jinja2 control flow such as
  conditionals (``{% if %}``), loops (``{% for %}``) and filters, in addition
  to variable injection.

  The following names are exposed to the template:
    - ``state``: the session state mapping, e.g. ``{{ state['var_name'] }}`` or
      ``{% if state['flag'] %}...{% endif %}``.
    - ``artifact``: an async accessor that loads an artifact by filename, e.g.
      ``{{ artifact('file_name') }}``. The environment runs with
      ``enable_async=True``, so the returned coroutine is awaited
      automatically; a missing artifact renders as an empty string.

  A ``jinja2.sandbox.SandboxedEnvironment`` is used because instruction
  templates may include user- or session-provided data, and the sandbox blocks
  access to unsafe attributes and operations. ``jinja2`` is imported lazily so
  that installations that never use this path are not required to have it.

  Args:
    template: The instruction template.
    readonly_context: The read-only context.

  Returns:
    The instruction template with values populated.

  Raises:
    ValueError: If the artifact service is required but not initialized.
  """
  try:
    from jinja2.sandbox import SandboxedEnvironment
  except ImportError as e:
    raise ImportError(
        'jinja2 is required to use Jinja2-based instruction templating'
        ' (use_jinja2=True). Install it with `pip install google-adk[jinja]`'
        ' (or `pip install jinja2`).'
    ) from e

  invocation_context = readonly_context._invocation_context
  session_state = invocation_context.session.state

  async def _artifact(name: str):
    if invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    artifact = await invocation_context.artifact_service.load_artifact(
        app_name=invocation_context.session.app_name,
        user_id=invocation_context.session.user_id,
        session_id=invocation_context.session.id,
        filename=name,
    )
    return str(artifact) if artifact is not None else ''

  env = SandboxedEnvironment(enable_async=True)
  jinja_template = env.from_string(template)
  return await jinja_template.render_async(
      state=session_state,
      artifact=_artifact,
  )


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
