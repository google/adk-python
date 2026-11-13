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

"""Environment toolset that provides tools to interact with an environment."""

from __future__ import annotations

import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from typing_extensions import override

from ..base_toolset import BaseToolset
from ._constants import ENVIRONMENT_INSTRUCTION
from ._constants import ENVIRONMENT_WORKING_DIR_INSTRUCTION
from ._edit_file_tool import EditFileTool
from ._execute_tool import ExecuteTool
from ._read_file_tool import ReadFileTool
from ._utils import close_if_initialized
from ._write_file_tool import WriteFileTool

if TYPE_CHECKING:
  from ...agents.readonly_context import ReadonlyContext
  from ...environment._base_environment import BaseEnvironment
  from ...models.llm_request import LlmRequest
  from ..base_tool import BaseTool
  from ..tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)


class EnvironmentToolset(BaseToolset):
  """Toolset providing tools to interact with an environment.

  Tools provided:
    - **Execute** -- run shell commands
    - **ReadFile** -- read file contents
    - **EditFile** -- surgical text replacement
    - **WriteFile**q -- create/overwrite files

  The toolset injects an environment-level system instruction on each
  LLM call that establishes environment identity and tool selection
  rules.

  The environment is initialized on the first tool call that uses it, with
  that call's context, so ``initialize`` can write to session state.
  """

  def __init__(
      self,
      *,
      environment: BaseEnvironment,
      max_output_chars: Optional[int] = None,
      **kwargs: Any,
  ):
    """Create an environment toolset.

    Args:
      environment: The environment used to execute commands and perform file
        I/O.
      max_output_chars: Maximum character limit for stdout/stderr/file
        truncation.
      **kwargs: Forwarded to ``BaseToolset.__init__``.
    """
    super().__init__(**kwargs)
    self._environment = environment
    self._max_output_chars = max_output_chars

  def __getstate__(self) -> dict[str, Any]:
    state = dict(super().__getstate__())
    # Cached prefixed tools wrap local functions, which cannot be pickled.
    state['_cached_prefixed_tools'] = None
    state['_cached_invocation_id'] = None
    return state

  def __setstate__(self, state: dict[str, Any]) -> None:
    super_setstate = getattr(super(), '__setstate__', None)
    if super_setstate is not None:
      super_setstate(state)
    else:
      self.__dict__.update(state)
    # What `initialize` set up, such as a temporary working directory, may not
    # exist in the process that unpickles the environment, so make the first
    # tool call initialize it again.
    if self._environment.is_initialized:
      try:
        self._environment.is_initialized = False
      except AttributeError:
        # `is_initialized` is a read-only property the environment manages.
        pass

  @override
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    return [
        ExecuteTool(self._environment, max_output_chars=self._max_output_chars),
        ReadFileTool(
            self._environment, max_output_chars=self._max_output_chars
        ),
        EditFileTool(self._environment),
        WriteFileTool(self._environment),
    ]

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Inject environment-level system instruction.

    Does not initialize the environment. Until a tool call has, the
    instruction leaves out the working directory, which may not exist yet.
    """
    instruction = ENVIRONMENT_INSTRUCTION
    # TODO: Check initialization through a method that takes `ctx`, so a
    # session-scoped environment can answer from `ctx.state`.
    if self._environment.is_initialized:
      instruction = (
          ENVIRONMENT_WORKING_DIR_INSTRUCTION.format(
              working_dir=self._environment.working_dir
          )
          + '\n'
          + instruction
      )
    llm_request.append_instructions([instruction])

  @override
  async def close(self) -> None:
    await close_if_initialized(self._environment)
    self._cached_prefixed_tools = None
    self._cached_invocation_id = None
    await super().close()
