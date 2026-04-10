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

from ...utils.feature_decorator import experimental
from ..base_toolset import BaseToolset
from ._constants import ENVIRONMENT_INSTRUCTION
from ._tools import EditFileTool
from ._tools import ExecuteTool
from ._tools import ReadFileTool
from ._tools import WriteFileTool

if TYPE_CHECKING:
  from ...agents.readonly_context import ReadonlyContext
  from ...environment._base_environment import BaseEnvironment
  from ...models.llm_request import LlmRequest
  from ..base_tool import BaseTool
  from ..tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)


@experimental
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
  """

  def __init__(
      self,
      *,
      environment: BaseEnvironment,
      **kwargs: Any,
  ):
    """Create an environment toolset.

    Args:
      environment: The environment used to execute commands and
        perform file I/O.
      **kwargs: Forwarded to ``BaseToolset.__init__``.
    """
    super().__init__(**kwargs)
    self._environment = environment
    self._environment_initialized = False

  @override
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    if not self._environment_initialized:
      await self._environment.initialize()
      self._environment_initialized = True
    return [
        ExecuteTool(self._environment),
        ReadFileTool(self._environment),
        EditFileTool(self._environment),
        WriteFileTool(self._environment),
    ]

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Inject environment-level system instruction."""
    if not self._environment_initialized:
      await self._environment.initialize()
      self._environment_initialized = True
    working_dir = self._environment.working_dir
    instruction = ENVIRONMENT_INSTRUCTION.format(
        working_dir=working_dir,
    )
    llm_request.append_instructions([instruction])

  @override
  async def close(self) -> None:
    if self._environment_initialized:
      await self._environment.close()
      self._environment_initialized = False
