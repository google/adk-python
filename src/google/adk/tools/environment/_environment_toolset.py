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

import asyncio
import logging
import threading
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from typing_extensions import override

from ...utils.feature_decorator import experimental
from ..base_toolset import BaseToolset
from ._constants import ENVIRONMENT_INSTRUCTION
from ._edit_file_tool import EditFileTool
from ._execute_tool import ExecuteTool
from ._read_file_tool import ReadFileTool
from ._write_file_tool import WriteFileTool

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
    self._environment_initialized = False
    self._init_lock_map: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
    self._lock_map_lock = threading.Lock()

  def _get_init_lock(self) -> asyncio.Lock:
    # Note: Mutual exclusion for lazy initialization is guaranteed per event
    # loop; concurrent threads running distinct event loops on a shared toolset
    # instance each acquire their own loop-local lock.
    current_loop = asyncio.get_running_loop()
    with self._lock_map_lock:
      lock = self._init_lock_map.get(current_loop)
      if lock is None:
        for stale_loop in [
            loop for loop in self._init_lock_map if loop.is_closed()
        ]:
          del self._init_lock_map[stale_loop]
        lock = asyncio.Lock()
        self._init_lock_map[current_loop] = lock
      return lock

  def __getstate__(self) -> dict[str, Any]:
    state = dict(super().__getstate__())
    state.pop('_lock_map_lock', None)
    state['_init_lock_map'] = {}
    state['_environment_initialized'] = False
    state['_cached_prefixed_tools'] = None
    state['_cached_invocation_id'] = None
    return state

  def __setstate__(self, state: dict[str, Any]) -> None:
    super_setstate = getattr(super(), '__setstate__', None)
    if super_setstate is not None:
      super_setstate(state)
    else:
      self.__dict__.update(state)
    self._init_lock_map = {}
    self._lock_map_lock = threading.Lock()

  async def _ensure_initialized(self) -> None:
    if self._environment_initialized:
      return
    async with self._get_init_lock():
      if not self._environment_initialized:
        await self._environment.initialize()
        self._environment_initialized = True

  @override
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    await self._ensure_initialized()
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
    """Inject environment-level system instruction."""
    await self._ensure_initialized()
    working_dir = self._environment.working_dir
    instruction = ENVIRONMENT_INSTRUCTION.format(
        working_dir=working_dir,
    )
    llm_request.append_instructions([instruction])

  @override
  async def close(self) -> None:
    async with self._get_init_lock():
      if self._environment_initialized:
        await self._environment.close()
        self._environment_initialized = False
      self._cached_prefixed_tools = None
      self._cached_invocation_id = None
    await super().close()
