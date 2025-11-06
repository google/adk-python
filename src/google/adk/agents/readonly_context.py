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

from types import MappingProxyType
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from google.genai import types

  from ..sessions.session import Session
  from .invocation_context import InvocationContext
  from .run_config import RunConfig


class ReadonlyContext:

  def __init__(
      self,
      invocation_context: InvocationContext,
  ) -> None:
    self._invocation_context = invocation_context

  @property
  def user_content(self) -> Optional[types.Content]:
    """The user content that started this invocation. READONLY field."""
    return self._invocation_context.user_content

  @property
  def invocation_id(self) -> str:
    """The current invocation id."""
    return self._invocation_context.invocation_id

  @property
  def agent_name(self) -> str:
    """The name of the agent that is currently running."""
    return self._invocation_context.agent.name

  @property
  def state(self) -> MappingProxyType[str, Any]:
    """The state of the current session. READONLY field."""
    return MappingProxyType(self._invocation_context.session.state)

  @property
  def session(self) -> Session:
    """The current session for this invocation."""
    return self._invocation_context.session

  @property
  def run_config(self) -> Optional[RunConfig]:
    """The run config of the current invocation. READONLY field."""
    return self._invocation_context.run_config
