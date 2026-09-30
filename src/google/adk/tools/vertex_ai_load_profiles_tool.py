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

from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ..features import FeatureName
from ..features import is_feature_enabled
from .function_tool import FunctionTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService


class VertexAiLoadProfilesTool(FunctionTool):
  """A tool that loads a user's structured profiles from Vertex Memory Bank."""

  def __init__(self, memory_service: VertexAiMemoryBankService):
    super().__init__(self.load_profiles)
    self._memory_service = memory_service

  async def load_profiles(self, tool_context: ToolContext) -> dict[str, Any]:
    """Loads structured user profiles for the current user."""
    profiles = await self._memory_service.retrieve_profiles(
        app_name=tool_context.session.app_name,
        user_id=tool_context.user_id,
    )
    return {
        'profiles': [profile.profile for profile in profiles if profile.profile]
    }

  @override
  def _get_declaration(self) -> types.FunctionDeclaration | None:
    if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
      return types.FunctionDeclaration(
          name=self.name,
          description=self.description,
          parameters_json_schema={
              'type': 'object',
              'properties': {},
          },
      )
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={},
        ),
    )
