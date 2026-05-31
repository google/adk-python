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

from google.genai import types
from typing_extensions import override

from ...features import FeatureName
from ...features import is_feature_enabled
from ..base_tool import BaseTool


class BaseRetrievalTool(BaseTool):

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
      return types.FunctionDeclaration(
          name=self.name,
          description=self.description,
          parameters_json_schema={
              'type': 'object',
              'properties': {
                  'query': {
                      'type': 'string',
                      'description': 'The query to retrieve.',
                  },
              },
          },
      )
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                'query': types.Schema(
                    type=types.Type.STRING,
                    description='The query to retrieve.',
                ),
            },
        ),
    )
