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

"""The BigQuery tool wrapper that applies the tool config's default ids."""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Optional

from google.auth.credentials import Credentials
from google.genai import types
from pydantic import BaseModel
from typing_extensions import override

from ...tools._google_credentials import BaseGoogleCredentialsConfig
from ...tools.google_tool import GoogleTool
from ...tools.tool_context import ToolContext
from .config import BigQueryToolConfig

# Tool argument name -> the BigQueryToolConfig field holding its default.
_ARG_DEFAULT_SETTINGS = {
    "project_id": "default_project_id",
    "dataset_id": "default_dataset_id",
}


class BigQueryTool(GoogleTool):
  """A BigQuery tool that applies the default ids from the tool settings.

  For every argument the settings supply a default for, the argument is dropped
  from the declaration's required list and filled in at call time. Without this
  the model has no way to guess the project or dataset the agent builder meant,
  so it asks the user for them at the start of every conversation.
  """

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      credentials_config: Optional[BaseGoogleCredentialsConfig] = None,
      tool_settings: Optional[BigQueryToolConfig] = None,
  ):
    super().__init__(
        func=func,
        credentials_config=credentials_config,
        tool_settings=tool_settings,
    )
    self._arg_defaults: dict[str, str] = {}
    if not self._spec.has_signature:
      return
    parameters = self._spec.signature.parameters
    for arg, setting in _ARG_DEFAULT_SETTINGS.items():
      value = getattr(tool_settings, setting, None)
      if value and arg in parameters:
        self._arg_defaults[arg] = value

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    declaration = super()._get_declaration()
    if declaration is None or not self._arg_defaults:
      return declaration

    if declaration.parameters is not None:
      for name, schema in (declaration.parameters.properties or {}).items():
        if name in self._arg_defaults:
          schema.description = self._describe_default(name, schema.description)
      declaration.parameters.required = self._drop_defaulted(
          declaration.parameters.required
      )

    json_schema = declaration.parameters_json_schema
    if isinstance(json_schema, dict):
      for name, schema in (json_schema.get("properties") or {}).items():
        if name in self._arg_defaults and isinstance(schema, dict):
          schema["description"] = self._describe_default(
              name, schema.get("description")
          )
      required = self._drop_defaulted(json_schema.get("required"))
      if required:
        json_schema["required"] = required
      else:
        json_schema.pop("required", None)

    return declaration

  def _describe_default(self, name: str, description: Optional[str]) -> str:
    sentence = f"Defaults to {self._arg_defaults[name]!r} when not provided."
    return f"{description} {sentence}" if description else sentence

  def _drop_defaulted(
      self, required: Optional[list[str]]
  ) -> Optional[list[str]]:
    if not required:
      return None
    return [name for name in required if name not in self._arg_defaults] or None

  @override
  async def _run_async_with_credential(
      self,
      credentials: Optional[Credentials],
      tool_settings: Optional[BaseModel],
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    if self._arg_defaults:
      args = dict(args)
      for name, value in self._arg_defaults.items():
        if not args.get(name):
          args[name] = value
    return await super()._run_async_with_credential(
        credentials, tool_settings, args, tool_context
    )
