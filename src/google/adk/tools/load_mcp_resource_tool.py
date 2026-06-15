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

import base64
import json
import logging
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ..features import FeatureName
from ..features import is_feature_enabled
from ..models.llm_request import LlmRequest
from .base_tool import BaseTool

if TYPE_CHECKING:
  from mcp_toolset import McpToolset

  from .tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


class LoadMcpResourceTool(BaseTool):
  """A tool that loads the MCP resources and adds them to the session."""

  def __init__(self, mcp_toolset: McpToolset):
    super().__init__(
        name="load_mcp_resource",
        description="""Loads resources from the MCP server.

NOTE: Call when you need access to resources.""",
    )
    self._mcp_toolset = mcp_toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
      return types.FunctionDeclaration(
          name=self.name,
          description=self.description,
          parameters_json_schema={
              "type": "object",
              "properties": {
                  "resource_names": {
                      "type": "array",
                      "items": {"type": "string"},
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
                "resource_names": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.STRING,
                    ),
                )
            },
        ),
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    resource_names: list[str] = args.get("resource_names", [])
    return {
        "resource_names": resource_names,
        "status": (
            "resource contents temporarily inserted and removed. to access"
            " these resources, call load_mcp_resource tool again."
        ),
    }

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    await super().process_llm_request(
        tool_context=tool_context,
        llm_request=llm_request,
    )
    await self._append_resources_to_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

  async def _append_resources_to_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ):
    try:
      resource_names = await self._mcp_toolset.list_resources()
      if resource_names:
        llm_request.append_instructions([f"""You have a list of MCP resources:
{json.dumps(resource_names)}

When the user asks questions about any of the resources, you should call the
`load_mcp_resource` function to load the resource. Always call load_mcp_resource
before answering questions related to the resources.
"""])
    except Exception as e:
      logger.warning("Failed to list MCP resources: %s", e)

    # Attach content
    if llm_request.contents and llm_request.contents[-1].parts:
      function_response = llm_request.contents[-1].parts[0].function_response
      if function_response and function_response.name == self.name:
        response = function_response.response or {}
        resource_names = response.get("resource_names", [])
        for resource_name in resource_names:
          try:
            contents = await self._mcp_toolset.read_resource(resource_name)

            for content in contents:
              part = self._mcp_content_to_part(content, resource_name)
              llm_request.contents.append(
                  types.Content(
                      role="user",
                      parts=[
                          types.Part.from_text(
                              text=f"Resource {resource_name} is:"
                          ),
                          part,
                      ],
                  )
              )
          except Exception as e:
            logger.warning(
                "Failed to read MCP resource '%s': %s", resource_name, e
            )
            continue

  def _mcp_content_to_part(
      self, content: Any, resource_name: str
  ) -> types.Part:
    if hasattr(content, "text") and content.text is not None:
      return types.Part.from_text(text=content.text)
    elif hasattr(content, "blob") and content.blob is not None:
      try:
        data = base64.b64decode(content.blob)
        # Basic check for mime type or default
        mime_type = content.mimeType or "application/octet-stream"
        return types.Part.from_bytes(data=data, mime_type=mime_type)
      except Exception:
        return types.Part.from_text(
            text=f"[Binary content for {resource_name} could not be decoded]"
        )
    else:
      return types.Part.from_text(
          text=f"[Unknown content type for {resource_name}]"
      )
