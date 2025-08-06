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

from abc import ABC
import inspect
import logging
from typing import Any
from typing import Callable
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict

from ..utils.variant_utils import get_google_llm_variant
from ..utils.variant_utils import GoogleLLMVariant
from .tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest

SelfTool = TypeVar("SelfTool", bound="BaseTool")


class BaseTool(ABC):
  """The base class for all tools."""

  name: str
  """The name of the tool."""
  description: str
  """The description of the tool."""

  is_long_running: bool = False
  """Whether the tool is a long running operation, which typically returns a
  resource id first and finishes the operation later."""

  custom_metadata: Optional[dict[str, Any]] = None
  """The custom metadata of the BaseTool.

  An optional key-value pair for storing and retrieving tool-specific metadata,
  such as tool manifests, etc.

  NOTE: the entire dict must be JSON serializable.
  """

  def __init__(
      self,
      *,
      name,
      description,
      is_long_running: bool = False,
      custom_metadata: Optional[dict[str, Any]] = None,
  ):
    self.name = name
    self.description = description
    self.is_long_running = is_long_running
    self.custom_metadata = custom_metadata

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    """Gets the OpenAPI specification of this tool in the form of a FunctionDeclaration.

    NOTE:
      - Required if subclass uses the default implementation of
        `process_llm_request` to add function declaration to LLM request.
      - Otherwise, can be skipped, e.g. for a built-in GoogleSearch tool for
        Gemini.

    Returns:
      The FunctionDeclaration of this tool, or None if it doesn't need to be
      added to LlmRequest.config.
    """
    return None

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Runs the tool with the given arguments and context.

    NOTE:
      - Required if this tool needs to run at the client side.
      - Otherwise, can be skipped, e.g. for a built-in GoogleSearch tool for
        Gemini.

    Args:
      args: The LLM-filled arguments.
      tool_context: The context of the tool.

    Returns:
      The result of running the tool.
    """
    raise NotImplementedError(f"{type(self)} is not implemented")

  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Processes the outgoing LLM request for this tool.

    Use cases:
    - Most common use case is adding this tool to the LLM request.
    - Some tools may just preprocess the LLM request before it's sent out.

    Args:
      tool_context: The context of the tool.
      llm_request: The outgoing LLM request, mutable this method.
    """
    if (function_declaration := self._get_declaration()) is None:
      return

    llm_request.tools_dict[self.name] = self
    if tool_with_function_declarations := _find_tool_with_function_declarations(
        llm_request
    ):
      if tool_with_function_declarations.function_declarations is None:
        tool_with_function_declarations.function_declarations = []
      tool_with_function_declarations.function_declarations.append(
          function_declaration
      )
    else:
      llm_request.config = (
          types.GenerateContentConfig()
          if not llm_request.config
          else llm_request.config
      )
      llm_request.config.tools = (
          [] if not llm_request.config.tools else llm_request.config.tools
      )
      llm_request.config.tools.append(
          types.Tool(function_declarations=[function_declaration])
      )

  @property
  def _api_variant(self) -> GoogleLLMVariant:
    return get_google_llm_variant()

  @classmethod
  def from_config(
      cls: Type[SelfTool], config: ToolArgsConfig, config_abs_path: str
  ) -> SelfTool:
    """Creates a tool instance from a config.

    This default implementation uses inspect to automatically map config values
    to constructor arguments based on their type hints. Subclasses should
    override this method for custom initialization logic.

    Args:
      config: The config for the tool.
      config_abs_path: The absolute path to the config file that contains the
        tool config.

    Returns:
      The tool instance.
    """
    from ..agents import config_agent_utils

    # Get the constructor signature and resolve type hints
    sig = inspect.signature(cls.__init__)
    type_hints = get_type_hints(cls.__init__)
    config_dict = config.model_dump()
    kwargs = {}

    # Iterate through constructor parameters (skip "self")
    for param_name, _ in sig.parameters.items():
      if param_name == "self":
        continue
      param_type = type_hints.get(param_name)

      if param_name in config_dict:
        value = config_dict[param_name]

        # Get the actual type T of the parameter if it's Optional[T]
        if get_origin(param_type) is Union:
          # This is Optional[T] which is Union[T, None]
          args = get_args(param_type)
          if len(args) == 2 and type(None) in args:
            # Get the non-None type
            actual_type = args[0] if args[1] is type(None) else args[1]
            param_type = actual_type

        if param_type in (int, str, bool, float):
          kwargs[param_name] = value
        elif (
            inspect.isclass(param_type)
            and issubclass(param_type, BaseModel)
            and value is not None
        ):
          kwargs[param_name] = param_type.model_validate(value)
        elif param_type is Callable or get_origin(param_type) is Callable:
          kwargs[param_name] = config_agent_utils.resolve_fully_qualified_name(
              value
          )
        elif param_type in (list, set, dict):
          kwargs[param_name] = param_type(value)
        elif get_origin(param_type) is list:
          list_args = get_args(param_type)
          if issubclass(list_args[0], BaseModel):
            kwargs[param_name] = [
                list_args[0].model_validate(item) for item in value
            ]
          elif list_args[0] in (int, str, bool, float):
            kwargs[param_name] = value
          elif list_args[0] is Callable or get_origin(list_args[0]) is Callable:
            kwargs[param_name] = [
                config_agent_utils.resolve_fully_qualified_name(item)
                for item in value
            ]
          else:
            logger.warning(
                "Unsupported parsing for list argument: %s.", param_name
            )
        else:
          logger.warning("Unsupported parsing for argument: %s.", param_name)
    return cls(**kwargs)


def _find_tool_with_function_declarations(
    llm_request: LlmRequest,
) -> Optional[types.Tool]:
  # TODO: add individual tool with declaration and merge in google_llm.py
  if not llm_request.config or not llm_request.config.tools:
    return None

  return next(
      (
          tool
          for tool in llm_request.config.tools
          if isinstance(tool, types.Tool) and tool.function_declarations
      ),
      None,
  )
