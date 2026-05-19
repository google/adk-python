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

"""Tool for setting model response when using output_schema with other tools."""

from __future__ import annotations

import inspect
from typing import Any
from typing import Optional

from google.genai import types
from pydantic import TypeAdapter
from typing_extensions import override

from ..utils._schema_utils import get_list_inner_type
from ..utils._schema_utils import is_basemodel_schema
from ..utils._schema_utils import is_list_of_basemodel
from ..utils._schema_utils import SchemaType
from ._automatic_function_calling_util import build_function_declaration
from .base_tool import BaseTool
from .tool_context import ToolContext


class SetModelResponseTool(BaseTool):
  """Internal tool used for output schema workaround.

  This tool allows the model to set its final response when output_schema
  is configured alongside other tools. The model should use this tool to
  provide its final structured response instead of outputting text directly.
  """

  def __init__(self, output_schema: SchemaType):
    """Initialize the tool with the expected output schema.

    Args:
      output_schema: The output schema. Supports all types from SchemaUnion:
        - type[BaseModel]: A pydantic model class (e.g., MySchema)
        - list[type[BaseModel]]: A generic list type (e.g., list[MySchema])
        - list[primitive]: e.g., list[str], list[int]
        - dict: Raw dict schemas
        - Schema: Google's Schema type
    """
    self.output_schema = output_schema
    self._is_basemodel = is_basemodel_schema(output_schema)
    self._is_list_of_basemodel = is_list_of_basemodel(output_schema)

    # Create a function that matches the output schema
    def set_model_response() -> str:
      """Set your final response using the required output schema.

      Use this tool to provide your final structured answer instead
      of outputting text directly.
      """
      return 'Response set successfully.'

    # Add the schema fields as parameters to the function dynamically
    if self._is_basemodel:
      # For regular BaseModel, use the model's fields
      schema_fields = output_schema.model_fields
      params = []
      for field_name, field_info in schema_fields.items():
        param = inspect.Parameter(
            field_name,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=field_info.annotation,
        )
        params.append(param)
    elif self._is_list_of_basemodel:
      # For list[BaseModel], create a single 'items' parameter
      inner_type = get_list_inner_type(output_schema)
      params = [
          inspect.Parameter(
              'items',
              inspect.Parameter.KEYWORD_ONLY,
              annotation=list[inner_type],
          )
      ]
    else:
      # For other schema types (list[str], dict, etc.),
      # create a single parameter with the actual schema type
      params = [
          inspect.Parameter(
              'response',
              inspect.Parameter.KEYWORD_ONLY,
              annotation=output_schema,
          )
      ]

    # Create new signature with schema parameters
    new_sig = inspect.Signature(parameters=params)
    setattr(set_model_response, '__signature__', new_sig)

    self.func = set_model_response

    super().__init__(
        name=self.func.__name__,
        description=self.func.__doc__.strip() if self.func.__doc__ else '',
    )

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    """Gets the OpenAPI specification of this tool."""
    function_decl = types.FunctionDeclaration.model_validate(
        build_function_declaration(
            func=self.func,
            ignore_params=[],
            variant=self._api_variant,
        )
    )
    return function_decl

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Process the model's response and return the validated data.

    Args:
      args: The structured response data matching the output schema.
      tool_context: Tool execution context.

    Returns:
      The validated response. Type depends on the output_schema:
        - dict for BaseModel
        - list of dicts for list[BaseModel]
        - raw value for other schema types (list[str], dict, etc.)
    """
    if self._is_basemodel:
      # For regular BaseModel, validate directly
      validated_response = self.output_schema.model_validate(args)
      return validated_response.model_dump(exclude_none=True)
    elif self._is_list_of_basemodel:
      # For list[BaseModel], extract and validate the 'items' field
      items = args.get('items', [])
      type_adapter = TypeAdapter(self.output_schema)
      validated_response = type_adapter.validate_python(items)
      return [item.model_dump(exclude_none=True) for item in validated_response]
    else:
      # For other schema types (list[str], dict, etc.),
      # return the value directly without pydantic validation
      return args.get('response')
