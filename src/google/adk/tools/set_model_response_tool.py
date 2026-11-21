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

import copy
import inspect
import logging
import types as typing_types
from typing import Any
from typing import cast
from typing import Final
from typing import get_args
from typing import get_origin
from typing import Optional
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import TypeAdapter
from pydantic import ValidationError
from pydantic.fields import FieldInfo
from typing_extensions import override

from ..utils._schema_utils import get_list_inner_type
from ..utils._schema_utils import is_basemodel_schema
from ..utils._schema_utils import lowercase_schema_types
from ..utils._schema_utils import SchemaType
from ._automatic_function_calling_util import build_function_declaration
from ._gemini_schema_util import _to_gemini_schema
from .base_tool import BaseTool
from .tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)


def _merge_json_schema_descriptions(
    target: dict[str, Any], source: dict[str, Any]
) -> None:
  """Copies ``description`` values from ``source`` onto ``target`` in place.

  Walks ``properties`` / ``items`` so nested object and list schemas keep the
  Field(description=...) metadata from the original Pydantic output schema.
  """
  source_props = source.get('properties')
  target_props = target.get('properties')
  if isinstance(source_props, dict) and isinstance(target_props, dict):
    for name, source_prop in source_props.items():
      if name not in target_props or not isinstance(source_prop, dict):
        continue
      target_prop = target_props[name]
      if not isinstance(target_prop, dict):
        continue
      description = source_prop.get('description')
      if isinstance(description, str) and description:
        target_prop['description'] = description
      _merge_json_schema_descriptions(target_prop, source_prop)

  source_items = source.get('items')
  target_items = target.get('items')
  if isinstance(source_items, dict) and isinstance(target_items, dict):
    description = source_items.get('description')
    if isinstance(description, str) and description:
      target_items['description'] = description
    _merge_json_schema_descriptions(target_items, source_items)


def _apply_descriptions_to_schema_properties(
    properties: dict[str, types.Schema] | None,
    model_fields: dict[str, FieldInfo],
) -> None:
  """Sets Schema.description from Pydantic FieldInfo.description when present."""
  if not properties:
    return
  for name, field_info in model_fields.items():
    prop = properties.get(name)
    if prop is None:
      continue
    if field_info.description:
      prop.description = field_info.description
    _apply_nested_descriptions(prop, field_info.annotation)


def _apply_nested_descriptions(prop: types.Schema, annotation: Any) -> None:
  """Recursively applies descriptions to nested BaseModel properties."""
  if isinstance(annotation, type) and issubclass(annotation, BaseModel):
    if prop.properties:
      _apply_descriptions_to_schema_properties(
          prop.properties, annotation.model_fields
      )
  elif get_origin(annotation) is list:
    args = get_args(annotation)
    if (
        args
        and isinstance(args[0], type)
        and issubclass(args[0], BaseModel)
        and prop.items
        and prop.items.properties
    ):
      _apply_descriptions_to_schema_properties(
          prop.items.properties, args[0].model_fields
      )
  elif get_origin(annotation) in (Union, typing_types.UnionType):
    for arg in get_args(annotation):
      if arg is not type(None):
        _apply_nested_descriptions(prop, arg)


def _point_refs_at_json_schema_defs(json_schema: dict[str, Any]) -> None:
  """Rewrites ``#/defs/...`` refs as ``#/$defs/...``, in place.

  A types.Schema keeps its definitions in ``defs``, which JSON Schema calls
  ``$defs``. Only ``properties``, ``items`` and ``anyOf`` are followed,
  because those are the subschemas that ``Schema.json_schema`` converts.
  """
  ref = json_schema.get('$ref')
  if isinstance(ref, str) and ref.startswith('#/defs/'):
    json_schema['$ref'] = '#/$defs/' + ref.removeprefix('#/defs/')
  subschemas = [
      *json_schema.get('properties', {}).values(),
      *json_schema.get('anyOf', []),
  ]
  if 'items' in json_schema:
    subschemas.append(json_schema['items'])
  for subschema in subschemas:
    _point_refs_at_json_schema_defs(subschema)


def _gemini_schema_to_json_schema(schema: types.Schema) -> dict[str, Any]:
  """Returns the JSON Schema form of a types.Schema.

  ``Schema.json_schema`` converts ``properties``, ``items`` and ``any_of``
  but copies ``defs`` and ``additional_properties`` unconverted, so each
  definition is converted here, type names are lowercased, and refs into
  ``defs`` are pointed at ``$defs``.
  """
  json_schema = schema.model_copy(update={'defs': None}).json_schema.model_dump(
      mode='json', exclude_none=True, by_alias=True
  )
  lowercase_schema_types(json_schema)
  _point_refs_at_json_schema_defs(json_schema)
  if schema.defs:
    json_schema['$defs'] = {
        name: _gemini_schema_to_json_schema(definition)
        for name, definition in schema.defs.items()
    }
  return json_schema


def _response_json_schema(
    output_schema: dict[str, Any] | types.Schema,
) -> dict[str, Any]:
  """Returns a JSON Schema copy of a dict or types.Schema output schema."""
  if isinstance(output_schema, types.Schema):
    return _gemini_schema_to_json_schema(output_schema)
  json_schema = copy.deepcopy(output_schema)
  # A dict may spell types in Gemini's uppercase dialect, such as `OBJECT`.
  lowercase_schema_types(json_schema)
  return json_schema


def _response_gemini_schema(
    output_schema: dict[str, Any] | types.Schema,
) -> types.Schema:
  """Returns a types.Schema copy of a dict or types.Schema output schema."""
  if isinstance(output_schema, types.Schema):
    return output_schema.model_copy(deep=True)
  return _to_gemini_schema(_response_json_schema(output_schema))


_SET_MODEL_RESPONSE_TOOL_NAME: Final[str] = 'set_model_response'


class SetModelResponseTool(BaseTool):
  """Internal tool used for output schema workaround.

  This tool allows the model to set its final response when output_schema
  is configured alongside other tools. The model should use this tool to
  provide its final structured response instead of outputting text directly.
  """

  NAME: Final[str] = _SET_MODEL_RESPONSE_TOOL_NAME

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
    # A dict or types.Schema cannot be a parameter annotation, so `response`
    # is annotated as a plain dict below. Keep the schema to declare it.
    self._response_schema: dict[str, Any] | types.Schema | None = (
        output_schema
        if isinstance(output_schema, (dict, types.Schema))
        else None
    )
    # Convert types.Schema instance to raw dict to avoid unhashable crash
    if isinstance(output_schema, types.Schema):
      output_schema = output_schema.model_dump(exclude_none=True)

    self.output_schema = output_schema
    self._model_type: Optional[type[BaseModel]] = (
        cast('type[BaseModel]', output_schema)
        if is_basemodel_schema(output_schema)
        else None
    )
    self._list_model_type = get_list_inner_type(output_schema)
    self._is_basemodel = self._model_type is not None
    self._is_list_of_basemodel = self._list_model_type is not None

    # Create a function that matches the output schema
    def set_model_response() -> str:
      """Set your final response using the required output schema.

      Use this tool to provide your final structured answer instead
      of outputting text directly.
      """
      return 'Response set successfully.'

    # Add the schema fields as parameters to the function dynamically
    if self._model_type is not None:
      # For regular BaseModel, use the model's fields
      schema_fields = self._model_type.model_fields
      params = []
      for field_name, field_info in schema_fields.items():
        # Carry the field's default across. Without it every parameter looks
        # required, so the model is told it must supply fields the caller
        # declared optional.
        param = inspect.Parameter(
            field_name,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=field_info.annotation,
            default=(
                inspect.Parameter.empty
                if field_info.is_required()
                else field_info.get_default(call_default_factory=True)
            ),
        )
        params.append(param)
    elif self._list_model_type is not None:
      # For list[BaseModel], create a single 'items' parameter
      params = [
          inspect.Parameter(
              'items',
              inspect.Parameter.KEYWORD_ONLY,
              annotation=typing_types.GenericAlias(list, self._list_model_type),
          )
      ]
    elif isinstance(output_schema, dict):
      # For raw dict schemas (e.g. {"type": "object", "properties": {...}}),
      # use the `dict` type itself as the annotation rather than the dict
      # instance. Passing the instance would later trigger
      # `annotation in _py_builtin_type_to_schema_type.keys()` inside
      # `_function_parameter_parse_util`, which calls `__hash__` on the
      # annotation and raises `TypeError: unhashable type: 'dict'`.
      params = [
          inspect.Parameter(
              'response',
              inspect.Parameter.KEYWORD_ONLY,
              annotation=dict,
          )
      ]
    else:
      # For other schema types (list[str], dict[str, int], etc.),
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
    self.func.__name__ = self.NAME

    super().__init__(
        name=self.NAME,
        description=self.func.__doc__.strip() if self.func.__doc__ else '',
    )

  def _preserve_output_schema_field_descriptions(
      self, function_decl: types.FunctionDeclaration
  ) -> None:
    """Restores Field(description=...) lost during function-declaration build.

    ``build_function_declaration`` rebuilds parameters from ``inspect.Parameter``
    objects, which cannot carry Pydantic field descriptions. Re-apply them from
    the original ``output_schema`` so the model still sees the semantic hints.
    """
    if isinstance(self.output_schema, type) and issubclass(
        self.output_schema, BaseModel
    ):
      source_schema = self.output_schema.model_json_schema()
      if function_decl.parameters_json_schema is not None:
        _merge_json_schema_descriptions(
            function_decl.parameters_json_schema, source_schema
        )
      elif function_decl.parameters is not None:
        _apply_descriptions_to_schema_properties(
            function_decl.parameters.properties,
            self.output_schema.model_fields,
        )
      return

    if self._is_list_of_basemodel:
      inner_type = get_list_inner_type(self.output_schema)
      if inner_type is None:
        return
      if (
          function_decl.parameters is not None
          and function_decl.parameters.properties
          and 'items' in function_decl.parameters.properties
      ):
        items_schema = function_decl.parameters.properties['items']
        if items_schema.items is not None:
          _apply_descriptions_to_schema_properties(
              items_schema.items.properties, inner_type.model_fields
          )

  def _declare_response_schema(
      self, function_decl: types.FunctionDeclaration
  ) -> None:
    """Declares a dict or types.Schema output schema as ``response``.

    ``response`` is annotated as a plain ``dict``, so
    ``build_function_declaration`` declares it as an object that takes any
    properties. Put the output schema in its place, in the form the builder
    used: JSON Schema, or ``types.Schema`` when JSON_SCHEMA_FOR_FUNC_DECL is
    off.
    """
    if self._response_schema is None:
      return
    if function_decl.parameters_json_schema is not None:
      response_schema = _response_json_schema(self._response_schema)
      # Nested under `response`, the schema is no longer the document root.
      # `$ref` pointers such as `#/$defs/X` resolve from the root, so the
      # definitions move up there. `$schema` may only appear at a root, and a
      # nested `$id` would make the pointers resolve against the nested schema
      # instead, so both are dropped.
      response_schema.pop('$schema', None)
      response_schema.pop('$id', None)
      parameters = function_decl.parameters_json_schema
      for defs_key in ('$defs', 'definitions'):
        if defs_key in response_schema:
          parameters[defs_key] = response_schema.pop(defs_key)
      parameters['properties']['response'] = response_schema
    elif (
        function_decl.parameters is not None
        and function_decl.parameters.properties
    ):
      try:
        gemini_schema = _response_gemini_schema(self._response_schema)
      except (TypeError, ValueError) as e:
        # The converter rejects schemas that types.Schema cannot express, such
        # as draft-07 tuple `items`. Keep `response` as any object then.
        logger.warning(
            'Could not convert the output schema to a types.Schema, so'
            ' set_model_response declares it as any object: %s',
            e,
        )
        return
      # `defs` is only allowed at the root of a types.Schema, and `ref`
      # resolves against the root, so the definitions move up there.
      if gemini_schema.defs:
        function_decl.parameters.defs = gemini_schema.defs
        gemini_schema.defs = None
      function_decl.parameters.properties['response'] = gemini_schema

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
    self._preserve_output_schema_field_descriptions(function_decl)
    self._declare_response_schema(function_decl)
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
      The validated response, or validation feedback for the model to retry.
      Type depends on the output_schema:
        - dict for BaseModel
        - list of dicts for list[BaseModel]
        - raw value for other schema types (list[str], dict, etc.)
        - dict with an error message when Pydantic validation fails
    """
    result: object
    try:
      if self._model_type is not None:
        # For regular BaseModel, validate directly. The declaration lists
        # field names, so accept them even when the fields have aliases.
        validated_model = self._model_type.model_validate(args, by_name=True)
        result = validated_model.model_dump(exclude_none=True)
      elif self._list_model_type is not None:
        # For list[BaseModel], extract and validate the 'items' field
        items = args.get('items', [])
        type_adapter: TypeAdapter[list[BaseModel]] = TypeAdapter(
            self.output_schema
        )
        validated_items = type_adapter.validate_python(items)
        result = [
            item.model_dump(exclude_none=True) for item in validated_items
        ]
      else:
        # For other schema types (list[str], dict, etc.),
        # return the value directly without pydantic validation
        result = args.get('response')
    except ValidationError as e:
      return {
          'error': (
              f'Validation Error found:\n{e}\n'
              'Recall the set_model_response function correctly, fix the'
              ' errors, and call it again with all required fields using the'
              ' correct types.'
          )
      }

    tool_context.actions.set_model_response = result
    return result
