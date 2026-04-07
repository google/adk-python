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

import collections.abc
import inspect
from types import FunctionType
import typing
from typing import Annotated
from typing import Any
from typing import Callable
from typing import Dict
from typing import get_args
from typing import get_origin
from typing import Optional
from typing import Union

from google.genai import types
import pydantic
from pydantic import BaseModel
from pydantic import create_model
from pydantic import fields as pydantic_fields
from pydantic.fields import FieldInfo

from . import _function_parameter_parse_util
from . import _function_tool_declarations
from ..features import FeatureName
from ..features import is_feature_enabled
from ..utils.variant_utils import GoogleLLMVariant

_py_type_2_schema_type = {
    "str": types.Type.STRING,
    "int": types.Type.INTEGER,
    "float": types.Type.NUMBER,
    "bool": types.Type.BOOLEAN,
    "string": types.Type.STRING,
    "integer": types.Type.INTEGER,
    "number": types.Type.NUMBER,
    "boolean": types.Type.BOOLEAN,
    "list": types.Type.ARRAY,
    "array": types.Type.ARRAY,
    "tuple": types.Type.ARRAY,
    "object": types.Type.OBJECT,
    "Dict": types.Type.OBJECT,
    "List": types.Type.ARRAY,
    "Tuple": types.Type.ARRAY,
    "Any": types.Type.TYPE_UNSPECIFIED,
}


def _extract_field_info_from_annotated(
    annotation: Any,
) -> Optional[FieldInfo]:
  """Extract pydantic FieldInfo from Annotated[T, Field(...)] if present.

  Args:
    annotation: The type annotation to inspect.

  Returns:
    The FieldInfo instance if found in Annotated metadata, None otherwise.
  """
  if get_origin(annotation) is Annotated:
    for metadata in get_args(annotation)[1:]:
      if isinstance(metadata, FieldInfo):
        return metadata
  return None


def _extract_base_type_from_annotated(annotation: Any) -> Any:
  """Extract the base type from Annotated[T, ...].

  Args:
    annotation: The type annotation to unwrap.

  Returns:
    The base type T if annotation is Annotated[T, ...], otherwise the original
    annotation.
  """
  if get_origin(annotation) is Annotated:
    return get_args(annotation)[0]
  return annotation


def _resolve_pydantic_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
  """Resolve $ref pointers in Pydantic JSON schema and inline nested objects.

  Pydantic generates JSON schemas with $ref pointers to $defs for nested
  BaseModel classes. This function resolves these references and inlines
  nested properties so that Field descriptions from nested models are
  directly accessible in the schema sent to the LLM.

  This is similar to the reference resolution in openapi_spec_parser.py but
  optimized for Pydantic v2 schema structure (handles allOf wrappers).

  Args:
    schema: Pydantic model_json_schema() output with $defs.

  Returns:
    Schema with all $ref resolved and nested properties inlined. The $defs
    section is removed as all definitions are now inlined.

  Example:
    Input:
      {
        "properties": {
          "user": {"allOf": [{"$ref": "#/$defs/Person"}], "description": "User"}
        },
        "$defs": {
          "Person": {"properties": {"name": {"description": "Name"}}}
        }
      }

    Output:
      {
        "properties": {
          "user": {
            "type": "object",
            "description": "User",
            "properties": {"name": {"description": "Name"}}
          }
        }
      }
  """
  import copy

  schema = copy.deepcopy(schema)
  defs = schema.get("$defs", {})

  def resolve_ref(ref_string: str) -> Optional[Dict[str, Any]]:
    """Resolve a $ref string like '#/$defs/Person'."""
    if not ref_string.startswith("#/$defs/"):
      return None
    def_name = ref_string.split("/")[-1]
    resolved: Optional[Dict[str, Any]] = defs.get(def_name)
    return resolved

  def resolve_property(
      prop_schema: Dict[str, Any],
      seen_refs: Optional[set[str]] = None,
  ) -> Dict[str, Any]:
    """Recursively resolve $ref in a property schema.

    Args:
      prop_schema: A property schema that may contain $ref or allOf with $ref.
      seen_refs: Set of already-visited $ref strings to prevent circular refs.

    Returns:
      Property schema with all $ref resolved and nested properties inlined.
    """
    if seen_refs is None:
      seen_refs = set()

    prop_schema = copy.deepcopy(prop_schema)

    # Handle allOf wrapper (Pydantic v2 pattern: {"allOf": [{"$ref": "..."}]})
    if "allOf" in prop_schema and len(prop_schema["allOf"]) == 1:
      ref_item = prop_schema["allOf"][0]
      if "$ref" in ref_item:
        ref_string = ref_item["$ref"]

        # Prevent circular references
        if ref_string in seen_refs:
          # Return schema without allOf to break the cycle
          return {k: v for k, v in prop_schema.items() if k != "allOf"}

        seen_refs_copy = seen_refs.copy()
        seen_refs_copy.add(ref_string)

        resolved = resolve_ref(ref_string)
        if resolved:
          resolved = copy.deepcopy(resolved)

          # Preserve parameter-level description (takes precedence over model docstring)
          param_description = prop_schema.get("description")

          # Recursively resolve nested properties within the resolved definition
          if "properties" in resolved:
            for nested_name, nested_schema in resolved["properties"].items():
              resolved["properties"][nested_name] = resolve_property(
                  nested_schema, seen_refs_copy
              )

          # If there was a parameter-level description, keep it
          # (e.g., "User info" instead of model's docstring "Person model")
          if param_description:
            resolved["description"] = param_description

          return resolved

    # Handle direct $ref (less common in Pydantic v2, but supported for completeness)
    elif "$ref" in prop_schema:
      ref_string = prop_schema["$ref"]
      if ref_string not in seen_refs:
        seen_refs_copy = seen_refs.copy()
        seen_refs_copy.add(ref_string)

        # Preserve parameter-level description (takes precedence over model docstring)
        param_description = prop_schema.get("description")

        resolved = resolve_ref(ref_string)
        if resolved:
          result = resolve_property(copy.deepcopy(resolved), seen_refs_copy)
          if param_description:
            result["description"] = param_description
          return result

    # Recursively resolve nested properties (for already-inlined objects)
    if "properties" in prop_schema:
      for nested_name in list(prop_schema["properties"].keys()):
        prop_schema["properties"][nested_name] = resolve_property(
            prop_schema["properties"][nested_name], seen_refs
        )

    # Handle arrays with items that might have refs
    if "items" in prop_schema:
      prop_schema["items"] = resolve_property(prop_schema["items"], seen_refs)

    return prop_schema

  # Resolve all top-level properties
  if "properties" in schema:
    for prop_name in list(schema["properties"].keys()):
      schema["properties"][prop_name] = resolve_property(
          schema["properties"][prop_name]
      )

  # Clean up $defs since all definitions are now inlined
  schema.pop("$defs", None)

  return schema


def _get_fields_dict(func: Callable) -> Dict:
  """Build a dictionary of field definitions for Pydantic model creation.

  This function extracts parameter information from a callable and creates
  field definitions compatible with Pydantic's create_model. It supports
  parameter descriptions via Annotated[T, Field(description=...)] syntax.

  Args:
    func: The callable to extract parameters from.

  Returns:
    A dictionary mapping parameter names to (type, FieldInfo) tuples.
  """
  param_signature = dict(inspect.signature(func).parameters)
  fields_dict = {}

  for name, param in param_signature.items():
    # We do not support *args or **kwargs
    if param.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_ONLY,
    ):
      continue

    annotation = (
        param.annotation if param.annotation != inspect.Parameter.empty else Any
    )

    # Extract FieldInfo from Annotated[T, Field(...)] if present
    field_info = _extract_field_info_from_annotated(annotation)

    # Extract the base type from Annotated[T, ...] for the model field
    base_type = _extract_base_type_from_annotated(annotation)

    # Determine the default value
    default = (
        param.default
        if param.default != inspect.Parameter.empty
        else pydantic_fields.PydanticUndefined
    )

    # Get description from FieldInfo if available
    description = field_info.description if field_info else None

    fields_dict[name] = (
        base_type,
        pydantic.Field(
            default=default,
            description=description,
        ),
    )

  return fields_dict


def _annotate_nullable_fields(schema: Dict):
  for _, property_schema in schema.get("properties", {}).items():
    # for Optional[T], the pydantic schema is:
    # {
    #   "type": "object",
    #   "properties": {
    #     "anyOf": [
    #       {
    #         "type": "null"
    #       },
    #       {
    #         "type": "T"
    #       }
    #     ]
    #   }
    # }
    for type_ in property_schema.get("anyOf", []):
      if type_.get("type") == "null":
        property_schema["nullable"] = True
        property_schema["anyOf"].remove(type_)
        break


def _annotate_required_fields(schema: Dict):
  required = [
      field_name
      for field_name, field_schema in schema.get("properties", {}).items()
      if not field_schema.get("nullable") and "default" not in field_schema
  ]
  schema["required"] = required


def _remove_any_of(schema: Dict):
  for _, property_schema in schema.get("properties", {}).items():
    union_types = property_schema.pop("anyOf", None)
    # Take the first non-null type.
    if union_types:
      for type_ in union_types:
        if type_.get("type") != "null":
          property_schema.update(type_)


def _remove_default(schema: Dict):
  for _, property_schema in schema.get("properties", {}).items():
    property_schema.pop("default", None)


def _remove_nullable(schema: Dict):
  for _, property_schema in schema.get("properties", {}).items():
    property_schema.pop("nullable", None)


def _remove_title(schema: Dict):
  for _, property_schema in schema.get("properties", {}).items():
    property_schema.pop("title", None)


def _get_pydantic_schema(func: Callable) -> Dict[str, Any]:
  from ..utils.context_utils import find_context_parameter

  fields_dict = _get_fields_dict(func)
  # Remove context parameter (detected by type or fallback to 'tool_context' name)
  context_param = find_context_parameter(func) or "tool_context"
  if context_param in fields_dict.keys():
    fields_dict.pop(context_param)

  # Capture per-parameter descriptions before schema generation, because
  # Pydantic may replace them with model docstrings for nested BaseModel types.
  param_descriptions: Dict[str, str] = {}
  for name, (_, field_info) in fields_dict.items():
    if field_info.description:
      param_descriptions[name] = field_info.description

  schema = pydantic.create_model(
      func.__name__, **fields_dict
  ).model_json_schema()

  # Resolve $ref for nested Pydantic models to inline Field descriptions
  schema = _resolve_pydantic_refs(schema)

  # Re-apply per-parameter descriptions that may have been lost during
  # schema generation (Pydantic uses model docstrings for nested models).
  for name, description in param_descriptions.items():
    if name in schema.get("properties", {}):
      schema["properties"][name]["description"] = description

  return schema


def _process_pydantic_schema(vertexai: bool, schema: Dict) -> Dict:
  _annotate_nullable_fields(schema)
  _annotate_required_fields(schema)
  if not vertexai:
    _remove_any_of(schema)
    _remove_default(schema)
    _remove_nullable(schema)
    _remove_title(schema)
  return schema


def _map_pydantic_type_to_property_schema(property_schema: Dict):
  if "type" in property_schema:
    property_schema["type"] = _py_type_2_schema_type.get(
        property_schema["type"], "TYPE_UNSPECIFIED"
    )
    if property_schema["type"] == "ARRAY":
      _map_pydantic_type_to_property_schema(property_schema["items"])
  for type_ in property_schema.get("anyOf", []):
    if "type" in type_:
      type_["type"] = _py_type_2_schema_type.get(
          type_["type"], "TYPE_UNSPECIFIED"
      )
      # TODO: To investigate. Unclear why a Type is needed with 'anyOf' to
      # avoid google.genai.errors.ClientError: 400 INVALID_ARGUMENT.
      property_schema["type"] = type_["type"]


def _map_pydantic_type_to_schema_type(schema: Dict):
  for _, property_schema in schema.get("properties", {}).items():
    _map_pydantic_type_to_property_schema(property_schema)


def _get_return_type(func: Callable) -> Any:
  return _py_type_2_schema_type.get(
      inspect.signature(func).return_annotation.__name__,
      inspect.signature(func).return_annotation.__name__,
  )


def build_function_declaration(
    func: Union[Callable, BaseModel],
    ignore_params: Optional[list[str]] = None,
    variant: GoogleLLMVariant = GoogleLLMVariant.GEMINI_API,
) -> types.FunctionDeclaration:
  # ========== Pydantic-based function tool declaration (new feature) ==========
  if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
    declaration = (
        _function_tool_declarations.build_function_declaration_with_json_schema(
            func, ignore_params=ignore_params
        )
    )
    # Add response schema only for VERTEX_AI
    # TODO(b/421991354): Remove this check once the bug is fixed.
    if variant != GoogleLLMVariant.VERTEX_AI:
      declaration.response_json_schema = None
    return declaration

  # ========== ADK defined function tool declaration (old behavior) ==========
  signature = inspect.signature(func)
  should_update_signature = False
  new_func = None
  if not ignore_params:
    ignore_params = []
  for name, _ in signature.parameters.items():
    if name in ignore_params:
      should_update_signature = True
      break
  if should_update_signature:
    new_params = [
        param
        for name, param in signature.parameters.items()
        if name not in ignore_params
    ]
    if isinstance(func, type):
      fields = {
          name: (param.annotation, param.default)
          for name, param in signature.parameters.items()
          if name not in ignore_params
      }
      new_func = create_model(func.__name__, **fields)
    else:
      new_sig = signature.replace(parameters=new_params)
      new_func = FunctionType(
          func.__code__,
          func.__globals__,
          func.__name__,
          func.__defaults__,
          func.__closure__,
      )
      new_func.__signature__ = new_sig
      new_func.__doc__ = func.__doc__
      new_func.__annotations__ = func.__annotations__

  return (
      from_function_with_options(func, variant)
      if not should_update_signature
      else from_function_with_options(new_func, variant)
  )


def build_function_declaration_for_langchain(
    vertexai: bool, name, description, func, param_pydantic_schema
) -> types.FunctionDeclaration:
  param_pydantic_schema = _process_pydantic_schema(
      vertexai, {"properties": param_pydantic_schema}
  )["properties"]
  param_copy = param_pydantic_schema.copy()
  required_fields = param_copy.pop("required", [])
  before_param_pydantic_schema = {
      "properties": param_copy,
      "required": required_fields,
  }
  return build_function_declaration_util(
      vertexai, name, description, func, before_param_pydantic_schema
  )


def build_function_declaration_for_params_for_crewai(
    vertexai: bool, name, description, func, param_pydantic_schema
) -> types.FunctionDeclaration:
  param_pydantic_schema = _process_pydantic_schema(
      vertexai, param_pydantic_schema
  )
  param_copy = param_pydantic_schema.copy()
  return build_function_declaration_util(
      vertexai, name, description, func, param_copy
  )


def build_function_declaration_util(
    vertexai: bool, name, description, func, before_param_pydantic_schema
) -> types.FunctionDeclaration:
  _map_pydantic_type_to_schema_type(before_param_pydantic_schema)
  properties = before_param_pydantic_schema.get("properties", {})
  function_declaration = types.FunctionDeclaration(
      parameters=types.Schema(
          type="OBJECT",
          properties=properties,
      )
      if properties
      else None,
      description=description,
      name=name,
  )
  if vertexai and isinstance(func, Callable):
    return_pydantic_schema = _get_return_type(func)
    function_declaration.response = types.Schema(
        type=return_pydantic_schema,
    )
  return function_declaration


def from_function_with_options(
    func: Callable,
    variant: GoogleLLMVariant = GoogleLLMVariant.GEMINI_API,
) -> "types.FunctionDeclaration":

  parameters_properties = {}
  parameters_json_schema = {}
  try:
    annotation_under_future = typing.get_type_hints(func)
  except TypeError:
    # This can happen if func is a mock object
    annotation_under_future = {}
  # Collect Annotated field descriptions to apply after schema generation.
  annotated_descriptions: Dict[str, str] = {}
  try:
    for name, param in inspect.signature(func).parameters.items():
      if param.kind in (
          inspect.Parameter.POSITIONAL_OR_KEYWORD,
          inspect.Parameter.KEYWORD_ONLY,
          inspect.Parameter.POSITIONAL_ONLY,
      ):
        param = _function_parameter_parse_util._handle_params_as_deferred_annotations(
            param, annotation_under_future, name
        )

        # Unwrap Annotated[T, Field(...)] so the parser sees the base type.
        field_info = _extract_field_info_from_annotated(param.annotation)
        if field_info and field_info.description:
          annotated_descriptions[name] = field_info.description
        base_type = _extract_base_type_from_annotated(param.annotation)
        if base_type is not param.annotation:
          param = param.replace(annotation=base_type)

        schema = _function_parameter_parse_util._parse_schema_from_parameter(
            variant, param, func.__name__
        )
        if name in annotated_descriptions:
          schema.description = annotated_descriptions[name]
        parameters_properties[name] = schema
  except ValueError:
    # If the function has complex parameter types that fail in _parse_schema_from_parameter,
    # we try to generate a json schema for the parameter using pydantic.TypeAdapter.
    parameters_properties = {}
    for name, param in inspect.signature(func).parameters.items():
      if param.kind in (
          inspect.Parameter.POSITIONAL_OR_KEYWORD,
          inspect.Parameter.KEYWORD_ONLY,
          inspect.Parameter.POSITIONAL_ONLY,
      ):
        try:
          if param.annotation == inspect.Parameter.empty:
            param = param.replace(annotation=Any)

          param = _function_parameter_parse_util._handle_params_as_deferred_annotations(
              param, annotation_under_future, name
          )

          # Unwrap Annotated[T, Field(...)] for the fallback path too.
          field_info = _extract_field_info_from_annotated(param.annotation)
          if field_info and field_info.description:
            annotated_descriptions[name] = field_info.description
          base_type = _extract_base_type_from_annotated(param.annotation)
          if base_type is not param.annotation:
            param = param.replace(annotation=base_type)

          _function_parameter_parse_util._raise_for_invalid_enum_value(param)

          json_schema_dict = _function_parameter_parse_util._generate_json_schema_for_parameter(
              param
          )

          schema_obj = types.Schema.model_validate(json_schema_dict)
          if name in annotated_descriptions:
            schema_obj.description = annotated_descriptions[name]
          parameters_json_schema[name] = schema_obj
        except Exception as e:
          _function_parameter_parse_util._raise_for_unsupported_param(
              param, func.__name__, e
          )

  declaration = types.FunctionDeclaration(
      name=func.__name__,
      description=func.__doc__,
  )
  if parameters_properties:
    declaration.parameters = types.Schema(
        type="OBJECT",
        properties=parameters_properties,
    )
    declaration.parameters.required = (
        _function_parameter_parse_util._get_required_fields(
            declaration.parameters
        )
    )
  elif parameters_json_schema:
    declaration.parameters = types.Schema(
        type="OBJECT",
        properties=parameters_json_schema,
    )

  if variant == GoogleLLMVariant.GEMINI_API:
    return declaration

  return_annotation = inspect.signature(func).return_annotation

  # Handle AsyncGenerator and Generator return types (streaming tools)
  # AsyncGenerator[YieldType, SendType] -> use YieldType as response schema
  # Generator[YieldType, SendType, ReturnType] -> use YieldType as response schema
  origin = get_origin(return_annotation)
  if origin is not None and (
      origin is collections.abc.AsyncGenerator
      or origin is collections.abc.Generator
  ):
    type_args = get_args(return_annotation)
    if type_args:
      # First type argument is the yield type
      yield_type = type_args[0]
      return_annotation = yield_type

  # Handle functions with no return annotation
  if return_annotation is inspect._empty:
    # Functions with no return annotation can return any type
    return_value = inspect.Parameter(
        "return_value",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=typing.Any,
    )
    declaration.response = (
        _function_parameter_parse_util._parse_schema_from_parameter(
            variant,
            return_value,
            func.__name__,
        )
    )
    return declaration

  # Handle functions that explicitly return None
  if (
      return_annotation is None
      or return_annotation is type(None)
      or (isinstance(return_annotation, str) and return_annotation == "None")
  ):
    # Create a response schema for None/null return
    return_value = inspect.Parameter(
        "return_value",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=None,
    )
    declaration.response = (
        _function_parameter_parse_util._parse_schema_from_parameter(
            variant,
            return_value,
            func.__name__,
        )
    )
    return declaration

  return_value = inspect.Parameter(
      "return_value",
      inspect.Parameter.POSITIONAL_OR_KEYWORD,
      annotation=return_annotation,
  )
  if isinstance(return_value.annotation, str):
    return_value = return_value.replace(
        annotation=typing.get_type_hints(func)["return"]
    )

  response_schema: Optional[types.Schema] = None
  response_json_schema: Optional[Union[Dict[str, Any], types.Schema]] = None
  try:
    response_schema = (
        _function_parameter_parse_util._parse_schema_from_parameter(
            variant,
            return_value,
            func.__name__,
        )
    )
  except ValueError:
    try:
      response_json_schema = (
          _function_parameter_parse_util._generate_json_schema_for_parameter(
              return_value
          )
      )
      response_json_schema = types.Schema.model_validate(response_json_schema)
    except Exception as e:
      _function_parameter_parse_util._raise_for_unsupported_param(
          return_value, func.__name__, e
      )
  if response_schema:
    declaration.response = response_schema
  elif response_json_schema:
    declaration.response = response_json_schema
  return declaration
