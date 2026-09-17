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

import keyword
from typing import Any
from typing import Dict
from typing import List

from fastapi.openapi.models import Reference
from fastapi.openapi.models import Response
from fastapi.openapi.models import Schema
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_serializer

from ..._gemini_schema_util import _to_snake_case


def _schema_from_openapi(value: object, *, context: str) -> Schema:
  """Normalizes a schema-bearing OpenAPI field to a concrete Schema.

  OpenAPI models permit unresolved references and boolean JSON schemas. ADK's
  generated Python signature needs a concrete schema, so those cases are
  handled once at this boundary instead of leaking unions through the parser.
  """
  if value is None or value is True:
    return Schema()
  if value is False:
    raise ValueError(f'{context} uses an unsatisfiable false schema')
  if isinstance(value, Schema):
    return value
  if isinstance(value, Reference):
    raise ValueError(f'{context} contains unresolved reference {value.ref!r}')
  if isinstance(value, str):
    return Schema.model_validate_json(value)
  if isinstance(value, dict):
    return Schema.model_validate(value)
  raise TypeError(f'{context} must be an OpenAPI schema, got {type(value)!r}')


def rename_python_keywords(s: str, prefix: str = 'param_') -> str:
  """Renames Python keywords by adding a prefix.

  Example:
  ```
  rename_python_keywords('if') -> 'param_if'
  rename_python_keywords('for') -> 'param_for'
  ```

  Args:
      s: The input string.
      prefix: The prefix to add to the keyword.

  Returns:
      The renamed string.
  """
  if keyword.iskeyword(s):
    return prefix + s
  return s


class ApiParameter(BaseModel):
  """Data class representing a function parameter."""

  original_name: str
  param_location: str
  param_schema: str | Schema
  # Kept optional: callers pass None, and model_post_init normalizes it to ''.
  description: str | None = ''
  py_name: str | None = ''
  # Both are derived in model_post_init; the None defaults are never observed.
  type_value: object = Field(default=None, init_var=False)
  type_hint: str | None = Field(default=None, init_var=False)
  required: bool = False

  def model_post_init(self, _: Any) -> None:
    if not self.py_name:
      inferred_name = rename_python_keywords(_to_snake_case(self.original_name))
      self.py_name = inferred_name or self._default_py_name()
    if isinstance(self.param_schema, str):
      self.param_schema = Schema.model_validate_json(self.param_schema)

    schema = self.param_schema
    self.description = self.description or schema.description or ''
    self.type_value = TypeHintHelper.get_type_value(schema)
    self.type_hint = TypeHintHelper.get_type_hint(schema)

  @property
  def _openapi_schema(self) -> Schema:
    """Returns the normalized schema established during model validation."""
    if not isinstance(self.param_schema, Schema):
      raise RuntimeError('ApiParameter schema was not normalized')
    return self.param_schema

  def _default_py_name(self) -> str:
    location_defaults = {
        'body': 'body',
        'query': 'query_param',
        'path': 'path_param',
        'header': 'header_param',
        'cookie': 'cookie_param',
    }
    return location_defaults.get(self.param_location or '', 'value')

  @model_serializer
  def _serialize(self) -> dict[str, object]:
    return {
        'original_name': self.original_name,
        'param_location': self.param_location,
        'param_schema': self.param_schema,
        'description': self.description,
        'py_name': self.py_name,
    }

  def __str__(self) -> str:
    return f'{self.py_name}: {self.type_hint}'

  def to_arg_string(self) -> str:
    """Converts the parameter to an argument string for function call."""
    return f'{self.py_name}={self.py_name}'

  def to_dict_property(self) -> str:
    """Converts the parameter to a key:value string for dict property."""
    return f'"{self.py_name}": {self.py_name}'

  def to_pydoc_string(self) -> str:
    """Converts the parameter to a PyDoc parameter docstr."""
    return PydocHelper.generate_param_doc(self)


class TypeHintHelper:
  """Helper class for generating type hints."""

  @staticmethod
  def _get_schema_type(schema: Schema | bool | None) -> str | None:
    if schema is None or isinstance(schema, bool):
      return None
    schema_type = schema.type
    if isinstance(schema_type, list):
      non_null_types = [value for value in schema_type if value != 'null']
      return non_null_types[0] if len(non_null_types) == 1 else None
    return schema_type

  # `object` is the true return type, but this one is public: narrowing it
  # would break callers that feed the result straight into an annotation.
  @staticmethod
  def get_type_value(schema: Schema | bool) -> Any:
    """Generates the Python type value for a given parameter."""
    if isinstance(schema, bool):
      return Any
    param_type = TypeHintHelper._get_schema_type(schema)

    if param_type == 'integer':
      return int
    elif param_type == 'number':
      return float
    elif param_type == 'boolean':
      return bool
    elif param_type == 'string':
      return str
    elif param_type == 'array':
      items_type = TypeHintHelper._get_schema_type(schema.items)
      array_type_map: dict[str, object] = {
          'integer': List[int],
          'number': List[float],
          'boolean': List[bool],
          'string': List[str],
          'object': List[Dict[str, Any]],
          'array': List[List[Any]],
      }
      return array_type_map.get(items_type or '', List[Any])
    elif param_type == 'object':
      return Dict[str, Any]
    else:
      return Any

  @staticmethod
  def get_type_hint(schema: Schema | bool) -> str:
    """Generates the Python type in string for a given parameter."""
    if isinstance(schema, bool):
      return 'Any'
    param_type = TypeHintHelper._get_schema_type(schema)

    if param_type == 'integer':
      return 'int'
    elif param_type == 'number':
      return 'float'
    elif param_type == 'boolean':
      return 'bool'
    elif param_type == 'string':
      return 'str'
    elif param_type == 'array':
      items_type = TypeHintHelper._get_schema_type(schema.items)

      if items_type == 'object':
        return 'List[Dict[str, Any]]'
      else:
        type_map = {
            'integer': 'int',
            'number': 'float',
            'boolean': 'bool',
            'string': 'str',
        }
        return f"List[{type_map.get(items_type or '', 'Any')}]"
    elif param_type == 'object':
      return 'Dict[str, Any]'
    else:
      return 'Any'


class PydocHelper:
  """Helper class for generating PyDoc strings."""

  @staticmethod
  def generate_param_doc(
      param: ApiParameter,
  ) -> str:
    """Generates a parameter documentation string.

    Args:
      param: ApiParameter - The parameter to generate the documentation for.

    Returns:
      str: The generated parameter Python documentation string.
    """
    description = param.description.strip() if param.description else ''
    param_doc = f'{param.py_name} ({param.type_hint}): {description}'

    schema = param._openapi_schema
    if schema.type == 'object':
      properties = schema.properties
      if properties:
        param_doc += ' Object properties:\n'
        for prop_name, prop_details in properties.items():
          prop_desc = (
              prop_details.description
              if isinstance(prop_details, Schema) and prop_details.description
              else ''
          )
          prop_type = TypeHintHelper.get_type_hint(prop_details)
          param_doc += f'       {prop_name} ({prop_type}): {prop_desc}\n'

    return param_doc

  # The public annotation stays `Dict[str, Response]`; the body still guards
  # the looser values OpenAPI actually permits here.
  @staticmethod
  def generate_return_doc(responses: Dict[str, Response]) -> str:
    """Generates a return value documentation string.

    Args:
      responses: Dict[str, TypedDict[Response]] - Response in an OpenAPI
        Operation

    Returns:
      str: The generated return value Python documentation string.
    """
    return_doc = ''

    # Only consider 2xx responses for return type hinting.
    # Returns the 2xx response with the smallest status code number and with
    # content defined. Non-numeric OpenAPI response keys (e.g. 'default' or
    # range codes like '2XX') are valid and sorted after numeric status codes.
    qualified_responses = [
        (status, response)
        for status, response in responses.items()
        if status.startswith('2')
        and isinstance(response, Response)
        and response.content
    ]
    qualified_response = min(
        qualified_responses,
        key=lambda item: (
            0 if item[0].isdigit() else 1,
            int(item[0]) if item[0].isdigit() else item[0],
        ),
        default=None,
    )
    if not qualified_response:
      return ''
    response_details = qualified_response[1]

    description = (response_details.description or '').strip()
    content = response_details.content or {}

    # Prefer application/json when multiple content types are present;
    # otherwise use the first available content type.
    schema_details = content.get('application/json')
    if schema_details is None:
      schema_details = next(iter(content.values()), None)
    if schema_details is None:
      return return_doc

    schema = _schema_from_openapi(
        schema_details.schema_, context='response body'
    )

    # Use a dummy Parameter object for return type hinting.
    dummy_param = ApiParameter(
        original_name='', param_location='', param_schema=schema
    )
    return_doc = f'Returns ({dummy_param.type_hint}): {description}'

    response_type = schema.type or 'Any'
    if response_type == 'object':
      properties = schema.properties
      if properties:
        return_doc += ' Object properties:\n'
        for prop_name, prop_details in properties.items():
          prop_desc = (
              prop_details.description
              if isinstance(prop_details, Schema) and prop_details.description
              else ''
          )
          prop_type = TypeHintHelper.get_type_hint(prop_details)
          return_doc += f'        {prop_name} ({prop_type}): {prop_desc}\n'

    return return_doc
