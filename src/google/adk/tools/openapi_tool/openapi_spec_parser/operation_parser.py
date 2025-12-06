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

import inspect
from textwrap import dedent
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from fastapi.encoders import jsonable_encoder
from fastapi.openapi.models import Operation
from fastapi.openapi.models import Parameter
from fastapi.openapi.models import Schema

from ..._gemini_schema_util import _to_snake_case
from ..common.common import ApiParameter
from ..common.common import PydocHelper


class OperationParser:
  """Generates parameters for Python functions from an OpenAPI operation.

  This class processes an OpenApiOperation object and provides helper methods
  to extract information needed to generate Python function declarations,
  docstrings, signatures, and JSON schemas.  It handles parameter processing,
  name deduplication, and type hint generation.
  """

  def __init__(
      self, operation: Union[Operation, Dict[str, Any], str], should_parse=True
  ):
    """Initializes the OperationParser with an OpenApiOperation.

    Args:
        operation: The OpenApiOperation object or a dictionary to process.
        should_parse: Whether to parse the operation during initialization.
    """
    if isinstance(operation, dict):
      self._operation = Operation.model_validate(operation)
    elif isinstance(operation, str):
      self._operation = Operation.model_validate_json(operation)
    else:
      self._operation = operation

    self._params: List[ApiParameter] = []
    self._return_value: Optional[ApiParameter] = None
    if should_parse:
      self._process_operation_parameters()
      self._process_request_body()
      self._process_return_value()
      self._dedupe_param_names()

  @classmethod
  def load(
      cls,
      operation: Union[Operation, Dict[str, Any]],
      params: List[ApiParameter],
      return_value: Optional[ApiParameter] = None,
  ) -> 'OperationParser':
    parser = cls(operation, should_parse=False)
    parser._params = params
    parser._return_value = return_value
    return parser

  def _process_operation_parameters(self):
    """Processes parameters from the OpenAPI operation."""
    parameters = self._operation.parameters or []
    for param in parameters:
      if isinstance(param, Parameter):
        original_name = param.name
        description = param.description or ''
        location = param.in_ or ''
        schema = param.schema_ or {}  # Use schema_ instead of .schema
        schema.description = (
            description if not schema.description else schema.description
        )
        # param.required can be None
        required = param.required if param.required is not None else False

        self._params.append(
            ApiParameter(
                original_name=original_name,
                param_location=location,
                param_schema=schema,
                description=description,
                required=required,
            )
        )

  def _process_request_body(self):
    """Processes the request body from the OpenAPI operation."""
    request_body = self._operation.requestBody
    if not request_body:
      return

    content = request_body.content or {}
    if not content:
      return

    # If request body is an object, expand the properties as parameters
    for _, media_type_object in content.items():
      schema = media_type_object.schema_ or {}
      description = request_body.description or ''

      if schema and schema.type == 'object':
        properties = schema.properties or {}
        for prop_name, prop_details in properties.items():
          self._params.append(
              ApiParameter(
                  original_name=prop_name,
                  param_location='body',
                  param_schema=prop_details,
                  description=prop_details.description,
              )
          )

      elif schema and schema.type == 'array':
        self._params.append(
            ApiParameter(
                original_name='array',
                param_location='body',
                param_schema=schema,
                description=description,
            )
        )
      else:
        # Prefer explicit body name to avoid empty keys when schema lacks type
        # information (e.g., oneOf/anyOf/allOf) while retaining legacy behavior
        # for simple scalar types.
        if schema and (schema.oneOf or schema.anyOf or schema.allOf):
          param_name = 'body'
        elif not schema or not schema.type:
          param_name = 'body'
        else:
          param_name = ''

        self._params.append(
            ApiParameter(
                original_name=param_name,
                param_location='body',
                param_schema=schema,
                description=description,
            )
        )
      break  # Process first mime type only

  def _dedupe_param_names(self):
    """Deduplicates parameter names to avoid conflicts."""
    params_cnt = {}
    for param in self._params:
      name = param.py_name
      if name not in params_cnt:
        params_cnt[name] = 0
      else:
        params_cnt[name] += 1
        param.py_name = f'{name}_{params_cnt[name] -1}'

  def _process_return_value(self) -> Parameter:
    """Returns a Parameter object representing the return type."""
    responses = self._operation.responses or {}
    # Default to empty schema if no 2xx response or if schema is missing
    return_schema = Schema()

    # Take the 20x response with the smallest response code.
    valid_codes = list(
        filter(lambda k: k.startswith('2'), list(responses.keys()))
    )
    min_20x_status_code = min(valid_codes) if valid_codes else None

    if min_20x_status_code and responses[min_20x_status_code].content:
      content = responses[min_20x_status_code].content
      for mime_type in content:
        if content[mime_type].schema_:
          return_schema = content[mime_type].schema_
          break

    self._return_value = ApiParameter(
        original_name='',
        param_location='',
        param_schema=return_schema,
    )

  def get_function_name(self) -> str:
    """Returns the generated function name."""
    operation_id = self._operation.operationId
    if not operation_id:
      raise ValueError('Operation ID is missing')
    return _to_snake_case(operation_id)[:60]

  def get_return_type_hint(self) -> str:
    """Returns the return type hint string (like 'str', 'int', etc.)."""
    return self._return_value.type_hint

  def get_return_type_value(self) -> Any:
    """Returns the return type value (like str, int, List[str], etc.)."""
    return self._return_value.type_value

  def get_parameters(self) -> List[ApiParameter]:
    """Returns the list of Parameter objects."""
    return self._params

  def get_return_value(self) -> ApiParameter:
    """Returns the list of Parameter objects."""
    return self._return_value

  def get_auth_scheme_name(self) -> str:
    """Returns the name of the auth scheme for this operation from the spec."""
    if self._operation.security:
      scheme_name = list(self._operation.security[0].keys())[0]
      return scheme_name
    return ''

  def get_pydoc_string(self) -> str:
    """Returns the generated PyDoc string."""
    pydoc_params = [param.to_pydoc_string() for param in self._params]
    pydoc_description = (
        self._operation.summary or self._operation.description or ''
    )
    pydoc_return = PydocHelper.generate_return_doc(
        self._operation.responses or {}
    )
    pydoc_arg_list = chr(10).join(
        f'        {param_doc}' for param_doc in pydoc_params
    )
    return dedent(f"""
        \"\"\"{pydoc_description}

        Args:
        {pydoc_arg_list}

        {pydoc_return}
        \"\"\"
            """).strip()

  def get_json_schema(self) -> Dict[str, Any]:
    """Returns the JSON schema for the function arguments."""
    properties = {
        p.py_name: jsonable_encoder(p.param_schema, exclude_none=True)
        for p in self._params
    }
    return {
        'properties': properties,
        'required': [p.py_name for p in self._params if p.required],
        'title': f"{self._operation.operationId or 'unnamed'}_Arguments",
        'type': 'object',
    }

  def get_signature_parameters(self) -> List[inspect.Parameter]:
    """Returns a list of inspect.Parameter objects for the function."""
    return [
        inspect.Parameter(
            param.py_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=param.type_value,
        )
        for param in self._params
    ]

  def get_annotations(self) -> Dict[str, Any]:
    """Returns a dictionary of parameter annotations for the function."""
    annotations = {p.py_name: p.type_value for p in self._params}
    annotations['return'] = self.get_return_type_value()
    return annotations
