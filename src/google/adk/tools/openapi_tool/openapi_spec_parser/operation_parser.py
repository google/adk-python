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

import inspect
from textwrap import dedent
from typing import Any
from typing import cast
from typing import Dict
from typing import List

from fastapi.encoders import jsonable_encoder
from fastapi.openapi.models import Operation
from fastapi.openapi.models import Parameter
from fastapi.openapi.models import Reference
from fastapi.openapi.models import RequestBody
from fastapi.openapi.models import Response
from fastapi.openapi.models import Schema

from ..._gemini_schema_util import _to_snake_case
from ..common.common import _schema_from_openapi
from ..common.common import ApiParameter
from ..common.common import PydocHelper
from ..common.common import rename_python_keywords


class OperationParser:
  """Generates parameters for Python functions from an OpenAPI operation.

  This class processes an OpenApiOperation object and provides helper methods
  to extract information needed to generate Python function declarations,
  docstrings, signatures, and JSON schemas.  It handles parameter processing,
  name deduplication, and type hint generation.
  """

  def __init__(
      self,
      operation: Operation | Dict[str, Any] | str,
      should_parse: bool = True,
      *,
      preserve_property_names: bool = False,
  ) -> None:
    """Initializes the OperationParser with an OpenApiOperation.

    Args:
        operation: The OpenApiOperation object or a dictionary to process.
        should_parse: Whether to parse the operation during initialization.
        preserve_property_names: If True, preserve the original property names
          from the OpenAPI spec instead of converting them to snake_case.
          Useful for APIs that expect camelCase or other non-snake_case
          parameter names.
    """
    if isinstance(operation, dict):
      self._operation = Operation.model_validate(operation)
    elif isinstance(operation, str):
      self._operation = Operation.model_validate_json(operation)
    else:
      self._operation = operation

    self._preserve_property_names = preserve_property_names
    self._params: List[ApiParameter] = []
    self._return_value: ApiParameter | None = None
    if should_parse:
      self._process_operation_parameters()
      self._process_request_body()
      self._process_return_value()
      self._dedupe_param_names()

  @classmethod
  def load(
      cls,
      operation: Operation | Dict[str, Any],
      params: List[ApiParameter],
      return_value: ApiParameter | None = None,
      *,
      preserve_property_names: bool = False,
  ) -> 'OperationParser':
    parser = cls(
        operation,
        should_parse=False,
        preserve_property_names=preserve_property_names,
    )
    parser._params = params
    parser._return_value = return_value
    return parser

  def _get_py_name(self, original_name: str) -> str:
    """Determines the Python parameter name based on preserve_property_names."""
    if self._preserve_property_names:
      return rename_python_keywords(original_name)
    return ''

  def _process_operation_parameters(self) -> None:
    """Processes parameters from the OpenAPI operation."""
    parameters = self._operation.parameters or []
    for param in parameters:
      # Anything that is not a resolved Parameter (an unresolved $ref, say) is
      # skipped, not rejected: one dangling pointer must not take down every
      # tool in the toolset.
      if not isinstance(param, Parameter):
        continue

      original_name = param.name
      description = param.description or ''
      location = param.in_ or ''
      schema = _schema_from_openapi(
          param.schema_, context=f'operation parameter {original_name!r}'
      )
      if not schema.description:
        schema.description = description
      # param.required can be None
      required = param.required if param.required is not None else False

      self._params.append(
          ApiParameter(
              original_name=original_name,
              param_location=location,
              param_schema=schema,
              description=description,
              required=required,
              py_name=self._get_py_name(original_name),
          )
      )

  def _get_request_body(self) -> RequestBody | None:
    request_body = self._operation.requestBody
    if isinstance(request_body, Reference):
      raise ValueError(
          f'Request body contains unresolved reference {request_body.ref!r}'
      )
    if request_body is not None and not isinstance(request_body, RequestBody):
      raise TypeError(f'Unsupported request body {type(request_body)!r}')
    return request_body

  def _process_request_body(self) -> None:
    """Processes the request body from the OpenAPI operation."""
    request_body = self._get_request_body()
    if request_body is None:
      return

    content = request_body.content
    if not content:
      return

    # If request body is an object, expand the properties as parameters
    for media_type, media_type_object in content.items():
      schema = _schema_from_openapi(
          media_type_object.schema_,
          context=f'request body media type {media_type!r}',
      )
      description = request_body.description or ''

      if schema.type == 'object':
        properties = schema.properties or {}
        required_properties = set(schema.required or [])
        for prop_name, prop_details in properties.items():
          property_schema = _schema_from_openapi(
              prop_details,
              context=f'request body property {prop_name!r}',
          )
          self._params.append(
              ApiParameter(
                  original_name=prop_name,
                  param_location='body',
                  param_schema=property_schema,
                  description=property_schema.description or '',
                  required=prop_name in required_properties,
                  py_name=self._get_py_name(prop_name),
              )
          )

      elif schema.type == 'array':
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
        if schema.oneOf or schema.anyOf or schema.allOf:
          param_name = 'body'
        elif not schema.type:
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

  def _dedupe_param_names(self) -> None:
    """Deduplicates parameter names to avoid conflicts."""
    params_cnt: dict[str, int] = {}
    for param in self._params:
      # model_post_init guarantees py_name is set.
      name = cast(str, param.py_name)
      if name not in params_cnt:
        params_cnt[name] = 0
      else:
        params_cnt[name] += 1
        param.py_name = f'{name}_{params_cnt[name] -1}'

  def _process_return_value(self) -> None:
    """Processes the first successful response into a return type."""
    responses = self._operation.responses or {}
    # Default to empty schema if no 2xx response or if schema is missing
    return_schema = Schema()

    # Take the 20x response with the smallest response code.
    valid_codes = list(
        filter(lambda k: k.startswith('2'), list(responses.keys()))
    )
    min_20x_status_code = min(valid_codes) if valid_codes else None

    if min_20x_status_code:
      response = responses[min_20x_status_code]
      if isinstance(response, Reference):
        raise ValueError(
            f'Response contains unresolved reference {response.ref!r}'
        )
      if not isinstance(response, Response):
        raise TypeError(f'Unsupported response {type(response)!r}')
      content = response.content or {}
      for mime_type, media_type_object in content.items():
        if media_type_object.schema_ is not None:
          return_schema = _schema_from_openapi(
              media_type_object.schema_,
              context=f'response media type {mime_type!r}',
          )
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
    return cast(str, self._require_return_value().type_hint)

  # Public API, so the return stays `Any` rather than the true `object`.
  def get_return_type_value(self) -> Any:
    """Returns the return type value (like str, int, List[str], etc.)."""
    return self._require_return_value().type_value

  def get_parameters(self) -> List[ApiParameter]:
    """Returns the list of Parameter objects."""
    return self._params

  def get_return_value(self) -> ApiParameter:
    """Returns the list of Parameter objects."""
    return self._require_return_value()

  def _require_return_value(self) -> ApiParameter:
    if self._return_value is None:
      raise RuntimeError('Operation return value has not been parsed')
    return self._return_value

  def get_auth_scheme_name(self) -> str:
    """Returns the name of the auth scheme for this operation from the spec."""
    if self._operation.security:
      return next(iter(self._operation.security[0]), '')
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
    properties: dict[str, Any] = {
        cast(str, p.py_name): jsonable_encoder(
            p._openapi_schema, exclude_none=True
        )
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
            cast(str, param.py_name),
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=param.type_value,
        )
        for param in self._params
    ]

  def get_annotations(self) -> Dict[str, Any]:
    """Returns a dictionary of parameter annotations for the function."""
    annotations: dict[str, Any] = {
        cast(str, p.py_name): p.type_value for p in self._params
    }
    annotations['return'] = self.get_return_type_value()
    return annotations
