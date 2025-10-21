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

from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

from fastapi.openapi.models import Operation
from google.genai.types import FunctionDeclaration
import requests
from typing_extensions import override

from ....auth.auth_credential import AuthCredential
from ....auth.auth_schemes import AuthScheme
from ..._gemini_schema_util import _to_gemini_schema
from ..._gemini_schema_util import _to_snake_case
from ...base_tool import BaseTool
from ...tool_context import ToolContext
from ..auth.auth_helpers import credential_to_param
from ..auth.auth_helpers import dict_to_auth_scheme
from ..auth.credential_exchangers.auto_auth_credential_exchanger import AutoAuthCredentialExchanger
from ..common.common import ApiParameter
from .openapi_spec_parser import OperationEndpoint
from .openapi_spec_parser import ParsedOperation
from .operation_parser import OperationParser
from .tool_auth_handler import ToolAuthHandler


def snake_to_lower_camel(snake_case_string: str):
  """Converts a snake_case string to a lower_camel_case string.

  Args:
      snake_case_string: The input snake_case string.

  Returns:
      The lower_camel_case string.
  """
  if "_" not in snake_case_string:
    return snake_case_string

  return "".join([
      s.lower() if i == 0 else s.capitalize()
      for i, s in enumerate(snake_case_string.split("_"))
  ])


AuthPreparationState = Literal["pending", "done"]


class RestApiTool(BaseTool):
  """A generic tool that interacts with a REST API.

  * Generates request params and body
  * Attaches auth credentials to API call.

  Example::

    # Each API operation in the spec will be turned into its own tool
    # Name of the tool is the operationId of that operation, in snake case
    operations = OperationGenerator().parse(openapi_spec_dict)
    tool = [RestApiTool.from_parsed_operation(o) for o in operations]
  """

  def __init__(
      self,
      name: str,
      description: str,
      endpoint: Union[OperationEndpoint, str],
      operation: Union[Operation, str],
      auth_scheme: Optional[Union[AuthScheme, str]] = None,
      auth_credential: Optional[Union[AuthCredential, str]] = None,
      should_parse_operation=True,
  ):
    """Initializes the RestApiTool with the given parameters.

    To generate RestApiTool from OpenAPI Specs, use OperationGenerator.
    Example::

      # Each API operation in the spec will be turned into its own tool
      # Name of the tool is the operationId of that operation, in snake case
      operations = OperationGenerator().parse(openapi_spec_dict)
      tool = [RestApiTool.from_parsed_operation(o) for o in operations]

    Hint: Use google.adk.tools.openapi_tool.auth.auth_helpers to construct
    auth_scheme and auth_credential.

    Args:
        name: The name of the tool.
        description: The description of the tool.
        endpoint: Include the base_url, path, and method of the tool.
        operation: Pydantic object or a dict. Representing the OpenAPI Operation
          object
          (https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#operation-object)
        auth_scheme: The auth scheme of the tool. Representing the OpenAPI
          SecurityScheme object
          (https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#security-scheme-object)
        auth_credential: The authentication credential of the tool.
        should_parse_operation: Whether to parse the operation.
    """
    # Gemini restrict the length of function name to be less than 64 characters
    self.name = name[:60]
    self.description = description
    self.endpoint = (
        OperationEndpoint.model_validate_json(endpoint)
        if isinstance(endpoint, str)
        else endpoint
    )
    self.operation = (
        Operation.model_validate_json(operation)
        if isinstance(operation, str)
        else operation
    )
    self.auth_credential, self.auth_scheme = None, None

    self.configure_auth_credential(auth_credential)
    self.configure_auth_scheme(auth_scheme)

    # Private properties
    self.credential_exchanger = AutoAuthCredentialExchanger()
    if should_parse_operation:
      self._operation_parser = OperationParser(self.operation)

  @classmethod
  def from_parsed_operation(cls, parsed: ParsedOperation) -> "RestApiTool":
    """Initializes the RestApiTool from a ParsedOperation object.

    Args:
        parsed: A ParsedOperation object.

    Returns:
        A RestApiTool object.
    """
    operation_parser = OperationParser.load(
        parsed.operation, parsed.parameters, parsed.return_value
    )

    tool_name = _to_snake_case(operation_parser.get_function_name())
    generated = cls(
        name=tool_name,
        description=parsed.operation.description
        or parsed.operation.summary
        or "",
        endpoint=parsed.endpoint,
        operation=parsed.operation,
        auth_scheme=parsed.auth_scheme,
        auth_credential=parsed.auth_credential,
    )
    generated._operation_parser = operation_parser
    return generated

  @classmethod
  def from_parsed_operation_str(
      cls, parsed_operation_str: str
  ) -> "RestApiTool":
    """Initializes the RestApiTool from a dict.

    Args:
        parsed: A dict representation of a ParsedOperation object.

    Returns:
        A RestApiTool object.
    """
    operation = ParsedOperation.model_validate_json(parsed_operation_str)
    return RestApiTool.from_parsed_operation(operation)

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Returns the function declaration in the Gemini Schema format."""
    schema_dict = self._operation_parser.get_json_schema()
    parameters = _to_gemini_schema(schema_dict)
    function_decl = FunctionDeclaration(
        name=self.name, description=self.description, parameters=parameters
    )
    return function_decl

  def configure_auth_scheme(
      self, auth_scheme: Union[AuthScheme, Dict[str, Any]]
  ):
    """Configures the authentication scheme for the API call.

    Args:
        auth_scheme: AuthScheme|dict -: The authentication scheme. The dict is
          converted to a AuthScheme object.
    """
    if isinstance(auth_scheme, dict):
      auth_scheme = dict_to_auth_scheme(auth_scheme)
    self.auth_scheme = auth_scheme

  def configure_auth_credential(
      self, auth_credential: Optional[Union[AuthCredential, str]] = None
  ):
    """Configures the authentication credential for the API call.

    Args:
        auth_credential: AuthCredential|dict - The authentication credential.
          The dict is converted to an AuthCredential object.
    """
    if isinstance(auth_credential, str):
      auth_credential = AuthCredential.model_validate_json(auth_credential)
    self.auth_credential = auth_credential

  def _prepare_auth_request_params(
      self,
      auth_scheme: AuthScheme,
      auth_credential: AuthCredential,
  ) -> Tuple[List[ApiParameter], Dict[str, Any]]:
    # Handle Authentication
    if not auth_scheme or not auth_credential:
      return

    return credential_to_param(auth_scheme, auth_credential)

  def _prepare_request_params(
      self, parameters: List[ApiParameter], kwargs: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Prepares the request parameters for the API call.

    Args:
        parameters: A list of ApiParameter objects representing the parameters
          for the API call.
        kwargs: The keyword arguments passed to the call function from the Tool
          caller.

    Returns:
        A dictionary containing the  request parameters for the API call. This
        initializes a requests.request() call.

    Example:
        self._prepare_request_params({"input_id": "test-id"})
    """

    method = self.endpoint.method.lower()
    if not method:
      raise ValueError("Operation method not found.")

    path_params: Dict[str, Any] = {}
    query_params: Dict[str, Any] = {}
    header_params: Dict[str, Any] = {}
    cookie_params: Dict[str, Any] = {}

    from ....version import __version__ as adk_version

    # Set the custom User-Agent header
    user_agent = f"google-adk/{adk_version} (tool: {self.name})"
    header_params["User-Agent"] = user_agent

    params_map: Dict[str, ApiParameter] = {p.py_name: p for p in parameters}

    # Fill in path, query, header and cookie parameters to the request
    for param_k, v in kwargs.items():
      param_obj = params_map.get(param_k)
      if not param_obj:
        continue  # If input arg not in the ApiParameter list, ignore it.

      original_k = param_obj.original_name
      param_location = param_obj.param_location

      if param_location == "path":
        path_params[original_k] = v
      elif param_location == "query":
        if v:
          query_params[original_k] = v
      elif param_location == "header":
        header_params[original_k] = v
      elif param_location == "cookie":
        cookie_params[original_k] = v

    # Construct URL
    base_url = self.endpoint.base_url or ""
    base_url = base_url[:-1] if base_url.endswith("/") else base_url
    url = f"{base_url}{self.endpoint.path.format(**path_params)}"

    # Construct body
    body_kwargs: Dict[str, Any] = {}
    request_body = self.operation.requestBody
    if request_body:
      for mime_type, media_type_object in request_body.content.items():
        schema = media_type_object.schema_
        body_data = None

        if schema.type == "object":
          body_data = {}
          for param in parameters:
            if param.param_location == "body" and param.py_name in kwargs:
              body_data[param.original_name] = kwargs[param.py_name]

        elif schema.type == "array":
          for param in parameters:
            if param.param_location == "body" and param.py_name == "array":
              body_data = kwargs.get("array")
              break
        else:  # like string
          for param in parameters:
            # original_name = '' indicating this param applies to the full body.
            if param.param_location == "body" and not param.original_name:
              body_data = (
                  kwargs.get(param.py_name) if param.py_name in kwargs else None
              )
              break

        if mime_type == "application/json" or mime_type.endswith("+json"):
          if body_data is not None:
            body_kwargs["json"] = body_data
        elif mime_type == "application/x-www-form-urlencoded":
          body_kwargs["data"] = body_data
        elif mime_type == "multipart/form-data":
          body_kwargs["files"] = body_data
        elif mime_type == "application/octet-stream":
          body_kwargs["data"] = body_data
        elif mime_type == "text/plain":
          body_kwargs["data"] = body_data

        if mime_type:
          header_params["Content-Type"] = mime_type
        break  # Process only the first mime_type

    filtered_query_params: Dict[str, Any] = {
        k: v for k, v in query_params.items() if v is not None
    }

    request_params: Dict[str, Any] = {
        "method": method,
        "url": url,
        "params": filtered_query_params,
        "headers": header_params,
        "cookies": cookie_params,
        **body_kwargs,
    }

    return request_params

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: Optional[ToolContext]
  ) -> Dict[str, Any]:
    return await self.call(args=args, tool_context=tool_context)

  async def call(
      self, *, args: dict[str, Any], tool_context: Optional[ToolContext]
  ) -> Dict[str, Any]:
    """Executes the REST API call.

    Args:
        args: Keyword arguments representing the operation parameters.
        tool_context: The tool context (not used here, but required by the
          interface).

    Returns:
        The API response as a dictionary.
    """
    # Prepare auth credentials for the API call
    tool_auth_handler = ToolAuthHandler.from_tool_context(
        tool_context, self.auth_scheme, self.auth_credential
    )
    auth_result = await tool_auth_handler.prepare_auth_credentials()
    auth_state, auth_scheme, auth_credential = (
        auth_result.state,
        auth_result.auth_scheme,
        auth_result.auth_credential,
    )

    if auth_state == "pending":
      return {
          "pending": True,
          "message": "Needs your authorization to access your data.",
      }

    # Attach parameters from auth into main parameters list
    api_params, api_args = self._operation_parser.get_parameters().copy(), args
    if auth_credential:
      # Attach parameters from auth into main parameters list
      auth_param, auth_args = self._prepare_auth_request_params(
          auth_scheme, auth_credential
      )
      if auth_param and auth_args:
        api_params = [auth_param] + api_params
        api_args.update(auth_args)

    # Got all parameters. Call the API.
    request_params = self._prepare_request_params(api_params, api_args)
    response = requests.request(**request_params)

    # Parse API response
    try:
      response.raise_for_status()  # Raise HTTPError for bad responses
      return response.json()  # Try to decode JSON
    except requests.exceptions.HTTPError:
      error_details = response.content.decode("utf-8")
      return {
          "error": (
              f"Tool {self.name} execution failed. Analyze this execution error"
              " and your inputs. Retry with adjustments if applicable. But"
              " make sure don't retry more than 3 times. Execution Error:"
              f" {error_details}"
          )
      }
    except ValueError:
      return {"text": response.text}  # Return text if not JSON

  def __str__(self):
    return (
        f'RestApiTool(name="{self.name}", description="{self.description}",'
        f' endpoint="{self.endpoint}")'
    )

  def __repr__(self):
    return (
        f'RestApiTool(name="{self.name}", description="{self.description}",'
        f' endpoint="{self.endpoint}", operation="{self.operation}",'
        f' auth_scheme="{self.auth_scheme}",'
        f' auth_credential="{self.auth_credential}")'
    )
