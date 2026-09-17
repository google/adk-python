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

"""Helpers for tests that have to run against either MCP SDK major.

The pin admits 1.x and 2.x, so the suite runs against both. The differences
below reach only the tests. Production code is unaffected: it reads model
fields under both spellings already, and it only ever catches `McpError`.
"""

from __future__ import annotations

import re
from typing import Any

from google.adk.dependencies._mcp import IS_MCP_SDK_V2
from google.adk.dependencies._mcp import McpError

_CAMEL_BOUNDARY = re.compile(r'(?<!^)(?=[A-Z])')


def field(obj: Any, camel: str) -> Any:
  """Reads a model field under whichever spelling the installed SDK uses.

  1.x names these fields in camelCase (`inputSchema`); 2.x renamed them to
  snake_case (`input_schema`) and dropped the old attribute.

  Args:
    obj: The SDK model to read from.
    camel: The 1.x field name, e.g. `inputSchema`.

  Returns:
    The field's value.
  """
  snake = _CAMEL_BOUNDARY.sub('_', camel).lower()
  for name in (camel, snake):
    if hasattr(obj, name):
      return getattr(obj, name)
  raise AttributeError(
      f'{type(obj).__name__} has neither {camel!r} nor {snake!r}'
  )


def make_mcp_error(code: int, message: str, data: Any = None) -> McpError:
  """Builds the SDK's error exception.

  1.x takes a single `ErrorData`; 2.x takes the fields directly. Both expose
  `.error` afterwards, so only construction differs.

  Args:
    code: The JSON-RPC error code.
    message: The error message.
    data: Optional structured payload.

  Returns:
    An exception of the installed SDK's error type.
  """
  if IS_MCP_SDK_V2:
    return McpError(code, message, data)

  from mcp.types import ErrorData  # pylint: disable=g-import-not-at-top

  if data is None:
    return McpError(ErrorData(code=code, message=message))
  return McpError(ErrorData(code=code, message=message, data=data))


def sdk_progress_fn_t() -> Any:
  """The SDK's own `ProgressFnT` protocol.

  2.x moved it from `mcp.shared.session` to `mcp.shared.dispatcher`. Importing
  one path and skipping when it is missing would retire the comparison on the
  major it most needs to run on.

  Returns:
    The SDK's `ProgressFnT`.
  """
  if IS_MCP_SDK_V2:
    from mcp.shared.dispatcher import ProgressFnT  # pylint: disable=g-import-not-at-top
  else:
    from mcp.shared.session import ProgressFnT  # pylint: disable=g-import-not-at-top
  return ProgressFnT


def _restore_meta(value: Any) -> Any:
  """A second, independent implementation of the `_meta` -> `meta` rename.

  Deliberately not the production `_restore_meta_keys`. The dicts built here
  are the oracle the result assertions compare against, so calling the code
  under test to produce them would make all of those assertions tautological.
  """
  opaque = ('_meta', 'structuredContent', 'inputSchema', 'outputSchema')
  if isinstance(value, dict):
    return {
        'meta'
        if key == '_meta'
        else key: item if key in opaque else _restore_meta(item)
        for key, item in value.items()
    }
  if isinstance(value, list):
    return [_restore_meta(item) for item in value]
  return value


def expected_tool_result(response: Any) -> dict[str, Any]:
  """The dict ADK's contract says a caller gets back for `response`.

  ADK publishes 1.x's camelCase spellings whichever SDK is installed, so a
  plain `model_dump` is not the contract under 2.x. `TestResultDictKeys` pins
  the literal key names; this is for the many tests that only care that the
  payload round-trips.

  Spelled out here rather than delegated to the production dump, so that
  dropping `by_alias` or the `meta` pass from it fails these tests too.

  Args:
    response: The `CallToolResult` the session returned.

  Returns:
    The expected result dict.
  """
  if not IS_MCP_SDK_V2:
    # 1.x needs no normalizing, and must not get any: its models are
    # `extra='allow'`, so a server extra would be walked into.
    return response.model_dump(exclude_none=True, mode='json')
  dumped = _restore_meta(
      response.model_dump(exclude_none=True, mode='json', by_alias=True)
  )
  dumped.pop('resultType', None)
  return dumped
