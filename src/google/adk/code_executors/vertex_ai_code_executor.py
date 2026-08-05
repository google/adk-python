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

from collections.abc import Mapping
import logging
import mimetypes
import os
from typing import TYPE_CHECKING
from typing import TypedDict

from pydantic import PrivateAttr
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult
from .code_execution_utils import File

logger = logging.getLogger('google_adk.' + __name__)

if TYPE_CHECKING:
  from vertexai.preview.extensions import Extension

_SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg']
_SUPPORTED_DATA_FILE_TYPES = ['csv']


class _OutputFile(TypedDict):
  name: str
  contents: str | bytes


class _ExecutionResponse(TypedDict, total=False):
  execution_result: str
  execution_error: str
  output_files: list[_OutputFile]


_IMPORTED_LIBRARIES = '''
import io
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

def crop(s: str, max_chars: int = 64) -> str:
  """Crops a string to max_chars characters."""
  return s[: max_chars - 3] + '...' if len(s) > max_chars else s


def explore_df(df: pd.DataFrame) -> None:
  """Prints some information about a pandas DataFrame."""

  with pd.option_context(
      'display.max_columns', None, 'display.expand_frame_repr', False
  ):
    # Print the column names to never encounter KeyError when selecting one.
    df_dtypes = df.dtypes

    # Obtain information about data types and missing values.
    df_nulls = (len(df) - df.isnull().sum()).apply(
        lambda x: f'{x} / {df.shape[0]} non-null'
    )

    # Explore unique total values in columns using `.unique()`.
    df_unique_count = df.apply(lambda x: len(x.unique()))

    # Explore unique values in columns using `.unique()`.
    df_unique = df.apply(lambda x: crop(str(list(x.unique()))))

    df_info = pd.concat(
        (
            df_dtypes.rename('Dtype'),
            df_nulls.rename('Non-Null Count'),
            df_unique_count.rename('Unique Values Count'),
            df_unique.rename('Unique Values'),
        ),
        axis=1,
    )
    df_info.index.name = 'Columns'
    print(f"""Total rows: {df.shape[0]}
Total columns: {df.shape[1]}

{df_info}""")
'''


def _get_code_interpreter_extension(
    resource_name: str | None = None,
) -> Extension:
  """Returns: Load or create the code interpreter extension."""
  from vertexai.preview.extensions import Extension

  if not resource_name:
    resource_name = os.environ.get('CODE_INTERPRETER_EXTENSION_NAME')
  if resource_name:
    new_code_interpreter = Extension(resource_name)
  else:
    logger.info(
        'No CODE_INTERPRETER_ID found in the environment. Create a new one.'
    )
    new_code_interpreter = Extension.from_hub('code_interpreter')
    os.environ['CODE_INTERPRETER_EXTENSION_NAME'] = (
        new_code_interpreter.gca_resource.name
    )
  return new_code_interpreter


def _normalize_execution_response(response: object) -> _ExecutionResponse:
  """Validate the dynamic response returned by the Vertex extension SDK."""
  if not isinstance(response, Mapping):
    raise TypeError('Code interpreter response must be an object.')

  normalized: _ExecutionResponse = {}
  for field in ('execution_result', 'execution_error'):
    value = response.get(field)
    if value is not None:
      if not isinstance(value, str):
        raise TypeError(f'Code interpreter {field} must be a string.')
      normalized[field] = value

  raw_output_files = response.get('output_files', [])
  if not isinstance(raw_output_files, list):
    raise TypeError('Code interpreter output_files must be a list.')
  output_files: list[_OutputFile] = []
  for raw_file in raw_output_files:
    if not isinstance(raw_file, Mapping):
      raise TypeError('Each code interpreter output file must be an object.')
    name = raw_file.get('name')
    contents = raw_file.get('contents')
    if not isinstance(name, str):
      raise TypeError('Code interpreter output file name must be a string.')
    if not isinstance(contents, (str, bytes)):
      raise TypeError(
          'Code interpreter output file contents must be text or bytes.'
      )
    output_files.append({'name': name, 'contents': contents})
  normalized['output_files'] = output_files
  return normalized


class VertexAiCodeExecutor(BaseCodeExecutor):
  """A code executor that uses Vertex Code Interpreter Extension to execute code.

  Attributes:
    resource_name: If set, load the existing resource name of the code
      interpreter extension instead of creating a new one. Format:
      projects/123/locations/us-central1/extensions/456
  """

  resource_name: str | None = None
  """
  If set, load the existing resource name of the code interpreter extension
  instead of creating a new one.
  Format: projects/123/locations/us-central1/extensions/456
  """

  _code_interpreter_extension: Extension = PrivateAttr()

  def __init__(
      self,
      resource_name: str | None = None,
      **data: object,
  ) -> None:
    """Initializes the VertexAiCodeExecutor.

    Args:
      resource_name: If set, load the existing resource name of the code
        interpreter extension instead of creating a new one. Format:
        projects/123/locations/us-central1/extensions/456
      **data: Additional keyword arguments to be passed to the base class.
    """
    super().__init__(**data)
    self.resource_name = resource_name
    self._code_interpreter_extension = _get_code_interpreter_extension(
        self.resource_name
    )

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    # Execute the code.
    code_execution_result = self._execute_code_interpreter(
        self._get_code_with_imports(code_execution_input.code),
        code_execution_input.input_files,
        code_execution_input.execution_id,
    )
    logger.debug('Executed code:\n```\n%s\n```', code_execution_input.code)

    # Save output file as artifacts.
    saved_files = []
    file_count = 0
    for output_file in code_execution_result['output_files']:
      file_type = output_file['name'].split('.')[-1]
      if file_type in _SUPPORTED_IMAGE_TYPES:
        file_count += 1
        saved_files.append(
            File(
                name=output_file['name'],
                content=output_file['contents'],
                mime_type=f'image/{file_type}',
            )
        )
      elif file_type in _SUPPORTED_DATA_FILE_TYPES:
        file_count += 1
        saved_files.append(
            File(
                name=output_file['name'],
                content=output_file['contents'],
                mime_type=f'text/{file_type}',
            )
        )
      else:
        mime_type, _ = mimetypes.guess_type(output_file['name'])
        saved_files.append(
            File(
                name=output_file['name'],
                content=output_file['contents'],
                mime_type=mime_type or 'application/octet-stream',
            )
        )

    # Collect the final result.
    result = CodeExecutionResult(
        stdout=code_execution_result.get('execution_result', ''),
        stderr=code_execution_result.get('execution_error', ''),
        output_files=saved_files,
    )
    logger.debug('Code execution result: %s', result)
    return result

  def _execute_code_interpreter(
      self,
      code: str,
      input_files: list[File] | None = None,
      session_id: str | None = None,
  ) -> _ExecutionResponse:
    """Executes the code interpreter extension.

    Args:
      code: The code to execute.
      input_files: The input files to execute the code with.
      session_id: The session ID to execute the code with.

    Returns:
      The response from the code interpreter extension.
    """
    operation_params: dict[str, object] = {'code': code}
    if input_files:
      operation_params['files'] = [
          {'name': f.name, 'contents': f.content} for f in input_files
      ]
    if session_id:
      operation_params['session_id'] = session_id
    response: object = self._code_interpreter_extension.execute(
        operation_id='execute',
        operation_params=operation_params,
    )
    return _normalize_execution_response(response)

  def _get_code_with_imports(self, code: str) -> str:
    """Builds the code string with built-in imports.

    Args:
      code: The code to execute.

    Returns:
      The code string with built-in imports.
    """
    return f"""
{_IMPORTED_LIBRARIES}

{code}
"""
