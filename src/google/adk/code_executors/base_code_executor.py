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

import abc
from typing import List

from pydantic import BaseModel

from ..agents.invocation_context import InvocationContext
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult


class BaseCodeExecutor(BaseModel):
  """Abstract base class for all code executors.

  The code executor allows the agent to execute code blocks from model responses
  and incorporate the execution results into the final response.

  Attributes:
    optimize_data_file: If true, extract and process data files from the model
      request and attach them to the code executor. Supported data file
      MimeTypes are [text/csv]. Default to False.
    stateful: Whether the code executor is stateful. Default to False.
    error_retry_attempts: The number of attempts to retry on consecutive code
      execution errors. Default to 2.
    code_block_delimiters: The list of the enclosing delimiters to identify the
      code blocks.
    execution_result_delimiters: The delimiters to format the code execution
      result.
  """

  optimize_data_file: bool = False
  """If true, extract and process data files from the model request
  and attach them to the code executor.

  Supported data file MimeTypes are [text/csv].
  Default to False.
  """

  stateful: bool = False
  """Whether the code executor is stateful. Default to False."""

  error_retry_attempts: int = 2
  """The number of attempts to retry on consecutive code execution errors. Default to 2."""

  code_block_delimiters: List[tuple[str, str]] = [
      ('```tool_code\n', '\n```'),
      ('```python\n', '\n```'),
  ]
  """The list of the enclosing delimiters to identify the code blocks.

  For example, the delimiter ('```python\\n', '\\n```') can be
  used to identify code blocks with the following format::

      ```python
      print("hello")
      ```
  """

  execution_result_delimiters: tuple[str, str] = ('```tool_output\n', '\n```')
  """The delimiters to format the code execution result."""

  @abc.abstractmethod
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    """Executes code and return the code execution result.

    Args:
      invocation_context: The invocation context of the code execution.
      code_execution_input: The code execution input.

    Returns:
      The code execution result.
    """
    pass
