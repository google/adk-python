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

from typing import Callable
from typing import Optional

from google.genai import types
from typing_extensions import override

from .function_tool import FunctionTool


class LongRunningFunctionTool(FunctionTool):
  """A function tool that returns the result asynchronously.

  This tool is used for long-running operations that may take a significant
  amount of time to complete. The framework will call the function. Once the
  function returns, the response will be returned asynchronously to the
  framework which is identified by the function_call_id.

  Example:
  ```python
  tool = LongRunningFunctionTool(a_long_running_function)
  ```

  Attributes:
    is_long_running: Whether the tool is a long running operation.
  """

  def __init__(self, func: Callable):
    super().__init__(func)
    self.is_long_running = True

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    declaration = super()._get_declaration()
    if declaration:
      instruction = (
          "\n\nNOTE: This is a long-running operation. Do not call this tool"
          " again if it has already returned some intermediate or pending"
          " status."
      )
      if declaration.description:
        declaration.description += instruction
      else:
        declaration.description = instruction.lstrip()
    return declaration
