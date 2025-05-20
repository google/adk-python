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

import logging

from .base_code_executor import BaseCodeExecutor
from .built_in_code_executor import BuiltInCodeExecutor
from .code_executor_context import CodeExecutorContext
from .unsafe_local_code_executor import UnsafeLocalCodeExecutor

logger = logging.getLogger('google_adk.' + __name__)

__all__ = [
    'BaseCodeExecutor',
    'BuiltInCodeExecutor',
    'CodeExecutorContext',
    'UnsafeLocalCodeExecutor',
]

try:
  from .vertex_ai_code_executor import VertexAiCodeExecutor

  __all__.append('VertexAiCodeExecutor')
except ImportError:
  logger.debug(
      'The Vertex sdk is not installed. If you want to use the Vertex Code'
      ' Interpreter with agents, please install it. If not, you can ignore this'
      ' warning.'
  )

try:
  from .container_code_executor import ContainerCodeExecutor

  __all__.append('ContainerCodeExecutor')
except ImportError:
  logger.debug(
      'The docker sdk is not installed. If you want to use the Container Code'
      ' Executor with agents, please install it. If not, you can ignore this'
      ' warning.'
  )
