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

from typing import TYPE_CHECKING

from ..utils import _lazy

if TYPE_CHECKING:
  from ._base_node import BaseNode
  from ._base_node import START
  from ._errors import NodeTimeoutError
  from ._function_node import FunctionNode
  from ._graph import DEFAULT_ROUTE
  from ._graph import Edge
  from ._join_node import JoinNode
  from ._node import Node
  from ._node import node
  from ._retry_config import RetryConfig
  from ._workflow import Workflow

_LAZY_MEMBERS: dict[str, str] = {
    'BaseNode': '._base_node',
    'DEFAULT_ROUTE': '._graph',
    'Edge': '._graph',
    'FunctionNode': '._function_node',
    'JoinNode': '._join_node',
    'Node': '._node',
    'NodeTimeoutError': '._errors',
    'RetryConfig': '._retry_config',
    'START': '._base_node',
    'Workflow': '._workflow',
    'node': '._node',
}
__all__ = [
    'BaseNode',
    'DEFAULT_ROUTE',
    'Edge',
    'FunctionNode',
    'JoinNode',
    'Node',
    'NodeTimeoutError',
    'RetryConfig',
    'START',
    'Workflow',
    'node',
]

__getattr__, __dir__ = _lazy.accessors(globals(), _LAZY_MEMBERS)
