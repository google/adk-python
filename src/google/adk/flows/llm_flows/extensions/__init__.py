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

"""Extensions subpackage for LLM flows.

This subpackage contains specialized capabilities extending LLM flows:
- _agent_transfer: Transferring tasks between agents in an agent hierarchy
- _code_execution: Pre- and post-processing for external and built-in code executors
- _planning: Natural language planning and reasoning processor
"""

from . import _agent_transfer
from . import _code_execution
from . import _planning

__all__ = [
    '_agent_transfer',
    '_code_execution',
    '_planning',
]
