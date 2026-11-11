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

"""Prompt, instruction, and schema assembly for LLM flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....utils import _lazy

if TYPE_CHECKING:
  from . import _identity
  from . import _instructions
  from . import _instructions_utils
  from . import _schema
  from ._instructions_utils import inject_session_state as inject_session_state
  from ._instructions_utils import InstructionProvider as InstructionProvider

_LAZY_MEMBERS: dict[str, str] = {
    'InstructionProvider': '._instructions_utils',
    'inject_session_state': '._instructions_utils',
}
__all__ = [
    'InstructionProvider',
    '_identity',
    '_instructions',
    '_instructions_utils',
    '_schema',
    'inject_session_state',
]

__getattr__, __dir__ = _lazy.accessors(globals(), _LAZY_MEMBERS)
