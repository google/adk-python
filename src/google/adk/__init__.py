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

from . import version
from .utils import _lazy

if TYPE_CHECKING:
  from .agents.context import Context
  from .agents.llm_agent import Agent
  from .events.event import Event
  from .runners import Runner
  from .workflow import Workflow

__version__ = version.__version__
_LAZY_MEMBERS: dict[str, str] = {
    'Agent': '.agents.llm_agent',
    'Context': '.agents.context',
    'Event': '.events.event',
    'Runner': '.runners',
    'Workflow': '.workflow',
}
__all__ = ['Agent', 'Context', 'Event', 'Runner', 'Workflow']

__getattr__, __dir__ = _lazy.accessors(globals(), _LAZY_MEMBERS)
