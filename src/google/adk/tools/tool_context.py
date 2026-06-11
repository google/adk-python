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

import importlib
from typing import TYPE_CHECKING

from ..agents.callback_context import CallbackContext
from ..agents.context import Context
from .tool_confirmation import ToolConfirmation

if TYPE_CHECKING:
  from ..auth.auth_credential import AuthCredential
  from ..auth.auth_handler import AuthHandler
  from ..auth.auth_tool import AuthConfig

ToolContext = Context

_LAZY_REEXPORTS: dict[str, tuple[str, str]] = {
    'AuthCredential': ('google.adk.auth.auth_credential', 'AuthCredential'),
    'AuthHandler': ('google.adk.auth.auth_handler', 'AuthHandler'),
    'AuthConfig': ('google.adk.auth.auth_tool', 'AuthConfig'),
}


def __getattr__(name: str):
  if name in _LAZY_REEXPORTS:
    module_path, attr = _LAZY_REEXPORTS[name]
    module = importlib.import_module(module_path)
    return getattr(module, attr)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
