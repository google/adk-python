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

from .auth_credential import AuthCredential
from .auth_credential import AuthCredentialTypes
from .auth_credential import OAuth2Auth
from .auth_schemes import AuthScheme
from .auth_schemes import AuthSchemeType
from .auth_schemes import OpenIdConnectWithConfig
from .auth_tool import AuthConfig
from .base_auth_provider import BaseAuthProvider

if TYPE_CHECKING:
  from .auth_handler import AuthHandler


def __getattr__(name: str):
  if name == 'AuthHandler':
    return importlib.import_module(f'{__name__}.auth_handler').AuthHandler
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
