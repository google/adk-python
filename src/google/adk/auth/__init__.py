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
  from .auth_credential import AuthCredential
  from .auth_credential import AuthCredentialTypes
  from .auth_credential import OAuth2Auth
  from .auth_handler import AuthHandler
  from .auth_schemes import AuthScheme
  from .auth_schemes import AuthSchemeType
  from .auth_schemes import OpenIdConnectWithConfig
  from .auth_tool import AuthConfig
  from .base_auth_provider import BaseAuthProvider

_LAZY_MEMBERS: dict[str, str] = {
    'AuthConfig': '.auth_tool',
    'AuthCredential': '.auth_credential',
    'AuthCredentialTypes': '.auth_credential',
    'AuthHandler': '.auth_handler',
    'AuthScheme': '.auth_schemes',
    'AuthSchemeType': '.auth_schemes',
    'BaseAuthProvider': '.base_auth_provider',
    'OAuth2Auth': '.auth_credential',
    'OpenIdConnectWithConfig': '.auth_schemes',
}
__all__ = [
    'AuthConfig',
    'AuthCredential',
    'AuthCredentialTypes',
    'AuthHandler',
    'AuthScheme',
    'AuthSchemeType',
    'BaseAuthProvider',
    'OAuth2Auth',
    'OpenIdConnectWithConfig',
]

_lazy_getattr, __dir__ = _lazy.accessors(globals(), _LAZY_MEMBERS)


# griffe misses a tuple-assigned __getattr__ and reports it removed.
def __getattr__(name: str) -> object:
  return _lazy_getattr(name)
