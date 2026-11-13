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

"""Which types and field names carry secrets, for every redaction site.

Shared so that the redaction rule is one rule: a type added here is masked by
`AutoTracingPlugin`'s argument capture and by the telemetry serializer alike,
rather than by whichever of them was edited last. Field names are matched only
by `AutoTracingPlugin`, against the parameters it captures.
"""

from __future__ import annotations

import functools
import re
from typing import Callable

_ACRONYM_BOUNDARY_RE = re.compile(r'([A-Z]+)([A-Z][a-z])')
_CAMEL_BOUNDARY_RE = re.compile(r'([a-z0-9])([A-Z])')

# The marker recorded in place of a secret. Same spelling as the one
# `BaseModelWithConfig.__repr_args__` uses for unmodeled credential fields.
REDACTED = '<redacted>'

# Types whose repr() or model_dump() renders live secrets (tokens, keys,
# passwords). Matched by name over the MRO so this module never imports
# ``google.adk.auth``.
CREDENTIAL_TYPE_NAMES = frozenset({
    'AuthConfig',
    'AuthCredential',
    'AuthToolArguments',
    'Credentials',
    'HttpAuth',
    'HttpCredentials',
    'OAuth2Auth',
    'OAuth2Session',
    'ServiceAccount',
    'ServiceAccountCredential',
})
# Parameter and field names that conventionally carry secret material.
CREDENTIAL_ARG_NAMES = frozenset({
    'additional_headers',
    'api_key',
    'auth_code',
    'auth_config',
    'auth_credential',
    'auth_response_uri',
    'authorization',
    'code_verifier',
    'cookie',
    'cookies',
    'credential',
    'credentials',
    'password',
    'private_key',
    'private_key_id',
    'secret',
    'token',
})
CREDENTIAL_ARG_SUFFIXES = (
    '_api_key',
    '_auth_config',
    '_authorization',
    '_cookie',
    '_cookies',
    '_credential',
    '_credentials',
    '_password',
    '_private_key',
    '_secret',
    '_token',
)


def _mro_holds_credential(cls: type) -> bool:
  """True iff ``cls`` or one of its bases is a credential-bearing type."""
  return any(k.__name__ in CREDENTIAL_TYPE_NAMES for k in cls.__mro__)


# Cached because a walk asks this of every non-scalar node it visits. The
# annotation is spelled out because lru_cache erases the wrapped signature to
# ``*args: Hashable``, which the ``type(value)`` the callers pass does not
# satisfy.
is_credential_type: Callable[[type], bool] = functools.lru_cache(maxsize=512)(
    _mro_holds_credential
)


@functools.lru_cache(maxsize=1024)
def is_credential_arg_name(name: str) -> bool:
  """True iff a parameter or field called ``name`` holds a secret."""
  snake = _ACRONYM_BOUNDARY_RE.sub(r'\1_\2', name)
  snake = _CAMEL_BOUNDARY_RE.sub(r'\1_\2', snake)
  lowered = snake.lower().replace('-', '_')
  return lowered in CREDENTIAL_ARG_NAMES or lowered.endswith(
      CREDENTIAL_ARG_SUFFIXES
  )
