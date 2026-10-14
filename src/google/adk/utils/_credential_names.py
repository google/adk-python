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

Shared so that the redaction rule is one rule: a type or field name added
here is masked by `AutoTracingPlugin`'s argument capture and by the telemetry
serializer alike, rather than by whichever of them was edited last.
"""

from __future__ import annotations

import functools
from typing import Callable

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
    'api_key',
    'auth_config',
    'auth_credential',
    'authorization',
    'cookie',
    'cookies',
    'credential',
    'credentials',
    'password',
    'private_key',
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
# Bounds for the structural walks that use these names. Both are deliberately
# generous: only containers and objects consume node budget, so a list of a
# million ints costs one node.
MAX_REDACT_DEPTH = 10
MAX_REDACT_NODES = 1024


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
  """True iff a parameter or field called ``name`` conventionally holds a secret."""
  lowered = name.lower()
  return lowered in CREDENTIAL_ARG_NAMES or lowered.endswith(
      CREDENTIAL_ARG_SUFFIXES
  )
