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

"""Deprecated re-export shim.

The DNS-AID A2A bridge has moved to
``google.adk.integrations.dns_aid.a2a_bridge``. This module keeps the old
import path working for one release; new code should import from the new
location.
"""

from __future__ import annotations

import warnings

from google.adk.integrations.dns_aid.a2a_bridge import remote_a2a_agent_from_record

warnings.warn(
    'google.adk.tools.dns_aid_a2a_bridge is deprecated; import from '
    'google.adk.integrations.dns_aid instead.',
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['remote_a2a_agent_from_record']
