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

"""Backward-compatibility shim for flows/llm_flows/context/_fencing.py."""

from __future__ import annotations

from .context._fencing import _is_other_agent_reply as _is_other_agent_reply
from .context._fencing import _present_other_agent_message as _present_other_agent_message
from .context._fencing import elide_quote_markers as elide_quote_markers
from .context._fencing import OTHER_AGENT_CONTEXT_PREAMBLE as OTHER_AGENT_CONTEXT_PREAMBLE
from .context._fencing import quote_untrusted as quote_untrusted
from .context._fencing import QUOTED_CONTENT_BEGIN as QUOTED_CONTENT_BEGIN
from .context._fencing import QUOTED_CONTENT_ELIDED as QUOTED_CONTENT_ELIDED
from .context._fencing import QUOTED_CONTENT_END as QUOTED_CONTENT_END

__all__ = [
    'OTHER_AGENT_CONTEXT_PREAMBLE',
    'QUOTED_CONTENT_BEGIN',
    'QUOTED_CONTENT_ELIDED',
    'QUOTED_CONTENT_END',
    '_is_other_agent_reply',
    '_present_other_agent_message',
    'elide_quote_markers',
    'quote_untrusted',
]
