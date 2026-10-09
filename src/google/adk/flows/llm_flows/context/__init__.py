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

"""Context subpackage for LLM flows.

This subpackage contains components for managing conversation history,
events compaction, context caching, untrusted content fencing, and interaction
state:
- _contents: Assembles conversation history and context into LLM request contents
- _compaction: Token-threshold compaction and event history pruning
- _cache: Context cache configuration and metadata retrieval
- _fencing: Untrusted tool output fencing
- _interactions: Interactions API conversation chaining
"""

from . import _cache
from . import _compaction
from . import _contents
from . import _fencing
from . import _interactions

__all__ = [
    '_cache',
    '_compaction',
    '_contents',
    '_fencing',
    '_interactions',
]
