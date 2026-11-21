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

"""Open Knowledge Format (OKF) v0.2 bundle integration (Experimental).

Exposes a local directory of typed, declarative knowledge — markdown files with
YAML frontmatter describing concepts, with their trust, lifecycle, and staleness
signals — to an agent through a progressive-disclosure toolset, so context cost
is paid per concept read instead of once for the whole corpus.

    from google.adk.integrations.okf import OkfToolset

    agent = Agent(
        model="gemini-flash-latest",
        name="knowledge_agent",
        tools=[OkfToolset(bundle="./knowledge", tool_name_prefix="okf")],
    )
    # -> okf_index, okf_read, okf_resource, okf_search

Specification: Open Knowledge Format v0.2.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
  from ._toolset import OkfToolset

# Map attribute names to relative module paths.
_lazy_imports = {
    "OkfToolset": "._toolset",
}


def __getattr__(name: str) -> typing.Any:
  if name in _lazy_imports:
    import importlib

    module_path = _lazy_imports[name]
    # __name__ is 'google.adk.integrations.okf'
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return list(_lazy_imports.keys())


__all__ = [
    "OkfToolset",
]
