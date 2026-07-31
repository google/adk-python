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

"""PEP 562 support for packages that re-export heavyweight submodules."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
import importlib
from typing import Any


def accessors(
    module_globals: dict[str, Any],
    members: Mapping[str, str],
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
  """Builds a package's ``__getattr__`` and ``__dir__``.

  Args:
    module_globals: The calling package's ``globals()``. Resolved members are
      written back into it, so each submodule is imported at most once.
    members: Exported name to the relative module that defines it.
  """
  package: str = module_globals['__name__']

  def module_getattr(name: str) -> Any:
    if name not in members:
      raise AttributeError(f'module {package!r} has no attribute {name!r}')
    module = importlib.import_module(members[name], package)
    value = getattr(module, name)
    module_globals[name] = value
    return value

  def module_dir() -> list[str]:
    return sorted(set(module_globals) | set(module_globals.get('__all__', ())))

  return module_getattr, module_dir
