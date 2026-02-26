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

"""Agent Development Kit - Skills."""

from typing import Any
import warnings

from ._utils import _load_skill_from_dir as load_skill_from_dir
from .models import Frontmatter
from .models import Resources
from .models import Script
from .models import Skill

__all__ = [
    "DEFAULT_SKILL_SYSTEM_INSTRUCTION",
    "Frontmatter",
    "Resources",
    "Script",
    "Skill",
    "load_skill_from_dir",
]


def __getattr__(name: str) -> Any:
  if name == "DEFAULT_SKILL_SYSTEM_INSTRUCTION":

    from ..tools import skill_toolset

    warnings.warn(
        (
            "Importing DEFAULT_SKILL_SYSTEM_INSTRUCTION from"
            " google.adk.skills is deprecated."
            " Please import it from google.adk.tools.skill_toolset instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return skill_toolset.DEFAULT_SKILL_SYSTEM_INSTRUCTION
  raise AttributeError(f"module {__name__} has no attribute {name}")
