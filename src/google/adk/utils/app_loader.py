# Copyright 2025 Google LLC
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

"""Utilities for loading App instances from modules."""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..apps.app import App

logger = logging.getLogger("google_adk." + __name__)


def load_app_from_module(module_path: str) -> Optional["App"]:
  """Try to load an App instance from the agent module.

  Args:
      module_path: Python module path (e.g., 'my_package.my_agent')

  Returns:
      App instance if found, None otherwise
  """
  from ..apps.app import App

  try:
    module = importlib.import_module(module_path)

    # Find the first attribute that is an instance of App
    for name, candidate in inspect.getmembers(module):
      if isinstance(candidate, App):
        logger.info(f"Loaded App instance '{name}' from {module_path}")
        return candidate

    logger.debug(f"No App instance found in {module_path}")

  except (ImportError, AttributeError) as e:
    logger.debug(f"Could not load App from module {module_path}: {e}")

  return None
