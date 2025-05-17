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

"""The registry class for model."""

from __future__ import annotations

from functools import lru_cache
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .base_llm import BaseLlm

logger = logging.getLogger('google_adk.' + __name__)


_llm_registry_dict: dict[str, type[BaseLlm]] = {}
"""Registry for LLMs.

Key is the regex that matches the model name.
Value is the class that implements the model.
"""


class LLMRegistry:
  """Registry for LLMs."""

  @staticmethod
  def new_llm(model: str) -> BaseLlm:
    """Creates a new LLM instance.

    Args:
        model: The model name.

    Returns:
        The LLM instance.
    """

    return LLMRegistry.resolve(model)(model=model)

  @staticmethod
  def _register(model_name_regex: str, llm_cls: type[BaseLlm]):
    """Registers a new LLM class.

    Args:
        model_name_regex: The regex that matches the model name.
        llm_cls: The class that implements the model.
    """

    if model_name_regex in _llm_registry_dict:
      logger.info(
          'Updating LLM class for %s from %s to %s',
          model_name_regex,
          _llm_registry_dict[model_name_regex],
          llm_cls,
      )

    _llm_registry_dict[model_name_regex] = llm_cls

  @staticmethod
  def register(llm_cls: type[BaseLlm]):
    """Registers a new LLM class.

    Args:
        llm_cls: The class that implements the model.
    """

    for regex in llm_cls.supported_models():
      LLMRegistry._register(regex, llm_cls)

  @staticmethod
  @lru_cache(maxsize=32)
  def resolve(model: str) -> type[BaseLlm]:
    """Resolves the model to a BaseLlm subclass.

    Args:
        model: The model name.

    Returns:
        The BaseLlm subclass.
    Raises:
        ValueError: If the model is not found.
    """

    for regex, llm_class in _llm_registry_dict.items():
      if re.compile(regex).fullmatch(model):
        return llm_class

    raise ValueError(f'Model {model} not found.')
