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

"""The registry class for model."""

from __future__ import annotations

from functools import lru_cache
import importlib
import logging
import re
from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
  from .base_llm import BaseLlm

logger = logging.getLogger('google_adk.' + __name__)


_LazyEntry = tuple[str, str]
_llm_registry_dict: dict[str, Union[type['BaseLlm'], _LazyEntry]] = {}


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

    prefix, actual_model = LLMRegistry._parse_model(model)
    cls = LLMRegistry.resolve(model)

    if prefix and LLMRegistry._match_prefix(prefix, cls.__name__):
      return cls(model=actual_model)

    return cls(model=model)

  @staticmethod
  def _parse_model(model: str) -> tuple[str | None, str]:
    """Parses a model name into prefix and actual model.

    Example: "openai:gpt-4" -> ("openai", "gpt-4")
             "gpt-4" -> (None, "gpt-4")
    """
    if ':' in model:
      prefix, actual_model = model.split(':', 1)
      return prefix, actual_model
    return None, model

  @staticmethod
  def _match_prefix(prefix: str, class_name: str) -> bool:
    """Checks if a prefix matches a class name."""
    prefix_lower = prefix.lower()
    norm_class_name = class_name.lower()
    if norm_class_name.endswith('llm'):
      norm_class_name = norm_class_name[:-3]
    return prefix_lower == norm_class_name or prefix_lower == class_name.lower()

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
  def _register_lazy(
      model_name_regexes: list[str], module_path: str, class_name: str
  ):
    """Pre-registers a lazily-imported LLM class."""
    for regex in model_name_regexes:
      _llm_registry_dict[regex] = (module_path, class_name)

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

    # Support [model_class]:model_name format to override resolution
    prefix, _ = LLMRegistry._parse_model(model)
    if prefix:
      for regex, entry in list(_llm_registry_dict.items()):
        class_name = entry[1] if isinstance(entry, tuple) else entry.__name__
        if LLMRegistry._match_prefix(prefix, class_name):
          if isinstance(entry, tuple):
            module_path, c_name = entry
            # We let ImportError bubble up because the user explicitly
            # requested this provider via prefix.
            module = importlib.import_module(module_path)
            return getattr(module, c_name)
          return entry

    for regex, entry in list(_llm_registry_dict.items()):
      if not re.compile(regex).fullmatch(model):
        continue
      if isinstance(entry, tuple):
        module_path, class_name = entry
        try:
          module = importlib.import_module(module_path)
        except ImportError:
          _llm_registry_dict.pop(regex, None)
          continue
        llm_class = getattr(module, class_name)
        _llm_registry_dict[regex] = llm_class
        return llm_class
      return entry

    # Provide helpful error messages for known patterns
    error_msg = f'Model {model} not found.'

    # Check if it matches known patterns that require optional dependencies
    if re.match(r'^claude-', model):
      error_msg += (
          '\n\nClaude models require the anthropic package.'
          '\nInstall it with: pip install google-adk[extensions]'
          '\nOr: pip install anthropic>=0.43.0'
      )
    elif '/' in model:
      # Any model with provider/model format likely needs LiteLLM
      error_msg += (
          '\n\nProvider-style models (e.g., "provider/model-name") require'
          ' the litellm package.'
          '\nInstall it with: pip install google-adk[extensions]'
          '\nOr: pip install litellm>=1.75.5'
          '\n\nSupported providers include: openai, groq, anthropic, and 100+'
          ' others.'
          '\nSee https://docs.litellm.ai/docs/providers for a full list.'
      )

    raise ValueError(error_msg)
