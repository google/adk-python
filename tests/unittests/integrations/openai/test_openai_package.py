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

"""The google.adk.integrations.openai package loads its models lazily."""

import contextlib
import importlib
import sys
from unittest import mock

from google.adk import integrations
from google.adk import models
from google.adk.integrations import openai as integrations_openai
import pytest

_PACKAGE = 'google.adk.integrations.openai'


@contextlib.contextmanager
def _openai_uninstalled():
  """Makes `openai` unimportable and forces this package to re-import."""
  ours = [
      name
      for name in sys.modules
      if name == _PACKAGE or name.startswith(_PACKAGE + '.')
  ]
  saved_modules = {name: sys.modules.pop(name) for name in ours}
  saved_attr = integrations.__dict__.get('openai')
  hidden = {
      name: None
      for name in sys.modules
      if name == 'openai' or name.startswith('openai.')
  }
  hidden['openai'] = None
  try:
    with mock.patch.dict(sys.modules, hidden):
      yield
  finally:
    sys.modules.update(saved_modules)
    if saved_attr is not None:
      integrations.openai = saved_attr


def test_package_imports_without_openai():
  with _openai_uninstalled():
    package = importlib.import_module(_PACKAGE)

    # The config lives in a module that does not need the openai package.
    assert package.OpenAIGenerateContentConfig.__name__ == (
        'OpenAIGenerateContentConfig'
    )


@pytest.mark.parametrize(
    'name', ['AzureOpenAIResponsesLlm', 'OpenAILlm', 'OpenAIResponsesLlm']
)
def test_model_classes_raise_import_error_without_openai(name):
  with _openai_uninstalled():
    package = importlib.import_module(_PACKAGE)

    with pytest.raises(ImportError, match="'openai' package is not installed"):
      getattr(package, name)


def test_model_registry_module_fails_to_import_without_openai():
  """The registry skips a provider only when importing its module fails.

  Pointing the registry at the lazy package would import cleanly and defer the
  failure to the class lookup, which the registry does not guard.
  """
  module_path = models._LAZY_PROVIDERS['OpenAILlm'][1]

  with _openai_uninstalled():
    with pytest.raises(ImportError):
      importlib.import_module(module_path)


def test_lazy_exports_match_all():
  assert sorted(dir(integrations_openai)) == sorted(integrations_openai.__all__)
  for name in integrations_openai.__all__:
    assert getattr(integrations_openai, name).__name__ == name


def test_unknown_attribute_raises_attribute_error():
  with pytest.raises(AttributeError, match='no_such_name'):
    getattr(integrations_openai, 'no_such_name')
