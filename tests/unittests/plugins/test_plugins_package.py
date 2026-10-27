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

"""The google.adk.plugins package exports every shipped plugin lazily."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
import sys

from google.adk import plugins
from google.adk.plugins.base_plugin import BasePlugin
import pytest

from tests.unittests import isolated_import_utils

# Slated for removal, so it is deliberately kept out of the package exports.
_UNEXPORTED_PLUGINS = frozenset({'MultimodalToolResultsPlugin'})


def _shipped_plugins() -> dict[str, str]:
  """Maps each plugin class defined under google.adk.plugins to its module."""
  shipped = {}
  for module_info in pkgutil.iter_modules(plugins.__path__):
    module_name = f'{plugins.__name__}.{module_info.name}'
    module = importlib.import_module(module_name)
    for name, value in vars(module).items():
      if (
          inspect.isclass(value)
          and issubclass(value, BasePlugin)
          and value is not BasePlugin
          and value.__module__ == module_name
          and not name.startswith('_')
          and name not in _UNEXPORTED_PLUGINS
      ):
        shipped[name] = module_name
  return shipped


def _lazy_modules() -> set[str]:
  """Returns the absolute name of every module in plugins._LAZY_MEMBERS."""
  modules = set()
  for target in plugins._LAZY_MEMBERS.values():
    relative = target if target.startswith('.') else f'.{target}'
    modules.add(importlib.util.resolve_name(relative, plugins.__name__))
  return modules


_SHIPPED_PLUGINS = _shipped_plugins()


def test_plugin_discovery_finds_concrete_plugins():
  """Discovery finds the shipped plugins and skips the BasePlugin base."""
  assert 'LoggingPlugin' in _SHIPPED_PLUGINS
  assert 'BasePlugin' not in _SHIPPED_PLUGINS


@pytest.mark.parametrize('name, module', sorted(_SHIPPED_PLUGINS.items()))
def test_shipped_plugin_is_exported_from_package(name, module):
  """Every plugin class defined under plugins/ is exported by the package."""
  defining_module = importlib.import_module(module)

  assert name in plugins.__all__
  assert getattr(plugins, name) is getattr(defining_module, name)


def test_unexported_plugins_stay_out_of_package():
  """Plugins slated for removal are not part of the package exports."""
  for name in _UNEXPORTED_PLUGINS:
    assert name not in plugins.__all__
    with pytest.raises(AttributeError):
      getattr(plugins, name)


def test_every_name_in_all_resolves():
  """Every name in __all__ resolves to an object of that name."""
  for name in plugins.__all__:
    assert getattr(plugins, name).__name__ == name


def test_unknown_name_raises_attribute_error():
  """An unknown attribute raises AttributeError naming it."""
  with pytest.raises(AttributeError, match='NoSuchPlugin'):
    getattr(plugins, 'NoSuchPlugin')


def test_bigquery_plugin_names_extra_when_dependency_missing(monkeypatch):
  """A missing BigQuery dependency raises an ImportError naming the extra."""
  monkeypatch.delitem(
      sys.modules,
      'google.adk.plugins.bigquery_agent_analytics_plugin',
      raising=False,
  )
  monkeypatch.setitem(sys.modules, 'google.api_core', None)

  with pytest.raises(ImportError, match=r'google-adk\[bigquery-analytics\]'):
    getattr(plugins, 'BigQueryAgentAnalyticsPlugin')


def test_bigquery_plugin_keeps_pyarrow_error_when_pyarrow_missing(monkeypatch):
  """A missing pyarrow is reported as pyarrow, not as google-cloud-bigquery."""
  monkeypatch.delitem(
      sys.modules,
      'google.adk.plugins.bigquery_agent_analytics_plugin',
      raising=False,
  )
  monkeypatch.setitem(sys.modules, 'pyarrow', None)

  with pytest.raises(ImportError, match="'pyarrow'"):
    getattr(plugins, 'BigQueryAgentAnalyticsPlugin')


@pytest.mark.skipif(
    not isolated_import_utils.SOURCE_ROOT.is_dir(),
    reason='Import-loading checks need the source checkout layout.',
)
def test_package_import_does_not_load_plugin_modules():
  """Importing the package loads no lazily exported module."""
  isolated_import_utils.assert_modules_unloaded(
      'import google.adk.plugins',
      tuple(sorted(_lazy_modules())) + ('google.cloud.bigquery',),
  )
