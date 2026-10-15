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

from google.adk import plugins
import pytest

from .. import isolated_import_utils
from ..isolated_import_utils import assert_modules_unloaded

_SHIPPED_PLUGINS = {
    'AutoTracingPlugin': 'auto_tracing_plugin',
    'BigQueryAgentAnalyticsPlugin': 'bigquery_agent_analytics_plugin',
    'ContextFilterPlugin': 'context_filter_plugin',
    'DebugLoggingPlugin': 'debug_logging_plugin',
    'GlobalInstructionPlugin': 'global_instruction_plugin',
    'LoggingPlugin': 'logging_plugin',
    'MultimodalToolResultsPlugin': 'multimodal_tool_results_plugin',
    'ReflectAndRetryModelPlugin': '_reflect_retry_model_plugin',
    'ReflectAndRetryToolPlugin': 'reflect_retry_tool_plugin',
    'SaveFilesAsArtifactsPlugin': 'save_files_as_artifacts_plugin',
    'ToolCallIntegrityPlugin': '_tool_call_integrity_plugin',
}


@pytest.mark.parametrize('name, module', sorted(_SHIPPED_PLUGINS.items()))
def test_shipped_plugin_is_exported_from_package(name, module):
  defining_module = importlib.import_module(f'google.adk.plugins.{module}')

  assert name in plugins.__all__
  assert getattr(plugins, name) is getattr(defining_module, name)


def test_every_name_in_all_resolves():
  for name in plugins.__all__:
    assert getattr(plugins, name).__name__ == name


def test_unknown_name_raises_attribute_error():
  with pytest.raises(AttributeError, match='NoSuchPlugin'):
    getattr(plugins, 'NoSuchPlugin')


@pytest.mark.skipif(
    not isolated_import_utils.SOURCE_ROOT.is_dir(),
    reason='Import-loading checks need the source checkout layout.',
)
def test_package_import_does_not_load_plugin_modules():
  assert_modules_unloaded(
      'import google.adk.plugins',
      tuple(f'google.adk.plugins.{m}' for m in _SHIPPED_PLUGINS.values())
      + ('google.cloud.bigquery',),
  )
