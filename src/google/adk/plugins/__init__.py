# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may in obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
from typing import Any
from typing import TYPE_CHECKING

from .base_plugin import BasePlugin
from .plugin_manager import PluginManager

if TYPE_CHECKING:
  from ..tools.model_consult._force_escalation_plugin import ForceEscalationPlugin
  from ..tools.model_consult._observability_plugin import ModelConsultObservabilityPlugin
  from ._callback_helpers import ToolResultClassification
  from ._callback_helpers import ToolResultClassifier
  from ._reflect_retry_model_plugin import ReflectAndRetryModelPlugin
  from ._tool_call_integrity_plugin import ToolCallIntegrityError
  from ._tool_call_integrity_plugin import ToolCallIntegrityPlugin
  from .auto_tracing_plugin import AutoTracingPlugin
  from .bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
  from .context_filter_plugin import ContextFilterPlugin
  from .debug_logging_plugin import DebugLoggingPlugin
  from .global_instruction_plugin import GlobalInstructionPlugin
  from .logging_plugin import LoggingPlugin
  from .reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
  from .save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin

__all__ = [
    "AutoTracingPlugin",
    "BasePlugin",
    "BigQueryAgentAnalyticsPlugin",
    "ContextFilterPlugin",
    "DebugLoggingPlugin",
    "ForceEscalationPlugin",
    "GlobalInstructionPlugin",
    "LoggingPlugin",
    "ModelConsultObservabilityPlugin",
    "PluginManager",
    "ReflectAndRetryModelPlugin",
    "ReflectAndRetryToolPlugin",
    "SaveFilesAsArtifactsPlugin",
    "ToolCallIntegrityError",
    "ToolCallIntegrityPlugin",
    "ToolResultClassification",
    "ToolResultClassifier",
]

_LAZY_MEMBERS: dict[str, str] = {
    "AutoTracingPlugin": "auto_tracing_plugin",
    "BigQueryAgentAnalyticsPlugin": "bigquery_agent_analytics_plugin",
    "ContextFilterPlugin": "context_filter_plugin",
    "DebugLoggingPlugin": "debug_logging_plugin",
    "ForceEscalationPlugin": "..tools.model_consult._force_escalation_plugin",
    "GlobalInstructionPlugin": "global_instruction_plugin",
    "LoggingPlugin": "logging_plugin",
    "ModelConsultObservabilityPlugin": (
        "..tools.model_consult._observability_plugin"
    ),
    "ReflectAndRetryModelPlugin": "_reflect_retry_model_plugin",
    "ReflectAndRetryToolPlugin": "reflect_retry_tool_plugin",
    "SaveFilesAsArtifactsPlugin": "save_files_as_artifacts_plugin",
    "ToolCallIntegrityError": "_tool_call_integrity_plugin",
    "ToolCallIntegrityPlugin": "_tool_call_integrity_plugin",
    "ToolResultClassification": "_callback_helpers",
    "ToolResultClassifier": "_callback_helpers",
}


def __getattr__(name: str) -> Any:
  if name in _LAZY_MEMBERS:
    target = _LAZY_MEMBERS[name]
    module_name = target if target.startswith(".") else f".{target}"
    try:
      module = importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as e:
      if name == "BigQueryAgentAnalyticsPlugin":
        from ..utils._dependency import missing_extra

        raise missing_extra(
            "google-cloud-bigquery", "bigquery-analytics"
        ) from e
      raise
    return vars(module)[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
