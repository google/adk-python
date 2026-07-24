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

from typing import Any
from typing import TYPE_CHECKING

from google.adk.models.registry import LLMRegistry as _LLMRegistry

if TYPE_CHECKING:
  from ._llm_audio_user_simulator import LlmAudioUserSimulatorConfig

# Register _CloudTTSLlm lazily so that LLMRegistry.resolve("cloud_tts") works
# without eagerly importing the Cloud TTS dependency.
_LLMRegistry._register_lazy(
    [r"cloud_tts"],
    f"{__name__}._cloud_tts_llm",
    "_CloudTTSLlm",
)

__all__ = [
    "LlmAudioUserSimulatorConfig",
]


def __getattr__(name: str) -> Any:
  """Lazily exposes public symbols to avoid circular imports at init time.

  ``conversation_scenarios`` imports this subpackage (via
  ``pre_built_personas``), while ``_llm_audio_user_simulator`` imports
  ``conversation_scenarios``. Deferring the import until first attribute
  access breaks that cycle while keeping ``LlmAudioUserSimulatorConfig``
  importable from the package level.
  """
  if name == "LlmAudioUserSimulatorConfig":
    from ._llm_audio_user_simulator import LlmAudioUserSimulatorConfig

    return LlmAudioUserSimulatorConfig
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
