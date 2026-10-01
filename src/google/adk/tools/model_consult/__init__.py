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

"""Lets a fast executor model consult a stronger advisor model mid-task."""

from ._context import ContextMode
from ._context import ModelConsultContextConfig
from ._model_consult_tool import DEFAULT_ADVISOR_MODEL
from ._model_consult_tool import DEFAULT_TOOL_NAME
from ._model_consult_tool import ModelConsultTool
from ._prompts import ADVISOR_SYSTEM_INSTRUCTION
from ._prompts import EXECUTOR_INSTRUCTION
from ._prompts import TOOL_DESCRIPTION

__all__ = [
    'ADVISOR_SYSTEM_INSTRUCTION',
    'ContextMode',
    'DEFAULT_ADVISOR_MODEL',
    'DEFAULT_TOOL_NAME',
    'EXECUTOR_INSTRUCTION',
    'ModelConsultContextConfig',
    'ModelConsultTool',
    'TOOL_DESCRIPTION',
]
