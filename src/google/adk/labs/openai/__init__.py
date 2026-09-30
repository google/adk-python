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

"""Backward-compatible alias for :mod:`google.adk.integrations.openai`.

The OpenAI models moved to ``google.adk.integrations.openai``. This module
re-exports them so existing ``from google.adk.labs.openai import ...`` imports
keep working. New code should import from ``google.adk.integrations.openai``.
"""

from ...integrations.openai import AzureOpenAIResponsesLlm
from ...integrations.openai import OpenAIGenerateContentConfig
from ...integrations.openai import OpenAILlm
from ...integrations.openai import OpenAIResponsesLlm

__all__ = [
    'AzureOpenAIResponsesLlm',
    'OpenAIGenerateContentConfig',
    'OpenAILlm',
    'OpenAIResponsesLlm',
]
