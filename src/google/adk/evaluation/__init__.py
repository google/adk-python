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

import logging

logger = logging.getLogger('google_adk.' + __name__)

__all__ = []

try:
  from .agent_evaluator import AgentEvaluator

  __all__.append('AgentEvaluator')
except ImportError:
  logger.debug(
      'The Vertex[eval] sdk is not installed. If you want to use the Vertex'
      ' Evaluation with agents, please install it(pip install'
      ' "google-cloud-aiplatform[evaluation]). If not, you can ignore this'
      ' warning.'
  )
