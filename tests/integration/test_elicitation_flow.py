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

import os
import pytest

from google.adk.evaluation.agent_evaluator import AgentEvaluator

try:
  import pandas as pd
  HAS_EVAL_DEPS = True
except ImportError:
  HAS_EVAL_DEPS = False

HAS_CREDENTIALS = (
    "GOOGLE_API_KEY" in os.environ
    or ("GOOGLE_CLOUD_PROJECT" in os.environ and "GOOGLE_CLOUD_LOCATION" in os.environ)
)

pytestmark = pytest.mark.skipif(
    not (HAS_EVAL_DEPS and HAS_CREDENTIALS),
    reason="Integration test requires 'google-adk[eval]' dependencies and LLM API credentials.",
)


@pytest.mark.asyncio
async def test_elicitation_flow():
  """Test the full multi-turn stateless elicitation flow end-to-end."""
  await AgentEvaluator.evaluate(
      agent_module="tests.integration.fixture.elicitation_agent",
      eval_dataset_file_path_or_dir="tests/integration/fixture/elicitation_agent/elicitation_flow.test.json",
      num_runs=2,
  )
