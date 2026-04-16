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

import sys
import unittest
import google.adk

class TestLazyLoading(unittest.TestCase):
    def test_agent_not_loaded(self):
        # Before accessing Agent, it shouldn't be in sys.modules
        self.assertNotIn("google.adk.agents.llm_agent", sys.modules)
        
        # Accessing Agent
        _ = google.adk.Agent
        
        # Now it should be loaded
        self.assertIn("google.adk.agents.llm_agent", sys.modules)
