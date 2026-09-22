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

"""Import-isolation guard for adk_web_server.

Importing ``adk_web_server`` must not eagerly pull in the Agent Builder agent
stack. Doing so reaches ``google.adk.agents`` at import time and breaks
downstream consumers that import ``adk_web_server`` while ``google.adk.agents``
is still initializing.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_adk_web_server_does_not_import_agent_builder():
  # Run in a fresh interpreter so the check is not polluted by modules that
  # other tests already imported into sys.modules.
  code = (
      "import google.adk.cli.adk_web_server\n"
      "import sys\n"
      "forbidden = [\n"
      "    'google.adk.cli.built_in_agents.agent',\n"
      "    'google.adk.cli.built_in_agents.adk_agent_builder_assistant',\n"
      "]\n"
      "loaded = [name for name in forbidden if name in sys.modules]\n"
      "assert not loaded, loaded\n"
  )

  result = subprocess.run(
      [sys.executable, "-c", code],
      capture_output=True,
      text=True,
      check=False,
  )

  assert result.returncode == 0, result.stderr
