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
import warnings

from google.adk.integrations.secret_manager import secret_client
import pytest


def test_secret_client_module_deprecation():
  """Verifies that importing from clients.secret_client triggers a warning."""
  module_to_test = "google.adk.tools.apihub_tool.clients.secret_client"
  if module_to_test in sys.modules:
    sys.modules.pop(module_to_test)

  with pytest.warns(
      DeprecationWarning, match="google.adk.integrations.secret_manager"
  ):
    from google.adk.tools.apihub_tool.clients.secret_client import SecretManagerClient as deprecated_secret_client

  assert deprecated_secret_client is secret_client.SecretManagerClient
