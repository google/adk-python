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

from scripts import compliance_checks


def test_check_mtls_ignores_oauth_scope() -> None:
  content = 'scope = "https://www.googleapis.com/auth/cloud-platform"\n'
  assert compliance_checks.check_mtls(content, 'test_file.py') is True


def test_check_mtls_detects_missing_mtls() -> None:
  content = 'endpoint = "https://storage.googleapis.com"\n'
  assert compliance_checks.check_mtls(content, 'test_file.py') is False


def test_check_mtls_passes_with_mtls() -> None:
  content = (
      'endpoint = "https://storage.googleapis.com"\n'
      'mtls_endpoint = "https://storage.mtls.googleapis.com"\n'
  )
  assert compliance_checks.check_mtls(content, 'test_file.py') is True
