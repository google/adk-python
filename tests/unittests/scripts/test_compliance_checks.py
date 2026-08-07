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

import pathlib

from scripts import compliance_checks

# A filename that is not in the exclusion list, so check_mtls runs the real
# check instead of short-circuiting on the exclusion.
_UNEXCLUDED_NAME = 'unexcluded.py'

_REPO_ROOT = pathlib.Path(compliance_checks.__file__).resolve().parents[1]


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


def test_mtls_exclusions_are_all_still_needed() -> None:
  assert _UNEXCLUDED_NAME not in compliance_checks._EXCLUDED_FROM_MTLS
  redundant: list[str] = []
  for path in sorted(compliance_checks._EXCLUDED_FROM_MTLS):
    source = _REPO_ROOT / path
    if not source.is_file():
      continue
    content = source.read_text(encoding='utf-8')
    if compliance_checks.check_mtls(content, _UNEXCLUDED_NAME):
      redundant.append(path)
  assert not redundant, (
      'These files pass the mTLS check on their own; drop them from'
      f' _EXCLUDED_FROM_MTLS: {redundant}'
  )
