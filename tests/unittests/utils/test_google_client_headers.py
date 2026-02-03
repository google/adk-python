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

import sys

from google.adk import version
from google.adk.utils import _google_client_headers
import pytest

_EXPECTED_BASE_HEADER = (
    f"google-adk/{version.__version__} gl-python/{sys.version.split()[0]}"
)


def test_get_tracking_headers():
  """Test get_tracking_headers returns correct headers."""
  headers = _google_client_headers.get_tracking_headers()
  assert headers == {
      "x-goog-api-client": _EXPECTED_BASE_HEADER,
      "user-agent": _EXPECTED_BASE_HEADER,
  }


@pytest.mark.parametrize(
    "input_headers, expected_headers",
    [
        (
            None,
            {
                "x-goog-api-client": _EXPECTED_BASE_HEADER,
                "user-agent": _EXPECTED_BASE_HEADER,
            },
        ),
        (
            {},
            {
                "x-goog-api-client": _EXPECTED_BASE_HEADER,
                "user-agent": _EXPECTED_BASE_HEADER,
            },
        ),
        (
            {"x-goog-api-client": "label3 label4"},
            {
                "x-goog-api-client": f"{_EXPECTED_BASE_HEADER} label3 label4",
                "user-agent": _EXPECTED_BASE_HEADER,
            },
        ),
        (
            {"x-goog-api-client": f"gl-python/{sys.version.split()[0]} label3"},
            {
                "x-goog-api-client": f"{_EXPECTED_BASE_HEADER} label3",
                "user-agent": _EXPECTED_BASE_HEADER,
            },
        ),
        (
            {"other-header": "value"},
            {
                "x-goog-api-client": _EXPECTED_BASE_HEADER,
                "user-agent": _EXPECTED_BASE_HEADER,
                "other-header": "value",
            },
        ),
    ],
)
def test_merge_tracking_headers(input_headers, expected_headers):
  """Test merge_tracking_headers with various inputs."""
  headers = _google_client_headers.merge_tracking_headers(input_headers)
  assert headers == expected_headers
