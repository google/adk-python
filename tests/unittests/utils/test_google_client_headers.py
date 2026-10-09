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

from google.adk import version
from google.adk.utils import _google_client_headers
import httpx
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


@pytest.mark.parametrize(
    "user_agent_key, api_client_key",
    [
        ("User-Agent", "X-Goog-Api-Client"),
        ("USER-AGENT", "X-GOOG-API-CLIENT"),
    ],
)
def test_merge_tracking_headers_merges_case_insensitively(
    user_agent_key, api_client_key
):
  """Each tracking header reaches the transport once with custom tokens kept."""
  input_headers = {
      user_agent_key: "custom-client/1",
      api_client_key: "custom-sdk/1",
      "X-Custom": "value",
  }

  headers = _google_client_headers.merge_tracking_headers(input_headers)
  request = httpx.Request("GET", "https://example.test", headers=headers)

  assert request.headers.get_list("user-agent") == [
      f"{_EXPECTED_BASE_HEADER} custom-client/1"
  ]
  assert request.headers.get_list("x-goog-api-client") == [
      f"{_EXPECTED_BASE_HEADER} custom-sdk/1"
  ]
  assert headers["X-Custom"] == "value"
  assert input_headers == {
      user_agent_key: "custom-client/1",
      api_client_key: "custom-sdk/1",
      "X-Custom": "value",
  }


def test_merge_tracking_headers_preserves_all_case_aliases():
  """Case aliases merge into one transport field without losing custom tokens."""
  input_headers = {
      "User-Agent": f"first-client/1 shared/1 {_EXPECTED_BASE_HEADER}",
      "user-agent": "second-client/1 shared/1",
      "X-Goog-Api-Client": "first-sdk/1 shared/1",
      "x-goog-api-client": "second-sdk/1 shared/1",
      "X-Custom": "value",
  }
  original_headers = input_headers.copy()

  headers = _google_client_headers.merge_tracking_headers(input_headers)
  request = httpx.Request("GET", "https://example.test", headers=headers)

  assert request.headers.get_list("user-agent") == [
      f"{_EXPECTED_BASE_HEADER} first-client/1 shared/1 second-client/1"
  ]
  assert request.headers.get_list("x-goog-api-client") == [
      f"{_EXPECTED_BASE_HEADER} first-sdk/1 shared/1 second-sdk/1"
  ]
  assert headers["X-Custom"] == "value"
  assert input_headers == original_headers


def test_get_tracking_http_options():
  """get_tracking_http_options returns HttpOptions carrying tracking headers."""
  http_options = _google_client_headers.get_tracking_http_options()
  assert http_options.headers == {
      "x-goog-api-client": _EXPECTED_BASE_HEADER,
      "user-agent": _EXPECTED_BASE_HEADER,
  }


def test_get_tracking_headers_with_framework_label():
  """framework_label flows into both tracking header values."""
  expected = (
      f"google-adk/{version.__version__}+managed_agent"
      f" gl-python/{sys.version.split()[0]}"
  )
  headers = _google_client_headers.get_tracking_headers(
      framework_label="managed_agent"
  )
  assert headers == {
      "x-goog-api-client": expected,
      "user-agent": expected,
  }


def test_merge_tracking_headers_with_framework_label():
  """framework_label flows into the merged tracking header values."""
  expected = (
      f"google-adk/{version.__version__}+managed_agent"
      f" gl-python/{sys.version.split()[0]}"
  )
  headers = _google_client_headers.merge_tracking_headers(
      None, framework_label="managed_agent"
  )
  assert headers == {
      "x-goog-api-client": expected,
      "user-agent": expected,
  }


def test_merge_tracking_headers_with_framework_label_preserves_custom_headers():
  """The suffix is applied while unrelated custom headers pass through."""
  expected = (
      f"google-adk/{version.__version__}+managed_agent"
      f" gl-python/{sys.version.split()[0]}"
  )
  headers = _google_client_headers.merge_tracking_headers(
      {"x-custom": "v"}, framework_label="managed_agent"
  )
  assert headers == {
      "x-goog-api-client": expected,
      "user-agent": expected,
      "x-custom": "v",
  }
