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
import unittest
from unittest import mock

from google.adk.tools import _gda_stream_util


class MockResponse:

  def __init__(self, lines):
    self._lines = lines

  def iter_lines(self):
    return iter(self._lines)

  def raise_for_status(self):
    pass

  def __enter__(self):
    return self

  def __exit__(self, *args):
    pass


class GdaStreamUtilTest(unittest.TestCase):

  def test_extract_data_result_success(self):
    msg = {
        "systemMessage": {"data": {"result": {"data": [1, 2], "schema": {}}}}
    }
    self.assertEqual(
        _gda_stream_util._extract_data_result(msg),
        {"data": [1, 2], "schema": {}},
    )

  def test_extract_data_result_failure(self):
    self.assertIsNone(_gda_stream_util._extract_data_result({}))
    self.assertIsNone(
        _gda_stream_util._extract_data_result({"systemMessage": None})
    )
    self.assertIsNone(
        _gda_stream_util._extract_data_result({"systemMessage": {"data": None}})
    )
    self.assertIsNone(
        _gda_stream_util._extract_data_result(
            {"systemMessage": {"data": {"result": None}}}
        )
    )
    self.assertIsNone(
        _gda_stream_util._extract_data_result(
            {"systemMessage": {"data": {"result": {"no_data": 1}}}}
        )
    )

  def test_format_data_retrieved_simple(self):
    result = {
        "data": [{"col1": "val1", "col2": 10}],
        "schema": {"fields": [{"name": "col1"}, {"name": "col2"}]},
    }
    formatted = _gda_stream_util._format_data_retrieved(result, 10)
    self.assertEqual(
        formatted,
        {
            "Data Retrieved": {
                "headers": ["col1", "col2"],
                "rows": [["val1", 10]],
                "summary": "Showing all 1 rows.",
            }
        },
    )

  def test_format_data_retrieved_truncation(self):
    result = {
        "data": [{"col1": f"val{i}"} for i in range(5)],
        "schema": {"fields": [{"name": "col1"}]},
    }
    formatted = _gda_stream_util._format_data_retrieved(result, 2)
    self.assertEqual(
        formatted,
        {
            "Data Retrieved": {
                "headers": ["col1"],
                "rows": [["val0"], ["val1"]],
                "summary": "Showing the first 2 of 5 total rows.",
            }
        },
    )

  def test_format_data_retrieved_missing_schema(self):
    result = {"data": [{"col1": "val1"}], "schema": None}
    formatted = _gda_stream_util._format_data_retrieved(result, 10)
    self.assertEqual(
        formatted,
        {
            "Data Retrieved": {
                "headers": ["col1"],
                "rows": [["val1"]],
                "summary": "Showing all 1 rows.",
            }
        },
    )

  def test_get_stream(self):
    stream_lines = [
        b"[{",
        b'"systemMessage": {"text": "msg1"}',
        b"}",
        b",",
        b"{",
        (
            b'"systemMessage": { "data": { "result": { "data": [{"a":1}],'
            b' "schema": {"fields":[{"name":"a"}]}}}}'
        ),
        b"}",
        b",",
        b"{",
        (
            b'"systemMessage": { "data": { "result": { "data": [{"b":2}],'
            b' "schema": {"fields":[{"name":"b"}]}}}}'
        ),
        b"}",
        b",",
        b"{",
        b'"systemMessage": {"text": "msg4"}',
        b"}]",
    ]
    mock_session = mock.MagicMock()
    mock_session.post.return_value = MockResponse(stream_lines)
    messages = _gda_stream_util.get_stream(mock_session, "url", {}, {}, 10)
    self.assertEqual(len(messages), 4)
    self.assertEqual(messages[0], {"text": "msg1"})
    self.assertEqual(
        messages[1], {"Data Retrieved": "Intermediate result omitted"}
    )
    self.assertEqual(
        messages[2],
        {
            "Data Retrieved": {
                "headers": ["b"],
                "rows": [[2]],
                "summary": "Showing all 1 rows.",
            }
        },
    )
    self.assertEqual(messages[3], {"text": "msg4"})

  @mock.patch.object(
      _gda_stream_util._mtls_utils, "get_api_endpoint", autospec=True
  )
  @mock.patch.object(
      _gda_stream_util.mtls, "has_default_client_cert_source", autospec=True
  )
  @mock.patch.object(
      _gda_stream_util.auth_requests, "AuthorizedSession", autospec=True
  )
  def test_get_gda_session_use_mtls_and_cert(
      self, mock_authorized_session, mock_use_client_cert, mock_get_api_endpoint
  ):
    mock_session = mock.MagicMock()
    mock_authorized_session.return_value = mock_session
    mock_use_client_cert.return_value = True
    mock_get_api_endpoint.return_value = (
        "https://geminidataanalytics.mtls.googleapis.com"
    )

    creds = mock.MagicMock()
    session, endpoint = _gda_stream_util.get_gda_session(creds)

    self.assertEqual(session, mock_session)
    self.assertEqual(
        endpoint, "https://geminidataanalytics.mtls.googleapis.com"
    )
    mock_session.configure_mtls_channel.assert_called_once()
    mock_get_api_endpoint.assert_called_once_with(
        location="",
        default_template="https://geminidataanalytics.googleapis.com",
        mtls_template="https://geminidataanalytics.mtls.googleapis.com",
    )

  @mock.patch.object(
      _gda_stream_util._mtls_utils, "get_api_endpoint", autospec=True
  )
  @mock.patch.object(
      _gda_stream_util.mtls, "has_default_client_cert_source", autospec=True
  )
  @mock.patch.object(
      _gda_stream_util.auth_requests, "AuthorizedSession", autospec=True
  )
  def test_get_gda_session_use_mtls_no_cert(
      self, mock_authorized_session, mock_use_client_cert, mock_get_api_endpoint
  ):
    mock_session = mock.MagicMock()
    mock_authorized_session.return_value = mock_session
    mock_use_client_cert.return_value = False
    mock_get_api_endpoint.return_value = (
        "https://geminidataanalytics.mtls.googleapis.com"
    )

    creds = mock.MagicMock()
    with self.assertRaises(ValueError) as context:
      _gda_stream_util.get_gda_session(creds)

    self.assertIn(
        "mTLS endpoint is selected, but client certificate is not provisioned",
        str(context.exception),
    )

  @mock.patch.object(
      _gda_stream_util._mtls_utils, "get_api_endpoint", autospec=True
  )
  @mock.patch.object(
      _gda_stream_util.mtls, "has_default_client_cert_source", autospec=True
  )
  @mock.patch.object(
      _gda_stream_util.auth_requests, "AuthorizedSession", autospec=True
  )
  def test_get_gda_session_regular_endpoint(
      self, mock_authorized_session, mock_use_client_cert, mock_get_api_endpoint
  ):
    mock_session = mock.MagicMock()
    mock_authorized_session.return_value = mock_session
    mock_use_client_cert.return_value = True
    mock_get_api_endpoint.return_value = (
        "https://geminidataanalytics.googleapis.com"
    )

    creds = mock.MagicMock()
    session, endpoint = _gda_stream_util.get_gda_session(creds)

    self.assertEqual(session, mock_session)
    self.assertEqual(endpoint, "https://geminidataanalytics.googleapis.com")
    mock_session.configure_mtls_channel.assert_not_called()


if __name__ == "__main__":
  unittest.main()
