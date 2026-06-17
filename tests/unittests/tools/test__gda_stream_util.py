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
import requests


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

  @mock.patch("requests.Session.post")
  def test_get_stream(self, mock_post):
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
    mock_post.return_value = MockResponse(stream_lines)
    messages = _gda_stream_util.get_stream("url", {}, {}, 10)
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


if __name__ == "__main__":
  unittest.main()
