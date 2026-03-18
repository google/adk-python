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

from unittest import mock

from google.adk.tools.load_web_page import load_web_page
import requests


def _mock_beautifulsoup(text_output):
  """Creates a mock BeautifulSoup class that returns the given text."""
  mock_soup = mock.Mock()
  mock_soup.get_text.return_value = text_output
  return mock.Mock(return_value=mock_soup)


class TestLoadWebPageUrlValidation:

  def test_rejects_file_scheme(self):
    result = load_web_page('file:///etc/passwd')
    assert 'Invalid URL scheme' in result
    assert 'file' in result

  def test_rejects_ftp_scheme(self):
    result = load_web_page('ftp://example.com/file.txt')
    assert 'Invalid URL scheme' in result
    assert 'ftp' in result

  def test_rejects_empty_scheme(self):
    result = load_web_page('not-a-url')
    assert 'Invalid URL scheme' in result

  @mock.patch(
      'bs4.BeautifulSoup', _mock_beautifulsoup('some words in a line here')
  )
  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_accepts_http_scheme(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b'<html><body><p>words</p></body></html>'
    mock_get.return_value = mock_response
    load_web_page('http://example.com')
    mock_get.assert_called_once()

  @mock.patch(
      'bs4.BeautifulSoup', _mock_beautifulsoup('some words in a line here')
  )
  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_accepts_https_scheme(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b'<html><body><p>words</p></body></html>'
    mock_get.return_value = mock_response
    load_web_page('https://example.com')
    mock_get.assert_called_once()


class TestLoadWebPageTimeout:

  @mock.patch(
      'bs4.BeautifulSoup', _mock_beautifulsoup('some words in a line here')
  )
  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_passes_timeout_to_requests(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b'<html><body><p>words</p></body></html>'
    mock_get.return_value = mock_response
    load_web_page('https://example.com')
    _, kwargs = mock_get.call_args
    assert kwargs['timeout'] == 10

  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_handles_timeout_error(self, mock_get):
    mock_get.side_effect = requests.exceptions.Timeout()
    result = load_web_page('https://slow-server.com')
    assert 'timed out' in result
    assert 'slow-server.com' in result

  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_handles_connection_error(self, mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError()
    result = load_web_page('https://unreachable.com')
    assert 'Connection error' in result
    assert 'unreachable.com' in result


class TestLoadWebPageResponse:

  @mock.patch(
      'bs4.BeautifulSoup',
      _mock_beautifulsoup('This is a line with more than three words'),
  )
  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_returns_text_on_success(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b'<html><body><p>text</p></body></html>'
    mock_get.return_value = mock_response
    result = load_web_page('https://example.com')
    assert 'more than three words' in result

  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_returns_error_on_non_200(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response
    result = load_web_page('https://example.com/missing')
    assert 'Failed to fetch url' in result

  @mock.patch(
      'bs4.BeautifulSoup',
      _mock_beautifulsoup(
          'Short\nThis line has enough words to pass the filter'
      ),
  )
  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_filters_short_lines(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b'<html><body><p>text</p></body></html>'
    mock_get.return_value = mock_response
    result = load_web_page('https://example.com')
    assert 'Short' not in result
    assert 'enough words to pass' in result
