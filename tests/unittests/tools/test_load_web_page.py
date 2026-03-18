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


def _mock_beautiful_soup(text='This is a test paragraph with enough words'):
  """Create a mock BeautifulSoup class that returns the given text."""
  mock_soup = mock.Mock()
  mock_soup.get_text.return_value = text
  mock_cls = mock.Mock(return_value=mock_soup)
  return mock_cls


class TestLoadWebPage:

  def test_invalid_scheme_file(self):
    result = load_web_page('file:///etc/passwd')
    assert 'Invalid URL scheme' in result
    assert 'file' in result

  def test_invalid_scheme_ftp(self):
    result = load_web_page('ftp://example.com/file')
    assert 'Invalid URL scheme' in result
    assert 'ftp' in result

  def test_invalid_scheme_empty(self):
    result = load_web_page('not-a-url')
    assert 'Invalid URL scheme' in result

  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_timeout_returns_error_message(self, mock_get):
    mock_get.side_effect = requests.exceptions.Timeout()
    result = load_web_page('https://example.com')
    assert 'timed out' in result

  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_connection_error_returns_error_message(self, mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError()
    result = load_web_page('https://example.com')
    assert 'Connection error' in result

  @mock.patch('builtins.__import__')
  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_successful_request(self, mock_get, mock_import):
    mock_soup_instance = mock.Mock()
    mock_soup_instance.get_text.return_value = (
        'This is a test paragraph with enough words'
    )
    mock_bs_module = mock.Mock()
    mock_bs_module.BeautifulSoup.return_value = mock_soup_instance

    original_import = (
        __builtins__.__import__
        if hasattr(__builtins__, '__import__')
        else __import__
    )

    def side_effect(name, *args, **kwargs):
      if name == 'bs4':
        return mock_bs_module
      return original_import(name, *args, **kwargs)

    mock_import.side_effect = side_effect

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = (
        b'<html><body><p>This is a test paragraph with enough words</p>'
        b'</body></html>'
    )
    mock_get.return_value = mock_response

    result = load_web_page('https://example.com')

    mock_get.assert_called_once_with(
        'https://example.com', allow_redirects=False, timeout=10
    )
    assert 'test paragraph' in result

  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_failed_request_non_200(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    result = load_web_page('https://example.com/missing')
    assert 'Failed to fetch url' in result

  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_timeout_parameter_passed(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    load_web_page('http://example.com')

    _, kwargs = mock_get.call_args
    assert kwargs['timeout'] == 10

  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_allow_redirects_false(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    load_web_page('https://example.com')

    _, kwargs = mock_get.call_args
    assert kwargs['allow_redirects'] is False

  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_http_scheme_accepted(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    result = load_web_page('http://example.com')
    assert 'Invalid URL scheme' not in result
    mock_get.assert_called_once()

  @mock.patch('google.adk.tools.load_web_page.requests.get')
  def test_https_scheme_accepted(self, mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    result = load_web_page('https://example.com')
    assert 'Invalid URL scheme' not in result
    mock_get.assert_called_once()
