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

from __future__ import annotations

import socket
from unittest import mock

from google.adk.tools.load_web_page import load_web_page
import google.adk.tools.load_web_page as load_web_page_module
import requests


def _create_response(html: str) -> requests.Response:
  response = requests.Response()
  response.status_code = 200
  response._content = html.encode('utf-8')  # pylint: disable=protected-access
  response.url = 'https://example.com'
  return response


def _clear_proxy_env(monkeypatch):
  for proxy_variable in (
      'HTTP_PROXY',
      'HTTPS_PROXY',
      'http_proxy',
      'https_proxy',
      'ALL_PROXY',
      'all_proxy',
      'NO_PROXY',
      'no_proxy',
  ):
    monkeypatch.delenv(proxy_variable, raising=False)


def test_load_web_page_blocks_file_scheme_urls(monkeypatch):
  _clear_proxy_env(monkeypatch)
  mock_get = mock.Mock()
  monkeypatch.setattr(load_web_page_module.requests, 'get', mock_get)
  mock_send = mock.Mock()
  monkeypatch.setattr(load_web_page_module.HTTPAdapter, 'send', mock_send)

  result = load_web_page('file:///etc/passwd')

  assert result == 'Failed to fetch url: file:///etc/passwd'
  mock_get.assert_not_called()
  mock_send.assert_not_called()


def test_load_web_page_blocks_loopback_ip_urls(monkeypatch):
  _clear_proxy_env(monkeypatch)
  mock_get = mock.Mock()
  monkeypatch.setattr(load_web_page_module.requests, 'get', mock_get)
  mock_send = mock.Mock()
  monkeypatch.setattr(load_web_page_module.HTTPAdapter, 'send', mock_send)

  result = load_web_page(
      'http://127.0.0.1:19876/latest/meta-data/iam/security-credentials/'
  )

  assert (
      result
      == 'Failed to fetch url:'
      ' http://127.0.0.1:19876/latest/meta-data/iam/security-credentials/'
  )
  mock_get.assert_not_called()
  mock_send.assert_not_called()


def test_load_web_page_blocks_shared_address_space_urls(monkeypatch):
  _clear_proxy_env(monkeypatch)
  mock_get = mock.Mock()
  monkeypatch.setattr(load_web_page_module.requests, 'get', mock_get)
  mock_send = mock.Mock()
  monkeypatch.setattr(load_web_page_module.HTTPAdapter, 'send', mock_send)

  result = load_web_page('http://100.64.0.1/internal')

  assert result == 'Failed to fetch url: http://100.64.0.1/internal'
  mock_get.assert_not_called()
  mock_send.assert_not_called()


def test_load_web_page_blocks_private_hostname_targets(monkeypatch):
  _clear_proxy_env(monkeypatch)
  monkeypatch.setattr(
      load_web_page_module.socket,
      'getaddrinfo',
      mock.Mock(
          return_value=[(
              socket.AF_INET,
              socket.SOCK_STREAM,
              socket.IPPROTO_TCP,
              '',
              ('169.254.169.254', 0),
          )]
      ),
  )
  mock_get = mock.Mock()
  monkeypatch.setattr(load_web_page_module.requests, 'get', mock_get)
  mock_send = mock.Mock()
  monkeypatch.setattr(load_web_page_module.HTTPAdapter, 'send', mock_send)

  result = load_web_page('http://metadata.google.internal/computeMetadata/v1/')

  assert (
      result
      == 'Failed to fetch url:'
      ' http://metadata.google.internal/computeMetadata/v1/'
  )
  mock_get.assert_not_called()
  mock_send.assert_not_called()


def test_load_web_page_uses_proxy_for_unresolved_public_hostnames(monkeypatch):
  monkeypatch.setenv('HTTPS_PROXY', 'http://proxy.example.test:8080')
  monkeypatch.setenv('NO_PROXY', '')
  monkeypatch.setattr(
      load_web_page_module.socket,
      'getaddrinfo',
      mock.Mock(side_effect=AssertionError('unexpected local DNS lookup')),
  )
  monkeypatch.setattr(
      'bs4.BeautifulSoup',
      mock.Mock(
          return_value=mock.Mock(
              get_text=mock.Mock(
                  return_value='This page has enough words to keep.'
              )
          )
      ),
  )
  mock_get = mock.Mock(
      return_value=_create_response(
          '<html><body><p>This page has enough words to keep.</p></body></html>'
      )
  )
  monkeypatch.setattr(load_web_page_module.requests, 'get', mock_get)
  mock_send = mock.Mock()
  monkeypatch.setattr(load_web_page_module.HTTPAdapter, 'send', mock_send)

  result = load_web_page('https://does-not-resolve.invalid')

  assert result == 'This page has enough words to keep.'
  mock_get.assert_called_once_with(
      'https://does-not-resolve.invalid', allow_redirects=False
  )
  mock_send.assert_not_called()


def test_load_web_page_fetches_public_urls_by_pinning_the_resolved_ip(
    monkeypatch,
):
  _clear_proxy_env(monkeypatch)
  monkeypatch.setattr(
      load_web_page_module.socket,
      'getaddrinfo',
      mock.Mock(
          return_value=[(
              socket.AF_INET,
              socket.SOCK_STREAM,
              socket.IPPROTO_TCP,
              '',
              ('93.184.216.34', 0),
          )]
      ),
  )
  mock_soup = mock.Mock()
  mock_soup.get_text.return_value = 'This page has enough words to keep.\ntiny'
  monkeypatch.setattr('bs4.BeautifulSoup', mock.Mock(return_value=mock_soup))
  captured_request: dict[str, object] = {}

  def _send(
      self,
      request,
      stream=False,
      timeout=None,
      verify=True,
      cert=None,
      proxies=None,
  ):
    del self, stream, timeout, verify, cert
    captured_request['url'] = request.url
    captured_request['host_header'] = request.headers['Host']
    captured_request['proxies'] = proxies
    return _create_response(
        '<html><body><p>This page has enough words to keep.</p>'
        '<p>tiny</p></body></html>'
    )

  monkeypatch.setattr(load_web_page_module.HTTPAdapter, 'send', _send)
  mock_get = mock.Mock()
  monkeypatch.setattr(load_web_page_module.requests, 'get', mock_get)

  result = load_web_page('https://example.com/search?q=adk')

  assert result == 'This page has enough words to keep.'
  assert captured_request['url'] == 'https://93.184.216.34/search?q=adk'
  assert captured_request['host_header'] == 'example.com'
  assert not captured_request['proxies']
  mock_get.assert_not_called()


def test_load_web_page_tries_another_resolved_address_after_connect_error(
    monkeypatch,
):
  _clear_proxy_env(monkeypatch)
  monkeypatch.setattr(
      load_web_page_module.socket,
      'getaddrinfo',
      mock.Mock(
          return_value=[
              (
                  socket.AF_INET,
                  socket.SOCK_STREAM,
                  socket.IPPROTO_TCP,
                  '',
                  ('93.184.216.34', 0),
              ),
              (
                  socket.AF_INET,
                  socket.SOCK_STREAM,
                  socket.IPPROTO_TCP,
                  '',
                  ('93.184.216.35', 0),
              ),
          ]
      ),
  )
  monkeypatch.setattr(
      'bs4.BeautifulSoup',
      mock.Mock(
          return_value=mock.Mock(
              get_text=mock.Mock(
                  return_value='This page has enough words to keep.'
              )
          )
      ),
  )
  captured_urls: list[str] = []

  def _send(
      self,
      request,
      stream=False,
      timeout=None,
      verify=True,
      cert=None,
      proxies=None,
  ):
    del self, stream, timeout, verify, cert, proxies
    captured_urls.append(request.url)
    if len(captured_urls) == 1:
      raise requests.ConnectionError('first address failed')
    return _create_response(
        '<html><body><p>This page has enough words to keep.</p></body></html>'
    )

  monkeypatch.setattr(load_web_page_module.HTTPAdapter, 'send', _send)
  mock_get = mock.Mock()
  monkeypatch.setattr(load_web_page_module.requests, 'get', mock_get)

  result = load_web_page('https://example.com')

  assert result == 'This page has enough words to keep.'
  assert captured_urls == [
      'https://93.184.216.34',
      'https://93.184.216.35',
  ]
  mock_get.assert_not_called()
