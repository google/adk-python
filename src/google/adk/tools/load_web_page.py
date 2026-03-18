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

"""Tool for web browse."""

from urllib.parse import urlparse

import requests


def load_web_page(url: str) -> str:
  """Fetches the content in the url and returns the text in it.

  Args:
      url (str): The url to browse.

  Returns:
      str: The text content of the url.
  """
  from bs4 import BeautifulSoup

  parsed = urlparse(url)
  if parsed.scheme not in ('http', 'https'):
    return (
        f'Invalid URL scheme: {parsed.scheme}. Only http and https are'
        ' supported.'
    )

  try:
    # Set allow_redirects=False to prevent SSRF attacks via redirection.
    response = requests.get(url, allow_redirects=False, timeout=10)
  except requests.exceptions.Timeout:
    return f'Request timed out when fetching url: {url}'
  except requests.exceptions.ConnectionError:
    return f'Connection error when fetching url: {url}'

  if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'lxml')
    text = soup.get_text(separator='\n', strip=True)
  else:
    text = f'Failed to fetch url: {url}'

  # Split the text into lines, filtering out very short lines
  # (e.g., single words or short subtitles)
  return '\n'.join(line for line in text.splitlines() if len(line.split()) > 3)
