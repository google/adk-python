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

"""Tool for web browse."""

import requests

from google.adk.utils import retry_on_network_error


@retry_on_network_error()
def load_web_page(url: str) -> str:
  """Fetches the content in the url and returns the text in it.

  This function automatically retries on network errors using exponential backoff.

  Args:
      url (str): The url to browse.

  Returns:
      str: The text content of the url.
      
  Raises:
      RetryError: If all retry attempts fail.
      requests.exceptions.HTTPError: For non-transient HTTP errors (4xx).
  """
  from bs4 import BeautifulSoup

  response = requests.get(url)
  
  # Raise for 5xx errors (will trigger retry) but not 4xx errors
  if 500 <= response.status_code < 600:
    response.raise_for_status()

  if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'lxml')
    text = soup.get_text(separator='\n', strip=True)
  else:
    # For 4xx errors, don't retry but return error message
    text = f'Failed to fetch url: {url} (Status: {response.status_code})'

  # Split the text into lines, filtering out very short lines
  # (e.g., single words or short subtitles)
  return '\n'.join(line for line in text.splitlines() if len(line.split()) > 3)
