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


import logging

from .google_api_toolset import GoogleApiToolset

logger = logging.getLogger(__name__)

_bigquery_toolset = None
_calendar_toolset = None
_gmail_toolset = None
_youtube_toolset = None
_slides_toolset = None
_sheets_toolset = None
_docs_toolset = None


def __getattr__(name):
  """This method dynamically loads and returns GoogleApiToolSet instances for

  various Google APIs. It uses a lazy loading approach, initializing each
  tool set only when it is first requested. This avoids unnecessary loading
  of tool sets that are not used in a given session.

  Args:
      name (str): The name of the tool set to retrieve (e.g.,
        "bigquery_toolset").

  Returns:
      GoogleApiToolSet: The requested tool set instance.

  Raises:
      AttributeError: If the requested tool set name is not recognized.
  """
  global _bigquery_toolset, _calendar_toolset, _gmail_toolset, _youtube_toolset, _slides_toolset, _sheets_toolset, _docs_toolset

  if name == "bigquery_toolset":
    if _bigquery_toolset is None:
      _bigquery_toolset = GoogleApiToolset.load_toolset(
          api_name="bigquery",
          api_version="v2",
      )

    return _bigquery_toolset

  if name == "calendar_toolset":
    if _calendar_toolset is None:
      _calendar_toolset = GoogleApiToolset.load_toolset(
          api_name="calendar",
          api_version="v3",
      )

    return _calendar_toolset

  if name == "gmail_toolset":
    if _gmail_toolset is None:
      _gmail_toolset = GoogleApiToolset.load_toolset(
          api_name="gmail",
          api_version="v1",
      )

    return _gmail_toolset

  if name == "youtube_toolset":
    if _youtube_toolset is None:
      _youtube_toolset = GoogleApiToolset.load_toolset(
          api_name="youtube",
          api_version="v3",
      )

    return _youtube_toolset

  if name == "slides_toolset":
    if _slides_toolset is None:
      _slides_toolset = GoogleApiToolset.load_toolset(
          api_name="slides",
          api_version="v1",
      )

    return _slides_toolset

  if name == "sheets_toolset":
    if _sheets_toolset is None:
      _sheets_toolset = GoogleApiToolset.load_toolset(
          api_name="sheets",
          api_version="v4",
      )

    return _sheets_toolset

  if name == "docs_toolset":
    if _docs_toolset is None:
      _docs_toolset = GoogleApiToolset.load_toolset(
          api_name="docs",
          api_version="v1",
      )

    return _docs_toolset
