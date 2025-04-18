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

from .google_api_tool_set import GoogleApiToolSet

logger = logging.getLogger(__name__)

_bigquery_tool_set = None
_calendar_tool_set = None
_gmail_tool_set = None
_youtube_tool_set = None
_slides_tool_set = None
_sheets_tool_set = None
_docs_tool_set = None


def __getattr__(name):
  """This method dynamically loads and returns GoogleApiToolSet instances for

  various Google APIs. It uses a lazy loading approach, initializing each
  tool set only when it is first requested. This avoids unnecessary loading
  of tool sets that are not used in a given session.

  Args:
      name (str): The name of the tool set to retrieve (e.g.,
        "bigquery_tool_set").

  Returns:
      GoogleApiToolSet: The requested tool set instance.

  Raises:
      AttributeError: If the requested tool set name is not recognized.
  """
  global _bigquery_tool_set, _calendar_tool_set, _gmail_tool_set, _youtube_tool_set, _slides_tool_set, _sheets_tool_set, _docs_tool_set

  match name:
    case "bigquery_tool_set":
      if _bigquery_tool_set is None:
        _bigquery_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="bigquery",
            api_version="v2",
        )

      return _bigquery_tool_set

    case "calendar_tool_set":
      if _calendar_tool_set is None:
        _calendar_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="calendar",
            api_version="v3",
        )

      return _calendar_tool_set

    case "gmail_tool_set":
      if _gmail_tool_set is None:
        _gmail_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="gmail",
            api_version="v1",
        )

      return _gmail_tool_set

    case "youtube_tool_set":
      if _youtube_tool_set is None:
        _youtube_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="youtube",
            api_version="v3",
        )

      return _youtube_tool_set

    case "slides_tool_set":
      if _slides_tool_set is None:
        _slides_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="slides",
            api_version="v1",
        )

      return _slides_tool_set

    case "sheets_tool_set":
      if _sheets_tool_set is None:
        _sheets_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="sheets",
            api_version="v4",
        )

      return _sheets_tool_set

    case "docs_tool_set":
      if _docs_tool_set is None:
        _docs_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="docs",
            api_version="v1",
        )

      return _docs_tool_set
