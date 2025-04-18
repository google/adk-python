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
__all__ = [
    'bigquery_tool_set',
    'calendar_tool_set',
    'gmail_tool_set',
    'youtube_tool_set',
    'slides_tool_set',
    'sheets_tool_set',
    'docs_tool_set',
]

# Nothing is imported here automatically
# Each tool set will only be imported when accessed

_bigquery_tool_set = None
_calendar_tool_set = None
_gmail_tool_set = None
_youtube_tool_set = None
_slides_tool_set = None
_sheets_tool_set = None
_docs_tool_set = None


def __getattr__(name):
  global _bigquery_tool_set, _calendar_tool_set, _gmail_tool_set, _youtube_tool_set, _slides_tool_set, _sheets_tool_set, _docs_tool_set

  match name:
    case 'bigquery_tool_set':
      if _bigquery_tool_set is None:
        from .google_api_tool_sets import bigquery_tool_set as bigquery

        _bigquery_tool_set = bigquery
      return _bigquery_tool_set

    case 'calendar_tool_set':
      if _calendar_tool_set is None:
        from .google_api_tool_sets import calendar_tool_set as calendar

        _calendar_tool_set = calendar
      return _calendar_tool_set

    case 'gmail_tool_set':
      if _gmail_tool_set is None:
        from .google_api_tool_sets import gmail_tool_set as gmail

        _gmail_tool_set = gmail
      return _gmail_tool_set

    case 'youtube_tool_set':
      if _youtube_tool_set is None:
        from .google_api_tool_sets import youtube_tool_set as youtube

        _youtube_tool_set = youtube
      return _youtube_tool_set

    case 'slides_tool_set':
      if _slides_tool_set is None:
        from .google_api_tool_sets import slides_tool_set as slides

        _slides_tool_set = slides
      return _slides_tool_set

    case 'sheets_tool_set':
      if _sheets_tool_set is None:
        from .google_api_tool_sets import sheets_tool_set as sheets

        _sheets_tool_set = sheets
      return _sheets_tool_set

    case 'docs_tool_set':
      if _docs_tool_set is None:
        from .google_api_tool_sets import docs_tool_set as docs

        _docs_tool_set = docs
      return _docs_tool_set
