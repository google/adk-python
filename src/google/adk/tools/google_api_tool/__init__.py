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
    'bigquery_toolset',
    'calendar_toolset',
    'gmail_toolset',
    'youtube_toolset',
    'slides_toolset',
    'sheets_toolset',
    'docs_toolset',
]

# Nothing is imported here automatically
# Each tool set will only be imported when accessed

_bigquery_toolset = None
_calendar_toolset = None
_gmail_toolset = None
_youtube_toolset = None
_slides_toolset = None
_sheets_toolset = None
_docs_toolset = None


def __getattr__(name):
  global _bigquery_toolset, _calendar_toolset, _gmail_toolset, _youtube_toolset, _slides_toolset, _sheets_toolset, _docs_toolset

  if name == 'bigquery_toolset':
    if _bigquery_toolset is None:
      from .google_api_toolsets import bigquery_toolset as bigquery

      _bigquery_toolset = bigquery
    return _bigquery_toolset

  if name == 'calendar_toolset':
    if _calendar_toolset is None:
      from .google_api_toolsets import calendar_toolset as calendar

      _calendar_toolset = calendar
    return _calendar_toolset

  if name == 'gmail_toolset':
    if _gmail_toolset is None:
      from .google_api_toolsets import gmail_toolset as gmail

      _gmail_toolset = gmail
    return _gmail_toolset

  if name == 'youtube_toolset':
    if _youtube_toolset is None:
      from .google_api_toolsets import youtube_toolset as youtube

      _youtube_toolset = youtube
    return _youtube_toolset

  if name == 'slides_toolset':
    if _slides_toolset is None:
      from .google_api_toolsets import slides_toolset as slides

      _slides_toolset = slides
    return _slides_toolset

  if name == 'sheets_toolset':
    if _sheets_toolset is None:
      from .google_api_toolsets import sheets_toolset as sheets

      _sheets_toolset = sheets
    return _sheets_toolset

  if name == 'docs_toolset':
    if _docs_toolset is None:
      from .google_api_toolsets import docs_toolset as docs

      _docs_toolset = docs
    return _docs_toolset
