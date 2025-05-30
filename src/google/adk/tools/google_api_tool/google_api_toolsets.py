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
from typing import List
from typing import Optional
from typing import Union

from ..base_toolset import ToolPredicate
from .google_api_toolset import GoogleApiToolset

logger = logging.getLogger("google_adk." + __name__)


class BigQueryToolset(GoogleApiToolset):
  """Auto-generated Bigquery toolset based on Google BigQuery API v2 spec exposed by Google API discovery API"""

  def __init__(
      self,
      client_id: str = None,
      client_secret: str = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    super().__init__("bigquery", "v2", client_id, client_secret, tool_filter)


class CalendarToolset(GoogleApiToolset):
  """Auto-generated Calendar toolset based on Google Calendar API v3 spec exposed by Google API discovery API"""

  def __init__(
      self,
      client_id: str = None,
      client_secret: str = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    super().__init__("calendar", "v3", client_id, client_secret, tool_filter)


class GmailToolset(GoogleApiToolset):
  """Auto-generated Gmail toolset based on Google Gmail API v1 spec exposed by Google API discovery API"""

  def __init__(
      self,
      client_id: str = None,
      client_secret: str = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    super().__init__("gmail", "v1", client_id, client_secret, tool_filter)


class YoutubeToolset(GoogleApiToolset):
  """Auto-generated Youtube toolset based on Youtube API v3 spec exposed by Google API discovery API"""

  def __init__(
      self,
      client_id: str = None,
      client_secret: str = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    super().__init__("youtube", "v3", client_id, client_secret, tool_filter)


class SlidesToolset(GoogleApiToolset):
  """Auto-generated Slides toolset based on Google Slides API v1 spec exposed by Google API discovery API"""

  def __init__(
      self,
      client_id: str = None,
      client_secret: str = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    super().__init__("slides", "v1", client_id, client_secret, tool_filter)


class SheetsToolset(GoogleApiToolset):
  """Auto-generated Sheets toolset based on Google Sheets API v4 spec exposed by Google API discovery API"""

  def __init__(
      self,
      client_id: str = None,
      client_secret: str = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    super().__init__("sheets", "v4", client_id, client_secret, tool_filter)


class DocsToolset(GoogleApiToolset):
  """Auto-generated Docs toolset based on Google Docs API v1 spec exposed by Google API discovery API"""

  def __init__(
      self,
      client_id: str = None,
      client_secret: str = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    super().__init__("docs", "v1", client_id, client_secret, tool_filter)
