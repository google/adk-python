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

"""Keyset pagination shared by the ``list_sessions`` implementations.

Sessions are listed in ascending ``(last_update_time, user_id, id)`` order. A
page token is an opaque cursor naming the last session of the previous page;
the next page holds the sessions that sort strictly after it. Unlike an offset,
a cursor stays correct when sessions are created, updated or deleted between
two requests: no session that exists throughout the whole walk is skipped or
repeated.
"""

from __future__ import annotations

import base64
import binascii
import bisect
import dataclasses
import json
import math
from typing import Optional
from typing import Protocol
from typing import TypeVar

_TOKEN_VERSION = 1


class _SessionLike(Protocol):
  """The attributes of a session that determine its position in a listing."""

  user_id: str
  id: str
  last_update_time: float


_S = TypeVar('_S', bound=_SessionLike)


@dataclasses.dataclass(frozen=True)
class PageCursor:
  """The sort key of the last session already returned."""

  last_update_time: float
  user_id: str
  session_id: str

  @property
  def key(self) -> tuple[float, str, str]:
    return (self.last_update_time, self.user_id, self.session_id)


def session_sort_key(session: _SessionLike) -> tuple[float, str, str]:
  """Returns the key ``list_sessions`` orders sessions by."""
  return (session.last_update_time, session.user_id, session.id)


def validate_page_size(page_size: Optional[int]) -> None:
  """Raises ValueError unless ``page_size`` is None or a positive integer."""
  if page_size is None:
    return
  if (
      isinstance(page_size, bool)
      or not isinstance(page_size, int)
      or page_size < 1
  ):
    raise ValueError(
        f'page_size must be a positive integer or None, got {page_size!r}.'
    )


def encode_page_token(
    *, app_name: str, user_id: Optional[str], last_session: _SessionLike
) -> str:
  """Builds the token that resumes a listing after ``last_session``."""
  payload = [
      _TOKEN_VERSION,
      app_name,
      user_id,
      last_session.last_update_time,
      last_session.user_id,
      last_session.id,
  ]
  raw = json.dumps(payload, separators=(',', ':')).encode('utf-8')
  return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')


def decode_page_token(
    page_token: Optional[str], *, app_name: str, user_id: Optional[str]
) -> Optional[PageCursor]:
  """Returns the cursor held by ``page_token``, or None if there is no token.

  Args:
    page_token: A token returned as ``next_page_token`` by a previous call.
    app_name: The ``app_name`` of the current call.
    user_id: The ``user_id`` filter of the current call.

  Raises:
    ValueError: If the token is malformed, or was issued for a different
      ``app_name`` or ``user_id`` than the current call.
  """
  if page_token is None:
    return None
  invalid = ValueError(
      'Invalid page_token: pass the next_page_token of a previous'
      ' list_sessions response unchanged.'
  )
  if not isinstance(page_token, str) or not page_token:
    raise invalid
  try:
    padded = page_token + '=' * (-len(page_token) % 4)
    payload = json.loads(
        base64.b64decode(padded, altchars=b'-_', validate=True).decode('utf-8')
    )
  except (binascii.Error, UnicodeError, ValueError) as e:
    raise invalid from e
  if not (
      isinstance(payload, list)
      and len(payload) == 6
      and payload[0] == _TOKEN_VERSION
      and isinstance(payload[1], str)
      and (payload[2] is None or isinstance(payload[2], str))
      and isinstance(payload[3], (int, float))
      and not isinstance(payload[3], bool)
      and math.isfinite(payload[3])
      and isinstance(payload[4], str)
      and isinstance(payload[5], str)
  ):
    raise invalid
  if payload[1] != app_name or payload[2] != user_id:
    raise ValueError(
        'Invalid page_token: it was issued for a different app_name or user_id.'
    )
  return PageCursor(float(payload[3]), payload[4], payload[5])


def paginate_sorted_sessions(
    sessions: list[_S],
    *,
    app_name: str,
    user_id: Optional[str],
    page_size: Optional[int],
    page_token: Optional[str],
) -> tuple[list[_S], Optional[str]]:
  """Slices one page out of fully loaded, sorted sessions.

  For backends that cannot page natively.

  Args:
    sessions: Every matching session, sorted by ``session_sort_key``.
    app_name: The ``app_name`` of the call.
    user_id: The ``user_id`` filter of the call.
    page_size: Maximum sessions to return, or None for all remaining.
    page_token: The cursor of the previous page, or None for the first page.

  Returns:
    The page, and the token of the next page (None on the last page).

  Raises:
    ValueError: If ``page_size`` or ``page_token`` is invalid.
  """
  validate_page_size(page_size)
  cursor = decode_page_token(page_token, app_name=app_name, user_id=user_id)
  start = 0
  if cursor is not None:
    start = bisect.bisect_right(sessions, cursor.key, key=session_sort_key)
  end = len(sessions) if page_size is None else start + page_size
  page = sessions[start:end]
  if end >= len(sessions) or not page:
    return page, None
  return page, encode_page_token(
      app_name=app_name, user_id=user_id, last_session=page[-1]
  )
