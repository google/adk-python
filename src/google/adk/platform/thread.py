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

import threading
from typing import Callable
from typing import cast
from typing import Protocol


class _ThreadFactory(Protocol):
  """Optional platform-specific thread factory."""

  def create_thread(
      self,
      target: Callable[..., None],
      *args: object,
      **kwargs: object,
  ) -> threading.Thread:
    """Creates a thread."""


internal_thread: _ThreadFactory | None
try:
  from .internal import thread as _internal_thread
except ImportError:
  internal_thread = None
else:
  internal_thread = cast(_ThreadFactory, _internal_thread)


def create_thread(
    target: Callable[..., None], *args: object, **kwargs: object
) -> threading.Thread:
  """Creates a thread."""
  if internal_thread is not None:
    return internal_thread.create_thread(target, *args, **kwargs)
  return threading.Thread(target=target, args=args, kwargs=kwargs)
