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

"""Utilities for ADK context management.

This module is for ADK internal use only.
Please do not rely on the implementation details.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any
from typing import AsyncGenerator


class Aclosing(AbstractAsyncContextManager):
  """Async context manager for safely finalizing an asynchronously cleaned-up
  resource such as an async generator, calling its ``aclose()`` method.
  Needed to correctly close contexts for OTel spans.
  See https://github.com/google/adk-python/issues/1670#issuecomment-3115891100.

  Based on
  https://docs.python.org/3/library/contextlib.html#contextlib.aclosing
  which is available in Python 3.10+.

  TODO: replace all occurrences with contextlib.aclosing once Python 3.9 is no
  longer supported.
  """

  def __init__(self, async_generator: AsyncGenerator[Any, None]):
    self.async_generator = async_generator

  async def __aenter__(self):
    return self.async_generator

  async def __aexit__(self, *exc_info):
    await self.async_generator.aclose()
