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

from google.adk.runners import Runner


class ExecutorContext:
  """Context for the executor."""

  def __init__(
      self,
      app_name: str,
      user_id: str,
      session_id: str,
      runner: Runner,
  ):
    self._app_name = app_name
    self._user_id = user_id
    self._session_id = session_id
    self._runner = runner

  @property
  def app_name(self) -> str:
    return self._app_name

  @property
  def user_id(self) -> str:
    return self._user_id

  @property
  def session_id(self) -> str:
    return self._session_id

  @property
  def runner(self) -> Runner:
    return self._runner
