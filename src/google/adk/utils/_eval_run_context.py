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

"""Identifies the eval run that the current task belongs to."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import contextvars
import dataclasses


@dataclasses.dataclass(frozen=True, eq=False)
class EvalRunContext:
  """One inference run of an eval case.

  Compared and hashed by identity: every run, including a repeat of the same
  eval case in a later eval invocation, gets its own instance, so data kept
  per run is never shared between runs.
  """

  app_name: str
  eval_set_id: str
  eval_case_id: str
  repetition_index: int


# Set by the eval service around the inference of one eval case run. Child
# tasks copy the context, so every turn and tool call of the run sees it. It
# stays in process: it is not sent in client headers or stored in a session.
_EVAL_RUN_CONTEXT: contextvars.ContextVar[EvalRunContext | None] = (
    contextvars.ContextVar("_EVAL_RUN_CONTEXT", default=None)
)


@contextmanager
def eval_run_context(
    *,
    app_name: str,
    eval_set_id: str,
    eval_case_id: str,
    repetition_index: int,
) -> Iterator[EvalRunContext]:
  """Runs the operation as one run of an eval case."""
  run = EvalRunContext(
      app_name=app_name,
      eval_set_id=eval_set_id,
      eval_case_id=eval_case_id,
      repetition_index=repetition_index,
  )
  token = _EVAL_RUN_CONTEXT.set(run)
  try:
    yield run
  finally:
    _EVAL_RUN_CONTEXT.reset(token)


def get_eval_run_context() -> EvalRunContext | None:
  """Returns the eval case run of the current task, or None outside an eval."""
  return _EVAL_RUN_CONTEXT.get()
