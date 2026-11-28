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

"""Tests for task-local eval run context lifetime and propagation."""

import asyncio
import contextlib

from google.adk.utils._eval_run_context import eval_run_context
from google.adk.utils._eval_run_context import get_eval_run_context
import pytest


@pytest.mark.parametrize("fails", [False, True])
def test_nested_eval_run_restores_outer_context(fails):
  """Leaving an inner run restores its enclosing run, including on failure."""
  with eval_run_context(
      app_name="app",
      eval_set_id="set",
      eval_case_id="outer",
      repetition_index=0,
  ) as outer:
    expected_error = (
        pytest.raises(RuntimeError, match="inference failed")
        if fails
        else contextlib.nullcontext()
    )
    with expected_error:
      with eval_run_context(
          app_name="app",
          eval_set_id="set",
          eval_case_id="inner",
          repetition_index=1,
      ) as inner:
        assert get_eval_run_context() is inner
        if fails:
          raise RuntimeError("inference failed")
    assert get_eval_run_context() is outer
  assert get_eval_run_context() is None


async def test_child_tasks_inherit_and_isolate_eval_runs():
  """Child tasks inherit their run; an inner run cannot change sibling tasks."""
  entered = asyncio.Event()
  release = asyncio.Event()

  async def child_with_inner_run():
    inherited = get_eval_run_context()
    with eval_run_context(
        app_name="app",
        eval_set_id="set",
        eval_case_id="inner",
        repetition_index=1,
    ) as inner:
      entered.set()
      await release.wait()
      observed_inner = get_eval_run_context()
    return inherited, inner, observed_inner, get_eval_run_context()

  async def sibling():
    await entered.wait()
    inherited = get_eval_run_context()
    release.set()
    return inherited

  with eval_run_context(
      app_name="app",
      eval_set_id="set",
      eval_case_id="outer",
      repetition_index=0,
  ) as outer:
    child_result, sibling_result = await asyncio.wait_for(
        asyncio.gather(child_with_inner_run(), sibling()), timeout=5
    )
    parent_result = get_eval_run_context()

  inherited, inner, observed_inner, restored = child_result
  assert inherited is outer
  assert observed_inner is inner
  assert restored is outer
  assert sibling_result is outer
  assert parent_result is outer
  assert get_eval_run_context() is None
