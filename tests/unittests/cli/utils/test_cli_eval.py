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

"""Unit tests for utilities in cli_eval."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest


@pytest.mark.parametrize(
    ("input_eval_set", "expected"),
    [
        pytest.param(
            r"C:\tmp\agent\eval.evalset.json",
            {r"C:\tmp\agent\eval.evalset.json": []},
            id="windows-backslash-path-without-selectors",
        ),
        pytest.param(
            r"C:\tmp\agent\eval.evalset.json:case1",
            {r"C:\tmp\agent\eval.evalset.json": ["case1"]},
            id="windows-backslash-path-with-one-selector",
        ),
        pytest.param(
            r"C:\tmp\agent\eval.evalset.json:case1,case2",
            {r"C:\tmp\agent\eval.evalset.json": ["case1", "case2"]},
            id="windows-backslash-path-with-multiple-selectors",
        ),
        pytest.param(
            "C:/tmp/agent/eval.evalset.json:case1",
            {"C:/tmp/agent/eval.evalset.json": ["case1"]},
            id="windows-forward-slash-path",
        ),
        pytest.param(
            r"d:\tmp\agent\eval.evalset.json:case1",
            {r"d:\tmp\agent\eval.evalset.json": ["case1"]},
            id="lowercase-windows-drive",
        ),
        pytest.param(
            "/tmp/agent/eval.evalset.json:case1,case2",
            {"/tmp/agent/eval.evalset.json": ["case1", "case2"]},
            id="posix-path-with-selectors",
        ),
        pytest.param(
            "my_eval_set:case1,case2",
            {"my_eval_set": ["case1", "case2"]},
            id="eval-set-id-with-selectors",
        ),
        pytest.param(
            "my_eval_set",
            {"my_eval_set": []},
            id="eval-set-id-without-selectors",
        ),
        pytest.param(
            "/tmp/agent/eval.evalset.json",
            {"/tmp/agent/eval.evalset.json": []},
            id="posix-path-without-selectors",
        ),
    ],
)
def test_parse_and_get_evals_to_run_parses_eval_set_and_selectors(
    input_eval_set: str, expected: dict[str, list[str]]
):
  """Eval-set paths and IDs retain their optional case selectors."""
  from google.adk.cli.cli_eval import parse_and_get_evals_to_run

  assert parse_and_get_evals_to_run([input_eval_set]) == expected


def test_get_eval_sets_manager_local(monkeypatch):
  mock_local_manager = mock.MagicMock()
  monkeypatch.setattr(
      "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager",
      lambda *a, **k: mock_local_manager,
  )
  from google.adk.cli.cli_eval import get_eval_sets_manager

  manager = get_eval_sets_manager(eval_storage_uri=None, agents_dir="some/dir")
  assert manager == mock_local_manager


def test_get_eval_sets_manager_gcs(monkeypatch):
  mock_gcs_manager = mock.MagicMock()
  mock_create_gcs = mock.MagicMock()
  mock_create_gcs.return_value = SimpleNamespace(
      eval_sets_manager=mock_gcs_manager
  )
  monkeypatch.setattr(
      "google.adk.cli.utils.evals.create_gcs_eval_managers_from_uri",
      mock_create_gcs,
  )
  from google.adk.cli.cli_eval import get_eval_sets_manager

  manager = get_eval_sets_manager(
      eval_storage_uri="gs://bucket", agents_dir="some/dir"
  )
  assert manager == mock_gcs_manager
  mock_create_gcs.assert_called_once_with("gs://bucket")
