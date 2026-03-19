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

"""Tests for utilities in eval."""

import os
from unittest import mock

from google.adk.cli.utils import evals
from google.adk.evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
from google.adk.evaluation.gcs_eval_sets_manager import GcsEvalSetsManager
import pytest


@mock.patch.dict(os.environ, {'GOOGLE_CLOUD_PROJECT': 'test-project'})
@mock.patch(
    'google.adk.cli.utils.evals.GcsEvalSetResultsManager',
    autospec=True,
)
@mock.patch(
    'google.adk.cli.utils.evals.GcsEvalSetsManager',
    autospec=True,
)
def test_create_gcs_eval_managers_from_uri_success(
    mock_gcs_eval_sets_manager, mock_gcs_eval_set_results_manager
):
  mock_gcs_eval_sets_manager.return_value = mock.MagicMock(
      spec=GcsEvalSetsManager
  )
  mock_gcs_eval_set_results_manager.return_value = mock.MagicMock(
      spec=GcsEvalSetResultsManager
  )

  managers = evals.create_gcs_eval_managers_from_uri('gs://test-bucket')

  assert managers is not None
  mock_gcs_eval_sets_manager.assert_called_once_with(
      bucket_name='test-bucket', project='test-project'
  )
  mock_gcs_eval_set_results_manager.assert_called_once_with(
      bucket_name='test-bucket', project='test-project'
  )
  assert managers.eval_sets_manager == mock_gcs_eval_sets_manager.return_value
  assert (
      managers.eval_set_results_manager
      == mock_gcs_eval_set_results_manager.return_value
  )


def test_create_gcs_eval_managers_from_uri_failure():
  with pytest.raises(ValueError):
    evals.create_gcs_eval_managers_from_uri('unsupported-uri')
