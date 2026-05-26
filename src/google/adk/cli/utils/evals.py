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

import os
from typing import Any, TYPE_CHECKING

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict

from ...evaluation.eval_case import Invocation
from ...evaluation.evaluation_generator import EvaluationGenerator
from ...sessions.session import Session

if TYPE_CHECKING:
  from ...evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
  from ...evaluation.gcs_eval_sets_manager import GcsEvalSetsManager


class GcsEvalManagers(BaseModel):
  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
      arbitrary_types_allowed=True,
  )

  eval_sets_manager: 'GcsEvalSetsManager'

  eval_set_results_manager: 'GcsEvalSetResultsManager'


def convert_session_to_eval_invocations(session: Session) -> list[Invocation]:
  """Converts a session data into a list of Invocation.

  Args:
      session: The session that should be converted.

  Returns:
      list: A list of invocation.
  """
  events = session.events if session and session.events else []
  return EvaluationGenerator.convert_events_to_eval_invocations(events)


def create_gcs_eval_managers_from_uri(
    eval_storage_uri: str,
) -> GcsEvalManagers:
  """Creates GcsEvalManagers from eval_storage_uri.

  Args:
      eval_storage_uri: The evals storage URI to use. Supported URIs:
        gs://<bucket name>. If a path is provided, the bucket will be extracted.

  Returns:
      GcsEvalManagers: The GcsEvalManagers object.

  Raises:
      ValueError: If the eval_storage_uri is not supported.
      RuntimeError: If GCP optional dependencies are missing.
  """
  if eval_storage_uri.startswith('gs://'):
    try:
      from ...evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
      from ...evaluation.gcs_eval_sets_manager import GcsEvalSetsManager
    except ImportError as e:
      raise RuntimeError(
          'GCS evaluation managers require Google Cloud optional dependencies.\n'
          'Please install them using: pip install google-adk[gcp]\n'
          'Or: pip install google-cloud-storage>=2.18'
      ) from e

    gcs_bucket = eval_storage_uri.split('://')[1]
    eval_sets_manager = GcsEvalSetsManager(
        bucket_name=gcs_bucket, project=os.environ['GOOGLE_CLOUD_PROJECT']
    )
    eval_set_results_manager = GcsEvalSetResultsManager(
        bucket_name=gcs_bucket, project=os.environ['GOOGLE_CLOUD_PROJECT']
    )
    return GcsEvalManagers(
        eval_sets_manager=eval_sets_manager,
        eval_set_results_manager=eval_set_results_manager,
    )
  else:
    raise ValueError(
        f'Unsupported evals storage URI: {eval_storage_uri}. Supported URIs:'
        ' gs://<bucket name>'
    )
