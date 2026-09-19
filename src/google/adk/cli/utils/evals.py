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

import logging
import os
from pathlib import Path
from typing import NamedTuple
from typing import TYPE_CHECKING
from urllib.parse import unquote
from urllib.parse import urlparse

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict

from ...evaluation.eval_case import Invocation
from ...evaluation.evaluation_generator import EvaluationGenerator
from ...sessions.session import Session

if TYPE_CHECKING:
  from ...evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
  from ...evaluation.gcs_eval_sets_manager import GcsEvalSetsManager

logger = logging.getLogger('google_adk.' + __name__)

ADK_EVAL_STORAGE_URI_ENV = 'ADK_EVAL_STORAGE_URI'
ADK_EVAL_STORAGE_DIR_ENV = 'ADK_EVAL_STORAGE_DIR'


class GcsEvalManagers(BaseModel):
  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
      arbitrary_types_allowed=True,
  )

  eval_sets_manager: 'GcsEvalSetsManager'

  eval_set_results_manager: 'GcsEvalSetResultsManager'


class ResolvedEvalStorage(NamedTuple):
  """Where eval sets and results should be stored.

  Attributes:
    gcs_uri: ``gs://`` URI when using Cloud Storage, otherwise None.
    local_dir: Directory for the local eval managers. Unused when ``gcs_uri``
      is set.
  """

  gcs_uri: str | None
  local_dir: str


def resolve_eval_storage_uri(eval_storage_uri: str | None) -> str | None:
  """Resolves the eval storage URI from an argument or the environment.

  Precedence is the explicit ``eval_storage_uri``, then
  ``ADK_EVAL_STORAGE_URI``, then ``ADK_EVAL_STORAGE_DIR`` converted to a
  ``file://`` URI.

  Args:
    eval_storage_uri: Explicit URI from a flag or ``get_fast_api_app``.

  Returns:
    The URI to use, or None to store evals under ``agents_dir``.
  """
  if eval_storage_uri:
    return eval_storage_uri

  env_uri = os.environ.get(ADK_EVAL_STORAGE_URI_ENV)
  if env_uri:
    logger.info(
        'Using eval storage URI from %s: %s', ADK_EVAL_STORAGE_URI_ENV, env_uri
    )
    return env_uri

  env_dir = os.environ.get(ADK_EVAL_STORAGE_DIR_ENV)
  if env_dir:
    file_uri = Path(os.path.abspath(os.path.expanduser(env_dir))).as_uri()
    logger.info(
        'Using eval storage directory from %s: %s',
        ADK_EVAL_STORAGE_DIR_ENV,
        env_dir,
    )
    return file_uri

  return None


def local_path_from_file_uri(eval_storage_uri: str) -> str:
  """Returns the filesystem path for a ``file://`` eval storage URI.

  Args:
    eval_storage_uri: A ``file://`` URI pointing at a local directory.

  Returns:
    The decoded filesystem path.

  Raises:
    ValueError: If the URI is not a ``file://`` URI.
  """
  parsed = urlparse(eval_storage_uri)
  if parsed.scheme != 'file':
    raise ValueError(
        f'Unsupported evals storage URI: {eval_storage_uri}. Supported URIs:'
        ' gs://<bucket name>, file://<path>'
    )

  path = unquote(parsed.path)
  if os.name == 'nt':
    if parsed.netloc and parsed.netloc.lower() != 'localhost':
      return '\\\\' + parsed.netloc + path.replace('/', '\\')
    if path.startswith('/') and len(path) >= 3 and path[2] == ':':
      path = path[1:]
    return path.replace('/', '\\')

  if parsed.netloc and parsed.netloc.lower() != 'localhost':
    return '//' + parsed.netloc + path
  return path


def prepare_local_eval_dir(path: str) -> str:
  """Creates ``path`` if needed and returns it."""
  os.makedirs(path, exist_ok=True)
  return path


def resolve_eval_storage(
    eval_storage_uri: str | None, agents_dir: str
) -> ResolvedEvalStorage:
  """Resolves GCS vs local eval storage from a URI, env vars, or agents_dir.

  Args:
    eval_storage_uri: Explicit URI from a flag or ``get_fast_api_app``.
    agents_dir: Fallback directory when no URI or env override is set.

  Returns:
    A ``ResolvedEvalStorage`` with either a GCS URI or a local directory.

  Raises:
    ValueError: If the resolved URI is neither ``gs://`` nor ``file://``.
  """
  resolved_uri = resolve_eval_storage_uri(eval_storage_uri)
  if not resolved_uri:
    return ResolvedEvalStorage(gcs_uri=None, local_dir=agents_dir)
  if resolved_uri.startswith('gs://'):
    return ResolvedEvalStorage(gcs_uri=resolved_uri, local_dir=agents_dir)
  if resolved_uri.startswith('file:'):
    local_dir = prepare_local_eval_dir(local_path_from_file_uri(resolved_uri))
    return ResolvedEvalStorage(gcs_uri=None, local_dir=local_dir)
  raise ValueError(
      f'Unsupported evals storage URI: {resolved_uri}. Supported URIs:'
      ' gs://<bucket name>, file://<path>'
  )


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
          'GCS evaluation managers require Google Cloud optional'
          ' dependencies.\nPlease install them using: pip install'
          ' google-adk[gcp]\nOr: pip install google-cloud-storage>=2.18'
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
