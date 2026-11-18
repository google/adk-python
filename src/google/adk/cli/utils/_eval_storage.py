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
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import url2pathname

from . import evals

if TYPE_CHECKING:
  from ...evaluation import local_eval_set_results_manager
  from ...evaluation.eval_set_results_manager import EvalSetResultsManager
  from ...evaluation.eval_sets_manager import EvalSetsManager
  from ...evaluation.local_eval_sets_manager import LocalEvalSetsManager

logger = logging.getLogger('google_adk.' + __name__)

ADK_EVAL_STORAGE_URI_ENV = 'ADK_EVAL_STORAGE_URI'


def _resolve_eval_storage_uri(eval_storage_uri: str | None) -> str | None:
  """Resolves the eval storage URI from an argument or the environment.

  Precedence is the explicit ``eval_storage_uri``, then
  ``ADK_EVAL_STORAGE_URI``.

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

  return None


def _local_eval_dir_from_uri(eval_storage_uri: str) -> str:
  """Returns and creates the filesystem directory for a ``file://`` URI.

  Args:
    eval_storage_uri: A ``file://`` URI pointing at a local directory.

  Returns:
    The decoded filesystem path.

  Raises:
    ValueError: If the URI is not a local ``file://`` URI with an absolute
      path, or if the directory cannot be created.
  """
  parsed = urlparse(eval_storage_uri)
  if parsed.scheme != 'file':
    raise ValueError(
        f'Unsupported evals storage URI: {eval_storage_uri}. Supported URIs:'
        ' gs://<bucket name>, file://<path>'
    )
  if parsed.netloc.lower() not in ('', 'localhost'):
    raise ValueError(
        f'Invalid evals storage URI: {eval_storage_uri}. file:// URIs must'
        ' target the local host, e.g. file:///abs/path'
    )
  if not parsed.path.startswith('/'):
    raise ValueError(
        f'Invalid evals storage URI: {eval_storage_uri}. file:// URIs must use'
        ' an absolute path, e.g. file:///abs/path'
    )
  path = url2pathname(parsed.path)
  try:
    os.makedirs(path, exist_ok=True)
  except OSError as exc:
    raise ValueError(
        f'Failed to create eval storage directory {path!r} for'
        f' {eval_storage_uri!r} (configured via --eval_storage_uri or'
        f' {ADK_EVAL_STORAGE_URI_ENV}): {exc}'
    ) from exc
  return path


def create_eval_managers(
    eval_storage_uri: str | None,
    agents_dir: str,
    *,
    local_eval_sets_manager_cls: type[LocalEvalSetsManager] | None = None,
    local_eval_set_results_manager_cls: (
        type[local_eval_set_results_manager.LocalEvalSetResultsManager] | None
    ) = None,
) -> tuple[EvalSetsManager, EvalSetResultsManager]:
  """Returns eval managers for the URI flag, env var, or agents_dir."""
  uri = _resolve_eval_storage_uri(eval_storage_uri)
  if uri and urlparse(uri).scheme == 'gs':
    try:
      gcs = evals.create_gcs_eval_managers_from_uri(uri)
    except KeyError as exc:
      raise ValueError(
          f'Missing required environment variable {exc} for GCS eval storage'
          f' URI {uri!r} (configured via --eval_storage_uri or'
          f' {ADK_EVAL_STORAGE_URI_ENV})'
      ) from exc
    return gcs.eval_sets_manager, gcs.eval_set_results_manager
  local_dir = _local_eval_dir_from_uri(uri) if uri else agents_dir
  if local_eval_sets_manager_cls is None:
    from ...evaluation.local_eval_sets_manager import LocalEvalSetsManager

    local_eval_sets_manager_cls = LocalEvalSetsManager
  if local_eval_set_results_manager_cls is None:
    from ...evaluation import local_eval_set_results_manager

    local_eval_set_results_manager_cls = (
        local_eval_set_results_manager.LocalEvalSetResultsManager
    )
  return (
      local_eval_sets_manager_cls(agents_dir=local_dir),
      local_eval_set_results_manager_cls(agents_dir=local_dir),
  )
